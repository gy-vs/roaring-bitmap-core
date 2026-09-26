import { test } from "node:test";
import assert from "node:assert/strict";
import { Roaring32, ArrayContainer, BitmapContainer, RunContainer, ARRAY_LIMIT } from "../index";
import type { Container } from "../containers";
import { mulberry32, randInt } from "./helpers";

// Reach into the shard map without widening the public API surface.
function probeContainer(s: Roaring32, key = 0): Container {
  const shards = (s as unknown as { shards: Map<number, Container> }).shards;
  return shards.get(key)!;
}

test("array -> bitmap at ARRAY_LIMIT + 1", () => {
  const s = new Roaring32();
  for (let i = 0; i <= ARRAY_LIMIT; i++) s.add(i);
  const c = probeContainer(s, 0);
  assert.ok(c instanceof BitmapContainer, `expected bitmap, got type ${c.type}`);
  assert.equal(c.cardinality, ARRAY_LIMIT + 1);
});

test("bitmap -> array when cardinality drops back to the threshold", () => {
  const s = new Roaring32();
  for (let i = 0; i <= ARRAY_LIMIT; i++) s.add(i);
  assert.ok(probeContainer(s, 0) instanceof BitmapContainer);
  s.delete(0); // ARRAY_LIMIT values remain
  const c = probeContainer(s, 0);
  assert.ok(c instanceof ArrayContainer, `expected array, got type ${c.type}`);
  assert.equal(c.cardinality, ARRAY_LIMIT);
  assert.ok(!s.has(0));
  assert.ok(s.has(1) && s.has(ARRAY_LIMIT));
});

test("dense range selects run encoding, deletions pull it back to array", () => {
  const s = new Roaring32();
  s.addRange(0, 4100); // 4101 values, one contiguous run
  let c = probeContainer(s, 0);
  assert.ok(c instanceof RunContainer, `expected run, got type ${c.type}`);
  assert.equal(c.cardinality, 4101);

  for (let v = 0; v <= 4100; v += 2) s.delete(v); // remove the 2051 even values
  c = probeContainer(s, 0);
  assert.equal(c.cardinality, 2050);
  assert.ok(c instanceof ArrayContainer, `expected array, got type ${c.type}`);
  for (let v = 1; v <= 4099; v += 2) assert.ok(s.has(v));
  assert.ok(!s.has(0) && !s.has(4100));
});

test("fragmented run converts to bitmap once its encoding grows past the bitmap", () => {
  const s = new Roaring32();
  s.addRange(0, 65535);
  assert.ok(probeContainer(s, 0) instanceof RunContainer);
  // Delete all even values: 32768 singleton runs, far larger than an 8KiB bitmap.
  for (let v = 0; v <= 65534; v += 2) s.delete(v);
  const c = probeContainer(s, 0);
  assert.ok(c instanceof BitmapContainer, `expected bitmap, got type ${c.type}`);
  assert.equal(c.cardinality, 32768);
  for (let v = 1; v <= 65535; v += 2) assert.ok(s.has(v));
  assert.ok(!s.has(0) && !s.has(65534));
});

test("deleting scattered points from a run keeps it a run while it stays dense", () => {
  const s = new Roaring32();
  s.addRange(1000, 5200); // 4201 values, one run
  assert.ok(probeContainer(s, 0) instanceof RunContainer);
  for (let v = 1100; v <= 5200; v += 100) s.delete(v); // 42 deletions
  const c = probeContainer(s, 0);
  assert.ok(c instanceof RunContainer, `expected run, got type ${c.type}`);
  assert.equal(c.cardinality, 4201 - 42);
  assert.equal(s.size, 4201 - 42);
});

test("sparse deletions fragment a run back into an array under the threshold", () => {
  const s = new Roaring32();
  s.addRange(0, 5000); // 5001 values -> run
  assert.ok(probeContainer(s, 0) instanceof RunContainer);
  for (let v = 0; v <= 5000; v += 3) s.delete(v); // 1667 deletions -> 3334 values in pairs
  const c = probeContainer(s, 0);
  assert.ok(c instanceof ArrayContainer, `expected array, got type ${c.type}`);
  assert.equal(c.cardinality, 3334);
  const values = s.toArray();
  assert.equal(values.length, 3334);
  for (let i = 1; i < values.length; i++) assert.ok(values[i - 1] < values[i]);
});

test("conversions preserve sorted order and cardinality (random walk)", () => {
  const rng = mulberry32(123);
  const ref = new Set<number>();
  const s = new Roaring32();
  for (let step = 0; step < 20000; step++) {
    const v = randInt(rng, 0, 0xffff); // single shard: every encoding gets exercised
    if (rng() < 0.55) {
      ref.add(v);
      s.add(v);
    } else {
      ref.delete(v);
      s.delete(v);
    }
  }
  assert.equal(s.size, ref.size);
  assert.deepEqual(s.toArray(), Array.from(ref).sort((a, b) => a - b));
  assert.equal(probeContainer(s, 0).cardinality, ref.size);
});

test("long contiguous interval stays compressed as runs across many shards", () => {
  const s = new Roaring32();
  s.addRange(0xfff0, 0x50000010);
  const st = s.stats();
  assert.equal(st.bitmap, 0);
  assert.equal(st.array, 2); // two partial boundary shards
  assert.equal(st.run, 0x5000 - 1); // fully covered inner shards
  assert.equal(s.size, 0x50000010 - 0xfff0 + 1);
  assert.ok(s.has(0x12345678));
  assert.ok(!s.has(0x50000011));
});

test("point add/delete on run boundaries merges and splits runs", () => {
  const s = new Roaring32();
  s.addRange(10, 5000); // 4991 values -> run
  assert.ok(probeContainer(s, 0) instanceof RunContainer);
  s.add(5001); // extends the run upward
  s.add(9); // extends the run downward
  assert.equal(s.size, 4993);
  assert.ok(s.has(9) && s.has(5001));
  assert.equal((probeContainer(s, 0) as RunContainer).runs.length, 1);
  s.delete(100); // splits into [9,99] + [101,5001]
  assert.equal((probeContainer(s, 0) as RunContainer).runs.length, 2);
  assert.ok(!s.has(100));
  assert.equal(s.size, 4992);
  s.add(100); // merges them back
  assert.equal((probeContainer(s, 0) as RunContainer).runs.length, 1);
  assert.equal(s.size, 4993);
});

test("optimize() converts dense bitmaps to runs and sparse runs to arrays", () => {
  const s = new Roaring32();
  for (let i = 0; i < 5000; i++) s.add(i); // contiguous points -> bitmap
  assert.ok(probeContainer(s, 0) instanceof BitmapContainer);
  s.optimize();
  assert.ok(probeContainer(s, 0) instanceof RunContainer);
  assert.equal(s.size, 5000);
  assert.deepEqual(s.toArray(), Array.from({ length: 5000 }, (_, i) => i));
});
