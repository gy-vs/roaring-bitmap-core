import { test } from "node:test";
import assert from "node:assert/strict";
import { Roaring32 } from "../index";
import { mulberry32, randInt, sorted, naiveRank } from "./helpers";

test("one million contiguous elements", () => {
  const s = new Roaring32();
  s.addRange(0, 999_999);
  assert.equal(s.size, 1_000_000);
  assert.equal(s.rank(0), 1);
  assert.equal(s.rank(999_999), 1_000_000);
  assert.equal(s.rank(1_000_000), 1_000_000);
  assert.equal(s.select(0), 0);
  assert.equal(s.select(999_999), 999_999);
  assert.equal(s.min(), 0);
  assert.equal(s.max(), 999_999);
  let count = 0;
  for (const _ of s) count++;
  assert.equal(count, 1_000_000);
  // delete half, verify cardinality and membership
  s.deleteRange(250_000, 749_999);
  assert.equal(s.size, 500_000);
  assert.ok(!s.has(250_000) && !s.has(749_999));
  assert.ok(s.has(249_999) && s.has(750_000));
  assert.equal(s.rank(249_999), 250_000);
  assert.equal(s.rank(750_000), 250_001);
});

test("one million scattered elements: differential rank/select", () => {
  const rng = mulberry32(0xc0ffee);
  const s = new Roaring32();
  const ref = new Set<number>();
  // 700k strided points across the whole domain + 300k uniform random
  for (let i = 0; i < 700_000; i++) {
    const v = (i * 6113) % 0x100000000; // 6113 is odd, so all values are distinct
    s.add(v);
    ref.add(v);
  }
  for (let i = 0; i < 300_000; i++) {
    const v = randInt(rng, 0, 0xffffffff);
    s.add(v);
    ref.add(v);
  }
  assert.equal(s.size, ref.size);
  assert.ok(ref.size >= 999_000, `expected about a million elements, got ${ref.size}`);

  const arr = sorted(ref);
  for (let q = 0; q < 2000; q++) {
    const v = randInt(rng, 0, 0xffffffff);
    assert.equal(s.rank(v), naiveRank(arr, v), `rank(${v})`);
  }
  for (let q = 0; q < 2000; q++) {
    const i = randInt(rng, 0, arr.length - 1);
    assert.equal(s.select(i), arr[i], `select(${i})`);
  }
  // full ordered scan matches the naive sorted array
  assert.deepEqual(s.toArray(), arr);
});

test("set operations on million-element sets", () => {
  const a = new Roaring32();
  a.addRange(0, 999_999);
  const b = new Roaring32();
  b.addRange(500_000, 1_499_999);

  const u = a.union(b);
  assert.equal(u.size, 1_500_000);
  assert.equal(u.min(), 0);
  assert.equal(u.max(), 1_499_999);

  const i = a.intersect(b);
  assert.equal(i.size, 500_000);
  assert.equal(i.min(), 500_000);
  assert.equal(i.max(), 999_999);

  const d = a.difference(b);
  assert.equal(d.size, 500_000);
  assert.equal(d.max(), 499_999);

  const x = a.xor(b);
  assert.equal(x.size, 1_000_000);
  assert.ok(!x.has(500_000) && !x.has(999_999));
  assert.ok(x.has(499_999) && x.has(1_000_000));

  // serialization round-trip at this scale
  const back = Roaring32.deserialize(u.serialize());
  assert.ok(back.equals(u));
});

test("random operation sequence matches a naive set", () => {
  const rng = mulberry32(2026);
  const s = new Roaring32();
  const ref = new Set<number>();
  const SPACE = 300_000;
  for (let step = 0; step < 4000; step++) {
    const op = rng();
    if (op < 0.3) {
      const v = randInt(rng, 0, SPACE);
      s.add(v);
      ref.add(v);
    } else if (op < 0.5) {
      const v = randInt(rng, 0, SPACE);
      s.delete(v);
      ref.delete(v);
    } else if (op < 0.75) {
      const a = randInt(rng, 0, SPACE);
      const b = Math.min(SPACE, a + randInt(rng, 0, 400));
      s.addRange(a, b);
      for (let v = a; v <= b; v++) ref.add(v);
    } else if (op < 0.9) {
      const a = randInt(rng, 0, SPACE);
      const b = Math.min(SPACE, a + randInt(rng, 0, 400));
      s.deleteRange(a, b);
      for (let v = a; v <= b; v++) ref.delete(v);
    } else {
      // random range query check
      const a = randInt(rng, 0, SPACE);
      const b = Math.min(SPACE, a + randInt(rng, 0, 2000));
      const expected = sorted([...ref].filter((v) => v >= a && v <= b));
      assert.deepEqual(Array.from(s.range(a, b)), expected, `range(${a}, ${b}) at step ${step}`);
    }
  }
  const arr = sorted(ref);
  assert.equal(s.size, arr.length);
  assert.deepEqual(s.toArray(), arr);
  for (let q = 0; q < 1000; q++) {
    const v = randInt(rng, 0, SPACE);
    assert.equal(s.rank(v), naiveRank(arr, v));
    assert.equal(s.has(v), ref.has(v));
  }
  // serialize the final state and verify it still matches
  const back = Roaring32.deserialize(s.serialize());
  assert.deepEqual(back.toArray(), arr);
});

test("in-place bulk operations on large sets", () => {
  const rng = mulberry32(555);
  const a = new Roaring32();
  const b = new Roaring32();
  const ra = new Set<number>();
  const rb = new Set<number>();
  for (let i = 0; i < 200_000; i++) {
    const v = randInt(rng, 0, 2_000_000);
    a.add(v);
    ra.add(v);
    if (rng() < 0.5) {
      b.add(v);
      rb.add(v);
    }
  }
  b.addRange(1_000_000, 1_100_000);
  for (let v = 1_000_000; v <= 1_100_000; v++) rb.add(v);

  const beforeA = a.toArray();
  const inter = a.clone().intersectAll(b);
  assert.deepEqual(inter.toArray(), sorted([...ra].filter((v) => rb.has(v))));
  // a is untouched because we operated on a clone
  assert.deepEqual(a.toArray(), beforeA);

  const removed = a.clone().deleteAll(b);
  assert.deepEqual(removed.toArray(), sorted([...ra].filter((v) => !rb.has(v))));

  const xored = a.clone().xorAll(b);
  const expectedXor = new Set<number>();
  for (const v of ra) if (!rb.has(v)) expectedXor.add(v);
  for (const v of rb) if (!ra.has(v)) expectedXor.add(v);
  assert.deepEqual(xored.toArray(), sorted(expectedXor));
});
