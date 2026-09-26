import { test } from "node:test";
import assert from "node:assert/strict";
import { Roaring32 } from "../index";
import { mulberry32, randInt, sorted, naiveRank } from "./helpers";

test("empty set", () => {
  const s = new Roaring32();
  assert.equal(s.size, 0);
  assert.equal(s.isEmpty, true);
  assert.deepEqual(s.toArray(), []);
  assert.equal(s.rank(0), 0);
  assert.equal(s.rank(0xffffffff), 0);
  assert.throws(() => s.select(0), RangeError);
  assert.throws(() => s.min(), RangeError);
  assert.throws(() => s.max(), RangeError);
});

test("0 and 2^32 - 1 boundaries", () => {
  const s = new Roaring32([0, 0xffffffff]);
  assert.equal(s.size, 2);
  assert.ok(s.has(0));
  assert.ok(s.has(0xffffffff));
  assert.ok(!s.has(1));
  assert.ok(!s.has(0xfffffffe));
  assert.equal(s.min(), 0);
  assert.equal(s.max(), 0xffffffff);
  assert.equal(s.select(0), 0);
  assert.equal(s.select(1), 0xffffffff);
  assert.equal(s.rank(0), 1);
  assert.equal(s.rank(0xfffffffe), 1);
  assert.equal(s.rank(0xffffffff), 2);
  assert.deepEqual(s.toArray(), [0, 0xffffffff]);

  s.delete(0);
  assert.equal(s.size, 1);
  assert.ok(!s.has(0));
  assert.deepEqual(s.toArray(), [0xffffffff]);
  s.delete(0xffffffff);
  assert.equal(s.size, 0);
  assert.equal(s.shardCount, 0);
});

test("rejects non-uint32 inputs", () => {
  const s = new Roaring32();
  for (const v of [-1, 0x100000000, 1.5, Number.NaN, Infinity]) {
    assert.throws(() => s.add(v), RangeError);
    assert.throws(() => s.has(v), RangeError);
    assert.throws(() => s.rank(v), RangeError);
  }
  assert.throws(() => s.addRange(10, 5), RangeError);
});

test("add is idempotent and reports presence", () => {
  const s = new Roaring32();
  assert.equal(s.add(42), true);
  assert.equal(s.add(42), false);
  assert.equal(s.add(43), true);
  assert.equal(s.size, 2);
});

test("delete on missing values and empty shards", () => {
  const s = new Roaring32([100, 65536]);
  assert.equal(s.delete(101), false);
  assert.equal(s.delete(65537), false);
  assert.equal(s.delete(100), true);
  assert.equal(s.shardCount, 1);
  assert.equal(s.delete(65536), true);
  assert.equal(s.shardCount, 0);
  assert.equal(s.size, 0);
});

test("rank/select differential against naive set", () => {
  const rng = mulberry32(42);
  const values = new Set<number>();
  while (values.size < 3000) values.add(randInt(rng, 0, 0xffffffff));
  const arr = sorted(values);
  const s = Roaring32.from(values);

  assert.equal(s.size, arr.length);
  for (const i of arr) {
    assert.ok(s.has(i));
    assert.equal(s.select(naiveRank(arr, i) - 1), i);
  }
  for (let q = 0; q < 2000; q++) {
    const v = randInt(rng, 0, 0xffffffff);
    assert.equal(s.rank(v), naiveRank(arr, v), `rank mismatch at ${v}`);
  }
  for (let i = 0; i < arr.length; i += 7) assert.equal(s.select(i), arr[i]);
});

test("rank counts values strictly within shard prefixes", () => {
  const s = new Roaring32();
  s.addRange(0, 70000); // full first shard + 4465 values of second shard
  assert.equal(s.rank(65535), 65536);
  assert.equal(s.rank(65536), 65537);
  assert.equal(s.rank(70000), 70001);
  assert.equal(s.rank(0xffff0000 - 1), 70001);
});

test("addRange and deleteRange across shards", () => {
  const s = new Roaring32();
  s.addRange(65530, 65542);
  assert.deepEqual(
    s.toArray(),
    [65530, 65531, 65532, 65533, 65534, 65535, 65536, 65537, 65538, 65539, 65540, 65541, 65542],
  );
  s.deleteRange(65532, 65538);
  assert.deepEqual(s.toArray(), [65530, 65531, 65539, 65540, 65541, 65542]);

  s.addRange(0, 0xffffffff); // the whole universe
  assert.equal(s.size, 0x100000000);
  assert.ok(s.has(0) && s.has(0xffffffff));
  assert.equal(s.rank(0xffffffff), 0x100000000);
  assert.equal(s.select(0), 0);
  assert.equal(s.select(0xffffffff), 0xffffffff);
  s.deleteRange(0, 0xffffffff);
  assert.equal(s.size, 0);
});

test("range iteration bounds across shards", () => {
  const s = Roaring32.from([0, 1, 100, 65535, 65536, 70000, 0xffffffff, 0xfffffffe]);
  assert.deepEqual(Array.from(s.range(0, 1)), [0, 1]);
  assert.deepEqual(Array.from(s.range(100, 65536)), [100, 65535, 65536]);
  assert.deepEqual(Array.from(s.range(65537, 0xffffffff)), [70000, 0xfffffffe, 0xffffffff]);
  assert.deepEqual(Array.from(s.range(50, 60)), []);
  assert.deepEqual(Array.from(s.range(70001, 0xfffffffd)), []);
});

test("range iteration over a long contiguous interval", () => {
  const s = new Roaring32();
  s.addRange(1000, 200000);
  let count = 0;
  let expected = 5000;
  for (const v of s.range(5000, 90000)) {
    assert.equal(v, expected);
    expected++;
    count++;
  }
  assert.equal(count, 85001);
  assert.equal(expected, 90001);
});

test("iterator releases temporary buffer on early cancellation", () => {
  const s = new Roaring32();
  // Force a bitmap container in shard 0.
  for (let i = 0; i < 5000; i++) s.add(i * 3);
  assert.equal(s.stats().bitmap, 1);
  assert.equal(s.pendingIteratorBuffers, 0);

  let seen = 0;
  for (const _v of s.range(0, 0xffff)) {
    if (++seen === 10) break; // caller cancels
  }
  assert.equal(seen, 10);
  assert.equal(s.pendingIteratorBuffers, 0);

  // Exception thrown by the consumer must also release the buffer.
  assert.throws(() => {
    for (const _v of s.range(0, 0xffff)) throw new Error("consumer failure");
  }, /consumer failure/);
  assert.equal(s.pendingIteratorBuffers, 0);

  // Full iteration releases on normal completion too.
  let total = 0;
  for (const _v of s.range(0, 0xffff)) total++;
  assert.equal(total, 5000);
  assert.equal(s.pendingIteratorBuffers, 0);
});

test("full iteration is sorted", () => {
  const rng = mulberry32(7);
  const values = new Set<number>();
  while (values.size < 10000) values.add(randInt(rng, 0, 0xffffffff));
  const s = Roaring32.from(values);
  const expected = sorted(values);
  assert.deepEqual(s.toArray(), expected);
});
