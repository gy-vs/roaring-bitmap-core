import { test } from "node:test";
import assert from "node:assert/strict";
import { Roaring32 } from "../index";
import {
  mulberry32,
  randInt,
  sorted,
  naiveUnion,
  naiveIntersect,
  naiveDifference,
  naiveXor,
} from "./helpers";

/** Build a set whose shard 0 is an array container. */
function makeArraySet(): { s: Roaring32; ref: Set<number> } {
  const ref = new Set([1, 2, 3, 100, 65535, 65536 + 7, 200000]);
  return { s: Roaring32.from(ref), ref };
}

/** Build a set whose shard 0 is a bitmap container (scattered, no long runs). */
function makeBitmapSet(key = 0): { s: Roaring32; ref: Set<number> } {
  const ref = new Set<number>();
  for (let i = 0; i < 5000; i++) ref.add(key * 65536 + i * 9); // 0..44991, step 9
  return { s: Roaring32.from(ref), ref };
}

/** Build a set whose shard 0 is a run container. */
function makeRunSet(): { s: Roaring32; ref: Set<number> } {
  const ref = new Set<number>();
  for (let v = 10; v <= 20000; v++) ref.add(v);
  for (let v = 30000; v <= 40000; v++) ref.add(v);
  return { s: Roaring32.from(ref), ref };
}

function checkOps(a: Roaring32, ra: Set<number>, b: Roaring32, rb: Set<number>): void {
  assert.deepEqual(a.union(b).toArray(), sorted(naiveUnion(ra, rb)), "union");
  assert.deepEqual(a.intersect(b).toArray(), sorted(naiveIntersect(ra, rb)), "intersect");
  assert.deepEqual(a.difference(b).toArray(), sorted(naiveDifference(ra, rb)), "difference");
  assert.deepEqual(a.xor(b).toArray(), sorted(naiveXor(ra, rb)), "xor");
}

test("set operations across container-type combinations", () => {
  const arr1 = makeArraySet();
  const arr2: { s: Roaring32; ref: Set<number> } = {
    s: Roaring32.from([3, 4, 100, 101, 65536 + 7, 300000]),
    ref: new Set([3, 4, 100, 101, 65536 + 7, 300000]),
  };
  const bmp1 = makeBitmapSet();
  const bmp2ref = new Set<number>();
  for (let i = 2500; i < 7500; i++) bmp2ref.add(i * 9); // overlaps bmp1 halfway
  const bmp2 = { s: Roaring32.from(bmp2ref), ref: bmp2ref };
  const run1 = makeRunSet();
  const run2ref = new Set<number>();
  for (let v = 15000; v <= 35000; v++) run2ref.add(v);
  const run2 = { s: Roaring32.from(run2ref), ref: run2ref };

  const groups = [arr1, arr2, bmp1, bmp2, run1, run2];
  for (const x of groups) {
    for (const y of groups) {
      checkOps(x.s, x.ref, y.s, y.ref);
    }
  }
});

test("union of two run sets stays run-encoded", () => {
  const a = new Roaring32();
  a.addRange(10, 5000);
  const b = new Roaring32();
  b.addRange(4000, 9000);
  const u = a.union(b);
  assert.equal(u.stats().run, 1);
  assert.equal(u.stats().array, 0);
  assert.equal(u.stats().bitmap, 0);
  assert.equal(u.size, 8991);
});

test("bitmap intersection collapses to a smaller encoding", () => {
  const a = makeBitmapSet();
  const bref = new Set<number>();
  for (let i = 0; i < 5000; i += 2) bref.add(i * 9); // half of a's values
  const b = Roaring32.from(bref);
  const inter = a.s.intersect(b);
  assert.equal(inter.size, 2500);
  assert.equal(inter.stats().array, 1);
});

test("in-place operations match their functional counterparts", () => {
  const rng = mulberry32(9);
  const mk = () => {
    const s = new Roaring32();
    for (let i = 0; i < 3000; i++) s.add(randInt(rng, 0, 200000));
    s.addRange(50000, 90000);
    for (let i = 0; i < 5000; i++) s.add(100000 + i * 7);
    return s;
  };
  const a = mk();
  const b = mk();
  const aBefore = a.toArray();
  const bBefore = b.toArray();

  assert.ok(a.clone().addAll(b).equals(a.union(b)));
  assert.ok(a.clone().intersectAll(b).equals(a.intersect(b)));
  assert.ok(a.clone().deleteAll(b).equals(a.difference(b)));
  assert.ok(a.clone().xorAll(b).equals(a.xor(b)));

  // The functional forms must not have mutated their inputs.
  assert.deepEqual(a.toArray(), aBefore);
  assert.deepEqual(b.toArray(), bBefore);

  // In-place results are exact, not just equal-up-to-encoding.
  const u = a.clone().addAll(b);
  assert.deepEqual(u.toArray(), a.union(b).toArray());
});

test("addAll/deleteAll accept plain iterables", () => {
  const s = new Roaring32([1, 2, 3]);
  s.addAll([4, 5, 6]);
  assert.deepEqual(s.toArray(), [1, 2, 3, 4, 5, 6]);
  s.deleteAll(new Set([2, 4, 6]));
  assert.deepEqual(s.toArray(), [1, 3, 5]);
});

test("sparse intersection over the full 32-bit domain", () => {
  const rng = mulberry32(2024);
  const a = new Roaring32();
  const b = new Roaring32();
  const ra = new Set<number>();
  const rb = new Set<number>();
  for (let i = 0; i < 100000; i++) {
    const v = randInt(rng, 0, 0xffffffff);
    a.add(v);
    ra.add(v);
    const w = randInt(rng, 0, 0xffffffff);
    b.add(w);
    rb.add(w);
  }
  const inter = a.intersect(b);
  const expected = sorted([...ra].filter((v) => rb.has(v)));
  assert.ok(expected.length < 100, `expected a tiny intersection, got ${expected.length}`);
  assert.deepEqual(inter.toArray(), expected);
});

test("operations with the empty set and self", () => {
  const a = Roaring32.from([1, 2, 3, 70000]);
  const empty = new Roaring32();
  assert.ok(a.union(empty).equals(a));
  assert.equal(a.intersect(empty).size, 0);
  assert.ok(a.difference(empty).equals(a));
  assert.ok(a.xor(empty).equals(a));
  assert.ok(a.union(a).equals(a));
  assert.ok(a.intersect(a).equals(a));
  assert.equal(a.difference(a).size, 0);
  assert.equal(a.xor(a).size, 0);
  // in-place self operations
  assert.ok(a.clone().addAll(a).equals(a));
  assert.equal(a.clone().deleteAll(a).size, 0);
  assert.equal(a.clone().xorAll(a).size, 0);
});

test("equals is logical, not encoding-dependent", () => {
  const viaRange = new Roaring32();
  viaRange.addRange(0, 4999); // run container
  const viaPoints = new Roaring32();
  for (let i = 0; i < 5000; i++) viaPoints.add(i); // bitmap container
  assert.ok(viaRange.equals(viaPoints));
  assert.ok(viaPoints.equals(viaRange));
});

test("differential fuzz of set operations", () => {
  const rng = mulberry32(31337);
  for (let trial = 0; trial < 30; trial++) {
    const mk = () => {
      const ref = new Set<number>();
      const s = new Roaring32();
      const n = randInt(rng, 0, 3000);
      for (let i = 0; i < n; i++) {
        if (rng() < 0.3) {
          const start = randInt(rng, 0, 100000);
          const end = Math.min(100000, start + randInt(rng, 0, 3000));
          s.addRange(start, end); // run-encoded shards
          for (let v = start; v <= end; v++) ref.add(v);
        } else {
          const v = randInt(rng, 0, 100000);
          s.add(v);
          ref.add(v);
        }
      }
      return { s, ref };
    };
    const a = mk();
    const b = mk();
    checkOps(a.s, a.ref, b.s, b.ref);
  }
});
