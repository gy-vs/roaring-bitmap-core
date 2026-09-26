import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RoaringSet, DeserializationError, wordPool, MAX_U32 } from '../src/index';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Deterministic PRNG (mulberry32) so failures are reproducible. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randUint32(rng: () => number): number {
  return Math.floor(rng() * 0x100000000);
}

/** Naive model: a plain sorted unique array. */
class Naive {
  sorted: number[];
  constructor(values: Iterable<number>) {
    this.sorted = [...new Set(values)].sort((a, b) => a - b);
  }
  get size(): number {
    return this.sorted.length;
  }
  rank(x: number): number {
    let lo = 0;
    let hi = this.sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.sorted[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
  select(i: number): number | undefined {
    return i >= 0 && i < this.sorted.length ? this.sorted[i] : undefined;
  }
  has(x: number): boolean {
    let lo = 0;
    let hi = this.sorted.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.sorted[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo < this.sorted.length && this.sorted[lo] === x;
  }
}

/** Differentially verify a RoaringSet against the naive model. */
function checkAgainstNaive(set: RoaringSet, naive: Naive, rng: () => number): void {
  assert.equal(set.size, naive.size, 'size');
  assert.deepEqual(set.toArray(), naive.sorted, 'sorted iteration');

  if (naive.size > 0) {
    assert.equal(set.minimum(), naive.sorted[0], 'minimum');
    assert.equal(set.maximum(), naive.sorted[naive.size - 1], 'maximum');
  } else {
    assert.equal(set.minimum(), undefined);
    assert.equal(set.maximum(), undefined);
  }

  // rank probes: extremes, random values, and neighbours of present values.
  const probes = new Set<number>([0, 1, MAX_U32 - 1, MAX_U32, 0x7fffffff, 0x80000000, 0x10000, 0xffff]);
  for (let i = 0; i < 64; i++) probes.add(randUint32(rng));
  for (let i = 0; i < Math.min(naive.size, 64); i++) {
    const v = naive.sorted[Math.floor(rng() * naive.size)];
    if (v > 0) probes.add(v - 1);
    probes.add(v);
    if (v < MAX_U32) probes.add(v + 1);
  }
  for (const p of probes) {
    assert.equal(set.rank(p), naive.rank(p), `rank(${p})`);
  }

  // select probes: boundaries and random indices.
  if (naive.size > 0) {
    assert.equal(set.select(0), naive.sorted[0], 'select(0)');
    assert.equal(set.select(naive.size - 1), naive.sorted[naive.size - 1], 'select(last)');
  }
  for (let i = 0; i < Math.min(naive.size, 128); i++) {
    const idx = Math.floor(rng() * naive.size);
    assert.equal(set.select(idx), naive.sorted[idx], `select(${idx})`);
  }
  assert.equal(set.select(naive.size), undefined, 'select(size)');
  assert.equal(set.select(-1), undefined, 'select(-1)');

  // membership probes.
  for (let i = 0; i < 64; i++) {
    const p = randUint32(rng);
    assert.equal(set.has(p), naive.has(p), `has(${p})`);
  }
}

function naiveOp(a: number[], b: number[], op: 'union' | 'inter' | 'diff' | 'xor'): number[] {
  const A = new Set(a);
  const B = new Set(b);
  const out = new Set<number>();
  if (op === 'union' || op === 'xor') for (const v of B) if (!A.has(v)) out.add(v);
  if (op === 'union') for (const v of A) out.add(v);
  if (op === 'inter') for (const v of A) if (B.has(v)) out.add(v);
  if (op === 'diff') for (const v of A) if (!B.has(v)) out.add(v);
  if (op === 'xor') for (const v of A) if (!B.has(v)) out.add(v);
  return [...out].sort((x, y) => x - y);
}

/** Builds a set with a mix of container types across shards. */
function buildMixedSet(rng: () => number): { set: RoaringSet; values: number[] } {
  const values: number[] = [];
  const pickShard = () => Math.floor(rng() * 24) * 0x10000;
  // Sparse shards -> array containers.
  for (let i = 0; i < 300; i++) values.push(pickShard() + Math.floor(rng() * 0x10000));
  // Dense shards -> bitmap containers.
  for (let c = 0; c < 3; c++) {
    const base = (40 + c * 3) * 0x10000;
    for (let i = 0; i < 6000; i++) values.push(base + Math.floor(rng() * 0x10000));
  }
  // Long runs -> run containers.
  for (let c = 0; c < 4; c++) {
    const base = (80 + c * 5) * 0x10000 + Math.floor(rng() * 1000);
    const len = 500 + Math.floor(rng() * 20000);
    for (let v = base; v <= base + len && v <= MAX_U32; v++) values.push(v);
  }
  const set = RoaringSet.from(values);
  set.runOptimize();
  return { set, values };
}

// ---------------------------------------------------------------------------
// Basics, boundaries, input validation
// ---------------------------------------------------------------------------

test('add/has/remove basics and boundaries 0 and 2^32-1', () => {
  const s = new RoaringSet();
  assert.equal(s.isEmpty, true);
  assert.equal(s.add(0), true);
  assert.equal(s.add(MAX_U32), true);
  assert.equal(s.add(0), false, 'duplicate add');
  assert.equal(s.add(65536), true);
  assert.equal(s.size, 3);
  assert.equal(s.has(0), true);
  assert.equal(s.has(MAX_U32), true);
  assert.equal(s.has(1), false);
  assert.deepEqual(s.toArray(), [0, 65536, MAX_U32]);
  assert.equal(s.minimum(), 0);
  assert.equal(s.maximum(), MAX_U32);
  assert.equal(s.rank(0), 1);
  assert.equal(s.rank(65535), 1);
  assert.equal(s.rank(65536), 2);
  assert.equal(s.rank(MAX_U32), 3);
  assert.equal(s.select(0), 0);
  assert.equal(s.select(1), 65536);
  assert.equal(s.select(2), MAX_U32);
  assert.equal(s.select(3), undefined);

  assert.equal(s.remove(1), false);
  assert.equal(s.remove(0), true);
  assert.equal(s.remove(MAX_U32), true);
  assert.equal(s.size, 1);
  assert.equal(s.has(65536), true);
  s.clear();
  assert.equal(s.isEmpty, true);
  assert.equal(s.minimum(), undefined);

  assert.throws(() => s.add(-1), RangeError);
  assert.throws(() => s.add(MAX_U32 + 1), RangeError);
  assert.throws(() => s.add(1.5), RangeError);
  assert.throws(() => s.rank(NaN), RangeError);
});

test('empty set operations', () => {
  const a = new RoaringSet();
  const b = RoaringSet.of(1, 2, 3);
  assert.equal(a.union(b).size, 3);
  assert.equal(a.intersection(b).size, 0);
  assert.equal(a.difference(b).size, 0);
  assert.equal(b.difference(a).size, 3);
  assert.equal(a.symmetricDifference(b).size, 3);
  assert.equal(a.rank(MAX_U32), 0);
  assert.equal(a.select(0), undefined);
  assert.deepEqual(a.toArray(), []);
  assert.equal(a.equals(new RoaringSet()), true);
  assert.equal(a.equals(b), false);
});

// ---------------------------------------------------------------------------
// Container threshold conversions (array <-> bitmap round trips)
// ---------------------------------------------------------------------------

test('array->bitmap->array threshold round trips preserve order and cardinality', () => {
  const s = new RoaringSet();
  const naive: number[] = [];

  for (let i = 0; i <= 4096; i++) {
    s.add(i);
    naive.push(i);
    if (i < 4096) assert.deepEqual(s.debugContainerTypes(), ['array'], `type at ${i + 1} elements`);
  }
  assert.deepEqual(s.debugContainerTypes(), ['bitmap'], 'bitmap above 4096');
  assert.deepEqual(s.toArray(), naive, 'sorted after array->bitmap');
  assert.equal(s.size, 4097);

  // Cross the threshold back and forth several times.
  for (let round = 0; round < 3; round++) {
    assert.equal(s.remove(round), true);
    assert.deepEqual(s.debugContainerTypes(), ['array'], `array at 4096 (round ${round})`);
    assert.equal(s.size, 4096);
    assert.equal(s.add(round), true);
    assert.deepEqual(s.debugContainerTypes(), ['bitmap'], `bitmap at 4097 (round ${round})`);
    assert.equal(s.size, 4097);
  }

  // Remove everything one by one; cardinality and order must stay exact.
  for (let i = 0; i <= 4096; i++) {
    assert.equal(s.has(i), true, `has(${i}) before remove`);
    s.remove(i);
  }
  assert.equal(s.size, 0);
  assert.equal(s.containerCount, 0, 'empty shards are dropped');
});

test('run container chosen for dense runs, bitmap for scattered dense data', () => {
  const runs = new RoaringSet();
  runs.addRange(100, 9999);
  assert.deepEqual(runs.debugContainerTypes(), ['run']);
  assert.equal(runs.size, 9900);

  const scattered = new RoaringSet();
  for (let i = 0; i < 5000; i++) scattered.add(i * 2); // even numbers: no long runs
  assert.deepEqual(scattered.debugContainerTypes(), ['bitmap']);
  assert.equal(scattered.size, 5000);

  // A run container stays coherent under point removals that split runs.
  runs.remove(5000);
  assert.equal(runs.size, 9899);
  assert.equal(runs.has(5000), false);
  assert.equal(runs.has(4999), true);
  assert.equal(runs.has(5001), true);
  assert.equal(runs.rank(5000), 4900);
  runs.remove(100); // shrink run start
  runs.remove(9999); // shrink run end
  assert.equal(runs.has(100), false);
  assert.equal(runs.has(101), true);
  assert.equal(runs.has(9999), false);
  assert.equal(runs.has(9998), true);
  // Remove a full run (the singleton gap left at 5000 is not a run; remove 101..4999 etc.)
  const before = runs.size;
  runs.remove(5000); // no-op
  assert.equal(runs.size, before);
});

// ---------------------------------------------------------------------------
// Long contiguous ranges
// ---------------------------------------------------------------------------

test('long contiguous ranges across many shards', () => {
  const rng = mulberry32(42);
  const s = new RoaringSet();
  s.addRange(0, 199_999);
  s.addRange(5_000_000, 5_100_000);
  s.addRange(MAX_U32 - 10_000, MAX_U32); // touches the top of the space
  const values: number[] = [];
  for (let v = 0; v <= 199_999; v++) values.push(v);
  for (let v = 5_000_000; v <= 5_100_000; v++) values.push(v);
  for (let v = MAX_U32 - 10_000; v <= MAX_U32; v++) values.push(v);
  checkAgainstNaive(s, new Naive(values), rng);
  assert.ok(s.debugContainerTypes().every((t) => t === 'run'), 'all shards are runs');

  // Round-trip through serialization keeps runs and cardinality.
  const back = RoaringSet.deserialize(s.serialize());
  assert.ok(back.equals(s));
  assert.deepEqual(back.debugContainerTypes(), s.debugContainerTypes());
});

// ---------------------------------------------------------------------------
// Sparse intersection
// ---------------------------------------------------------------------------

test('sparse intersection across many shards', () => {
  const rng = mulberry32(7);
  const aVals = new Set<number>();
  const bVals = new Set<number>();
  while (aVals.size < 5000) aVals.add(randUint32(rng));
  while (bVals.size < 5000) bVals.add(randUint32(rng));
  // Force a handful of guaranteed common values.
  for (let i = 0; i < 50; i++) {
    const v = randUint32(rng);
    aVals.add(v);
    bVals.add(v);
  }
  const a = RoaringSet.from(aVals);
  const b = RoaringSet.from(bVals);
  const inter = a.intersection(b);
  const expected = naiveOp([...aVals], [...bVals], 'inter');
  assert.deepEqual(inter.toArray(), expected);
  assert.ok(inter.containerCount <= expected.length);

  // Disjoint sets -> empty intersection, no leftover empty shards.
  const c = RoaringSet.from([1, 2, 3]);
  const d = RoaringSet.from([1 << 20, 5 << 20]);
  const empty = c.intersection(d);
  assert.equal(empty.size, 0);
  assert.equal(empty.containerCount, 0);
});

// ---------------------------------------------------------------------------
// Set operations: all container combinations, in-place vs functional
// ---------------------------------------------------------------------------

test('set operations match naive model for mixed container combinations', () => {
  const rng = mulberry32(1234);
  for (let trial = 0; trial < 8; trial++) {
    const A = buildMixedSet(rng);
    const B = buildMixedSet(rng);
    const aVals = [...new Set(A.values)];
    const bVals = [...new Set(B.values)];

    const aCopy = A.set.clone();
    const bCopy = B.set.clone();

    assert.deepEqual(A.set.union(B.set).toArray(), naiveOp(aVals, bVals, 'union'), 'union');
    assert.deepEqual(A.set.intersection(B.set).toArray(), naiveOp(aVals, bVals, 'inter'), 'intersection');
    assert.deepEqual(A.set.difference(B.set).toArray(), naiveOp(aVals, bVals, 'diff'), 'difference');
    assert.deepEqual(A.set.symmetricDifference(B.set).toArray(), naiveOp(aVals, bVals, 'xor'), 'xor');

    // Non-in-place ops must not mutate their operands.
    assert.ok(A.set.equals(aCopy), 'union/intersection/... left operand unchanged');
    assert.ok(B.set.equals(bCopy), 'right operand unchanged');

    // In-place variants must produce exactly the functional results.
    const u = A.set.union(B.set);
    const i = A.set.intersection(B.set);
    const d = A.set.difference(B.set);
    const x = A.set.symmetricDifference(B.set);

    const au = aCopy.clone();
    au.unionInPlace(bCopy);
    assert.ok(au.equals(u), 'unionInPlace');
    const ai = aCopy.clone();
    ai.intersectionInPlace(bCopy);
    assert.ok(ai.equals(i), 'intersectionInPlace');
    const ad = aCopy.clone();
    ad.differenceInPlace(bCopy);
    assert.ok(ad.equals(d), 'differenceInPlace');
    const ax = aCopy.clone();
    ax.symmetricDifferenceInPlace(bCopy);
    assert.ok(ax.equals(x), 'symmetricDifferenceInPlace');

    // In-place ops must not mutate the right operand either.
    assert.ok(bCopy.equals(B.set), 'in-place ops leave right operand unchanged');
  }
});

test('self operations', () => {
  const rng = mulberry32(99);
  const { set } = buildMixedSet(rng);
  const copy = set.clone();
  assert.ok(set.union(copy).equals(set));
  assert.ok(set.intersection(copy).equals(set));
  assert.equal(set.difference(copy).size, 0);
  assert.equal(set.symmetricDifference(copy).size, 0);

  const s2 = set.clone();
  s2.unionInPlace(s2);
  assert.ok(s2.equals(set));
  const s3 = set.clone();
  s3.symmetricDifferenceInPlace(s3);
  assert.equal(s3.size, 0);
  const s4 = set.clone();
  s4.differenceInPlace(s4);
  assert.equal(s4.size, 0);
});

// ---------------------------------------------------------------------------
// rank/select differential fuzzing across densities
// ---------------------------------------------------------------------------

test('rank/select differential vs naive set across densities', () => {
  const rng = mulberry32(2024);
  const cases: number[][] = [];

  // Sparse uniform over the whole 32-bit space.
  const sparse = new Set<number>();
  while (sparse.size < 3000) sparse.add(randUint32(rng));
  cases.push([...sparse]);

  // Dense cluster inside a few shards.
  const dense: number[] = [];
  for (let shard = 100; shard < 104; shard++) {
    for (let i = 0; i < 20000; i++) dense.push(shard * 0x10000 + Math.floor(rng() * 0x10000));
  }
  cases.push(dense);

  // Long runs with gaps.
  const runny: number[] = [];
  for (let r = 0; r < 30; r++) {
    const start = randUint32(rng) % (MAX_U32 - 60000);
    const len = Math.floor(rng() * 50000);
    for (let v = start; v <= start + len; v++) runny.push(v);
  }
  cases.push(runny);

  // Boundaries and singletons.
  cases.push([0, 1, 2, 65535, 65536, 65537, MAX_U32 - 1, MAX_U32]);

  for (const values of cases) {
    const set = RoaringSet.from(values);
    set.runOptimize();
    checkAgainstNaive(set, new Naive(values), rng);
  }
});

// ---------------------------------------------------------------------------
// Range iteration and iterator cancellation
// ---------------------------------------------------------------------------

test('iterateRange matches naive slice and releases scratch buffer', () => {
  const rng = mulberry32(555);
  const { set, values } = buildMixedSet(rng);
  const naive = new Naive(values);

  const ranges: Array<[number, number]> = [
    [0, MAX_U32],
    [0, 0],
    [MAX_U32, MAX_U32],
    [100, 1_000_000],
    [40 * 0x10000 + 123, 41 * 0x10000 + 4567], // cuts through bitmap shards
    [80 * 0x10000, 90 * 0x10000], // cuts through run shards
  ];
  for (const [lo, hi] of ranges) {
    const got = [...set.iterateRange(lo, hi)];
    const expected = naive.sorted.filter((v) => v >= lo && v <= hi);
    assert.deepEqual(got, expected, `range [${lo}, ${hi}]`);
  }

  // Cancellation: breaking out must release the scratch buffer.
  const before = { a: wordPool.totalAcquired, r: wordPool.totalReleased };
  for (const _ of set.iterateRange(0, MAX_U32)) break;
  assert.equal(wordPool.totalAcquired, before.a + 1, 'scratch acquired once');
  assert.equal(wordPool.totalReleased, before.r + 1, 'scratch released on break');

  // Explicit return() on the generator also releases.
  const it = set.iterateRange(0, MAX_U32);
  it.next();
  assert.equal(wordPool.totalAcquired, before.a + 2);
  it.return!(undefined);
  assert.equal(wordPool.totalReleased, before.r + 2, 'scratch released on return()');

  // Full iteration releases too.
  for (const _ of set.iterateRange(0, MAX_U32)) {
    // exhaust
  }
  assert.equal(wordPool.totalReleased, before.r + 3, 'scratch released on completion');
});

// ---------------------------------------------------------------------------
// Serialization round trips and corrupted inputs
// ---------------------------------------------------------------------------

test('serialization round trip preserves content and container types', () => {
  const rng = mulberry32(31337);
  const sets: RoaringSet[] = [
    new RoaringSet(),
    RoaringSet.of(0, MAX_U32),
    buildMixedSet(rng).set,
  ];
  const big = new RoaringSet();
  big.addRange(0, 300_000);
  sets.push(big);

  for (const s of sets) {
    const buf = s.serialize();
    assert.equal(buf.length, s.serializedSizeInBytes());
    const back = RoaringSet.deserialize(buf);
    assert.ok(back.equals(s), 'round trip equality');
    assert.deepEqual(back.debugContainerTypes(), s.debugContainerTypes(), 'container types preserved');
    assert.equal(back.size, s.size);
    // A view into a larger buffer must decode identically.
    const padded = new Uint8Array(buf.length + 8);
    padded.set(buf, 4);
    assert.ok(RoaringSet.deserialize(padded.subarray(4, 4 + buf.length)).equals(s));
  }
});

test('deserialize rejects corrupted inputs', () => {
  const rng = mulberry32(808);
  // One set containing every container type: array, bitmap, and run shards.
  const s = new RoaringSet();
  s.addMany([1, 2, 3, 70000, 70001]); // array shard (key 0 and 1)
  for (let i = 0; i < 5000; i++) s.add(2 * 0x10000 + i * 3); // bitmap shard (key 2)
  s.addRange(3 * 0x10000 + 10, 3 * 0x10000 + 20); // run shard (key 3)
  s.addRange(3 * 0x10000 + 100, 3 * 0x10000 + 110); // second run in same shard
  const good = s.serialize();
  const n = s.containerCount;
  assert.equal(n, 4);

  const view = () => new DataView(good.slice().buffer);
  const metaEnd = 16 + n * 5;
  const offsetsPos = (metaEnd + 3) & ~3;
  const dataStart = offsetsPos + n * 4;

  const expectCorrupt = (name: string, mutate: (b: Uint8Array) => void): void => {
    const b = good.slice();
    mutate(b);
    assert.throws(() => RoaringSet.deserialize(b), DeserializationError, name);
  };

  expectCorrupt('bad magic', (b) => {
    b[0] ^= 0xff;
  });
  expectCorrupt('bad version', (b) => {
    new DataView(b.buffer).setUint32(4, 99, true);
  });
  expectCorrupt('reserved flags', (b) => {
    b[12] = 1;
  });
  expectCorrupt('container count explosion', (b) => {
    new DataView(b.buffer).setUint32(8, 0x00ffffff, true);
  });
  expectCorrupt('keys not increasing', (b) => {
    // Swap the first two shard keys.
    const k0 = new DataView(b.buffer).getUint16(16, true);
    const k1 = new DataView(b.buffer).getUint16(18, true);
    new DataView(b.buffer).setUint16(16, k1, true);
    new DataView(b.buffer).setUint16(18, k0, true);
  });
  expectCorrupt('cardinality changed', (b) => {
    const p = 16 + n * 2; // first cardinality entry
    const v = new DataView(b.buffer).getUint16(p, true);
    new DataView(b.buffer).setUint16(p, v + 1, true);
  });
  expectCorrupt('unknown container type', (b) => {
    b[16 + n * 4] = 9;
  });
  expectCorrupt('offset out of range', (b) => {
    new DataView(b.buffer).setUint32(offsetsPos, 0x7ffffff0, true);
  });
  expectCorrupt('offset shifted', (b) => {
    const v = new DataView(b.buffer).getUint32(offsetsPos + 4, true);
    new DataView(b.buffer).setUint32(offsetsPos + 4, v + 2, true);
  });
  expectCorrupt('array payload unsorted', (b) => {
    // First shard is the array container; swap its first two values.
    const v0 = new DataView(b.buffer).getUint16(dataStart, true);
    const v1 = new DataView(b.buffer).getUint16(dataStart + 2, true);
    new DataView(b.buffer).setUint16(dataStart, v1, true);
    new DataView(b.buffer).setUint16(dataStart + 2, v0, true);
  });
  expectCorrupt('bitmap popcount mismatch', (b) => {
    // Second shard (key 2) is the bitmap; flip a payload bit.
    const off = new DataView(b.buffer).getUint32(offsetsPos + 2 * 4, true);
    b[off + 100] ^= 0x01;
  });
  expectCorrupt('run overlap', (b) => {
    // Fourth shard (key 3) is the run container with two runs; move the
    // second run's start back into the first run.
    const off = new DataView(b.buffer).getUint32(offsetsPos + 3 * 4, true);
    new DataView(b.buffer).setUint16(off + 4, 12, true); // second run start := 12
  });
  expectCorrupt('run cardinality mismatch', (b) => {
    const off = new DataView(b.buffer).getUint32(offsetsPos + 3 * 4, true);
    const l = new DataView(b.buffer).getUint16(off + 2, true);
    new DataView(b.buffer).setUint16(off + 2, l + 5, true);
  });
  expectCorrupt('run extends past 65535', (b) => {
    const off = new DataView(b.buffer).getUint32(offsetsPos + 3 * 4, true);
    new DataView(b.buffer).setUint16(off + 2, 65535 - 10 + 1, true); // length makes start+length > 65535
  });

  // Truncations.
  assert.throws(() => RoaringSet.deserialize(good.subarray(0, 8)), DeserializationError, 'short header');
  assert.throws(() => RoaringSet.deserialize(good.subarray(0, good.length - 1)), DeserializationError, 'truncated payload');
  assert.throws(() => RoaringSet.deserialize(good.subarray(0, dataStart - 1)), DeserializationError, 'truncated offsets');
  assert.throws(() => RoaringSet.deserialize(new Uint8Array(0)), DeserializationError, 'empty buffer');

  // Sanity: the untouched buffer still decodes.
  assert.ok(RoaringSet.deserialize(good).equals(s));
  void view;
});

// ---------------------------------------------------------------------------
// Million-element scale
// ---------------------------------------------------------------------------

test('million elements: sequential and random', { timeout: 120_000 }, () => {
  const rng = mulberry32(1_000_000);

  // 1M sequential values via individual adds.
  const seq = new RoaringSet();
  for (let i = 0; i < 1_000_000; i++) seq.add(i);
  assert.equal(seq.size, 1_000_000);
  assert.equal(seq.has(0), true);
  assert.equal(seq.has(999_999), true);
  assert.equal(seq.has(1_000_000), false);
  assert.equal(seq.rank(500_000), 500_001);
  assert.equal(seq.select(999_999), 999_999);
  assert.equal(seq.select(1_000_000), undefined);

  // 1M random values via bulk add, deduplicated against a JS Set.
  const raw: number[] = [];
  const seen = new Set<number>();
  for (let i = 0; i < 1_000_000; i++) {
    const v = randUint32(rng);
    raw.push(v);
    seen.add(v);
  }
  const rnd = RoaringSet.from(raw);
  assert.equal(rnd.size, seen.size);
  for (let i = 0; i < 2000; i++) {
    const v = raw[Math.floor(rng() * raw.length)];
    assert.equal(rnd.has(v), true);
  }
  const naive = new Naive([...seen]);
  checkAgainstNaive(rnd, naive, rng);

  // Large-set algebra: seq = [0, 1M), evens = {2k < 2M}.
  const evens = new RoaringSet();
  evens.addMany(Array.from({ length: 1_000_000 }, (_, i) => i * 2));
  assert.equal(evens.size, 1_000_000);

  const inter = seq.intersection(evens);
  assert.equal(inter.size, 500_000);
  assert.equal(inter.has(999_998), true);
  assert.equal(inter.has(999_999), false);
  assert.equal(inter.rank(1000), 501);

  const uni = seq.union(evens);
  assert.equal(uni.size, 1_500_000);

  const diff = seq.difference(evens);
  assert.equal(diff.size, 500_000);
  assert.equal(diff.has(1), true);
  assert.equal(diff.has(2), false);

  const xor = seq.symmetricDifference(evens);
  assert.equal(xor.size, 1_000_000);

  // In-place at scale.
  const seq2 = seq.clone();
  seq2.intersectionInPlace(evens);
  assert.ok(seq2.equals(inter));

  // Serialization round trip at scale.
  const back = RoaringSet.deserialize(seq.serialize());
  assert.ok(back.equals(seq));
  const backRnd = RoaringSet.deserialize(rnd.serialize());
  assert.ok(backRnd.equals(rnd));
});

// ---------------------------------------------------------------------------
// clone / equals / shrinkToFit
// ---------------------------------------------------------------------------

test('clone is deep and shrinkToFit preserves content', () => {
  const rng = mulberry32(64);
  const { set } = buildMixedSet(rng);
  const copy = set.clone();
  assert.ok(copy.equals(set));
  // Mutating the clone must not affect the original.
  copy.add(123_456_789);
  copy.remove([...copy][0]);
  assert.equal(set.has(123_456_789), false);
  assert.ok(!copy.equals(set));

  const before = set.toArray();
  set.shrinkToFit();
  assert.deepEqual(set.toArray(), before);
});

// ---------------------------------------------------------------------------
// Randomized operation-sequence differential test vs a JS Set model
// ---------------------------------------------------------------------------

test('randomized op sequence differential vs JS Set', { timeout: 120_000 }, () => {
  const rng = mulberry32(0xdeadbeef);
  const model = new Set<number>();
  const s = new RoaringSet();
  // Bias values into a few hot shards so all container types appear.
  const hot = () => {
    const r = rng();
    if (r < 0.5) return Math.floor(rng() * 8) * 0x10000 + Math.floor(rng() * 0x10000);
    if (r < 0.8) return 3 * 0x10000 + Math.floor(rng() * 30000); // dense shard
    return randUint32(rng);
  };

  for (let step = 0; step < 4000; step++) {
    const op = rng();
    if (op < 0.45) {
      const v = hot();
      assert.equal(s.add(v), !model.has(v), `add(${v}) changed-flag at step ${step}`);
      model.add(v);
    } else if (op < 0.7) {
      const v = hot();
      assert.equal(s.remove(v), model.has(v), `remove(${v}) changed-flag at step ${step}`);
      model.delete(v);
    } else if (op < 0.8) {
      const base = hot();
      const len = Math.floor(rng() * 2000);
      const end = Math.min(base + len, MAX_U32);
      s.addRange(base, end);
      for (let v = base; v <= end; v++) model.add(v);
    } else if (op < 0.9) {
      const v = hot();
      assert.equal(s.has(v), model.has(v), `has(${v}) at step ${step}`);
    } else {
      const v = randUint32(rng);
      let expected = 0;
      for (const m of model) if (m <= v) expected++;
      assert.equal(s.rank(v), expected, `rank(${v}) at step ${step}`);
    }
    if (step % 200 === 0) {
      assert.equal(s.size, model.size, `size at step ${step}`);
      if (model.size > 0) {
        const idx = Math.floor(rng() * model.size);
        const sorted = [...model].sort((a, b) => a - b);
        assert.equal(s.select(idx), sorted[idx], `select(${idx}) at step ${step}`);
      }
      if (step % 1000 === 0) s.runOptimize();
    }
  }
  assert.equal(s.size, model.size, 'final size');
  assert.deepEqual(s.toArray(), [...model].sort((a, b) => a - b), 'final content');

  // And the whole thing must survive a serialization round trip.
  const back = RoaringSet.deserialize(s.serialize());
  assert.ok(back.equals(s));
});

test('addRange over the full 32-bit space', () => {
  const s = new RoaringSet();
  s.addRange(0, MAX_U32);
  assert.equal(s.size, 0x100000000);
  assert.equal(s.containerCount, 0x10000);
  assert.equal(s.has(0), true);
  assert.equal(s.has(MAX_U32), true);
  assert.equal(s.rank(MAX_U32), 0x100000000);
  assert.equal(s.select(0x100000000 - 1), MAX_U32);
  assert.equal(s.minimum(), 0);
  assert.equal(s.maximum(), MAX_U32);
  // Removing both extremes keeps the extremes of the remainder.
  s.remove(0);
  s.remove(MAX_U32);
  assert.equal(s.minimum(), 1);
  assert.equal(s.maximum(), MAX_U32 - 1);
  assert.equal(s.size, 0x100000000 - 2);
});
