/** Deterministic PRNG (mulberry32) so tests are reproducible. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randInt(rng: () => number, lo: number, hi: number): number {
  return lo + Math.floor(rng() * (hi - lo + 1));
}

export function sorted(values: Iterable<number>): number[] {
  return Array.from(values).sort((a, b) => a - b);
}

/** Naive rank: count of elements <= v in a sorted array. */
export function naiveRank(sortedArr: number[], v: number): number {
  let lo = 0;
  let hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (sortedArr[mid] <= v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function naiveUnion(a: Set<number>, b: Set<number>): Set<number> {
  const out = new Set(a);
  for (const v of b) out.add(v);
  return out;
}

export function naiveIntersect(a: Set<number>, b: Set<number>): Set<number> {
  const out = new Set<number>();
  for (const v of a) if (b.has(v)) out.add(v);
  return out;
}

export function naiveDifference(a: Set<number>, b: Set<number>): Set<number> {
  const out = new Set<number>();
  for (const v of a) if (!b.has(v)) out.add(v);
  return out;
}

export function naiveXor(a: Set<number>, b: Set<number>): Set<number> {
  const out = new Set<number>();
  for (const v of a) if (!b.has(v)) out.add(v);
  for (const v of b) if (!a.has(v)) out.add(v);
  return out;
}
