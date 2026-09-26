import {
  ArrayContainer,
  BitmapContainer,
  RunContainer,
  Container,
  Run,
  TYPE_ARRAY,
  TYPE_BITMAP,
  TYPE_RUN,
  BITMAP_WORDS,
  ARRAY_LIMIT,
  SHARD_CAPACITY,
  canonicalizeArray,
  canonicalizeBitmap,
  canonicalizeRuns,
  runEnd,
} from "./containers";
import { ctz32, wordRangeMask } from "./bits";

/**
 * Container-level set algebra. Every pair of container types gets an
 * algorithm that works on the native encodings (sorted merge, word-wise
 * bit operations, or interval sweeps) instead of expanding to values.
 */

export type SetOp = "union" | "intersect" | "difference" | "xor";

export function containerOp(a: Container, b: Container, op: SetOp): Container | null {
  const key = (a.type << 4) | b.type;
  switch (key) {
    case (TYPE_ARRAY << 4) | TYPE_ARRAY:
      return arrayArray(a as ArrayContainer, b as ArrayContainer, op);
    case (TYPE_ARRAY << 4) | TYPE_BITMAP:
      return arrayBitmap(a as ArrayContainer, b as BitmapContainer, op, false);
    case (TYPE_BITMAP << 4) | TYPE_ARRAY:
      return arrayBitmap(b as ArrayContainer, a as BitmapContainer, op, true);
    case (TYPE_ARRAY << 4) | TYPE_RUN:
      return arrayRun(a as ArrayContainer, b as RunContainer, op, false);
    case (TYPE_RUN << 4) | TYPE_ARRAY:
      return arrayRun(b as ArrayContainer, a as RunContainer, op, true);
    case (TYPE_BITMAP << 4) | TYPE_BITMAP:
      return bitmapBitmap(a as BitmapContainer, b as BitmapContainer, op);
    case (TYPE_BITMAP << 4) | TYPE_RUN:
      return bitmapRun(a as BitmapContainer, b as RunContainer, op, false);
    case (TYPE_RUN << 4) | TYPE_BITMAP:
      return bitmapRun(b as BitmapContainer, a as RunContainer, op, true);
    case (TYPE_RUN << 4) | TYPE_RUN:
      return runRun(a as RunContainer, b as RunContainer, op);
    default:
      throw new Error(`unknown container type combination: ${a.type}, ${b.type}`);
  }
}

/** Append [s, e] to a run list, merging with the last run when adjacent. */
function pushRun(out: Run[], s: number, e: number): void {
  const last = out[out.length - 1];
  if (last && s <= runEnd(last) + 1) {
    if (e > runEnd(last)) last.length = e - last.start + 1;
  } else {
    out.push({ start: s, length: e - s + 1 });
  }
}

/* ----------------------------- array x array ----------------------------- */

function arrayArray(a: ArrayContainer, b: ArrayContainer, op: SetOp): Container | null {
  if (op === "union" || op === "xor") {
    const out = new Uint16Array(a.cardinality + b.cardinality);
    let i = 0;
    let j = 0;
    let p = 0;
    while (i < a.cardinality && j < b.cardinality) {
      const x = a.values[i];
      const y = b.values[j];
      if (x < y) {
        out[p++] = x;
        i++;
      } else if (x > y) {
        out[p++] = y;
        j++;
      } else {
        if (op === "union") out[p++] = x;
        i++;
        j++;
      }
    }
    while (i < a.cardinality) out[p++] = a.values[i++];
    while (j < b.cardinality) out[p++] = b.values[j++];
    if (p === 0) return null;
    return canonicalizeArray(new ArrayContainer(out.slice(0, p), p));
  }

  if (op === "intersect") {
    const out = new Uint16Array(Math.min(a.cardinality, b.cardinality));
    let i = 0;
    let j = 0;
    let p = 0;
    while (i < a.cardinality && j < b.cardinality) {
      if (a.values[i] === b.values[j]) {
        out[p++] = a.values[i];
        i++;
        j++;
      } else if (a.values[i] < b.values[j]) {
        i = a.lowerBoundAt(i, b.values[j]);
      } else {
        j = b.lowerBoundAt(j, a.values[i]);
      }
    }
    if (p === 0) return null;
    return new ArrayContainer(out.slice(0, p), p);
  }

  // difference: a - b
  const out = new Uint16Array(a.cardinality);
  let i = 0;
  let j = 0;
  let p = 0;
  while (i < a.cardinality && j < b.cardinality) {
    if (a.values[i] < b.values[j]) out[p++] = a.values[i++];
    else if (a.values[i] > b.values[j]) j++;
    else {
      i++;
      j++;
    }
  }
  while (i < a.cardinality) out[p++] = a.values[i++];
  if (p === 0) return null;
  return new ArrayContainer(out.slice(0, p), p);
}

/* ----------------------------- array x bitmap ---------------------------- */

function arrayBitmap(
  arr: ArrayContainer,
  bm: BitmapContainer,
  op: SetOp,
  swapped: boolean,
): Container | null {
  if (op === "union" || op === "xor") {
    const words = bm.words.slice();
    for (let i = 0; i < arr.cardinality; i++) {
      const v = arr.values[i];
      const w = v >>> 5;
      const m = 1 << (v & 31);
      words[w] = op === "union" ? (words[w] | m) >>> 0 : (words[w] ^ m) >>> 0;
    }
    const out = new BitmapContainer(words);
    out.recount();
    return out.cardinality === 0 ? null : canonicalizeBitmap(out);
  }

  if (op === "intersect") {
    const out = new Uint16Array(arr.cardinality);
    let p = 0;
    for (let i = 0; i < arr.cardinality; i++) {
      const v = arr.values[i];
      if ((bm.words[v >>> 5] & (1 << (v & 31))) !== 0) out[p++] = v;
    }
    return p === 0 ? null : new ArrayContainer(out.slice(0, p), p);
  }

  // difference
  if (!swapped) {
    // array - bitmap: keep array values absent from the bitmap.
    const out = new Uint16Array(arr.cardinality);
    let p = 0;
    for (let i = 0; i < arr.cardinality; i++) {
      const v = arr.values[i];
      if ((bm.words[v >>> 5] & (1 << (v & 31))) === 0) out[p++] = v;
    }
    return p === 0 ? null : new ArrayContainer(out.slice(0, p), p);
  }
  // bitmap - array: clear the array's bits from a bitmap copy.
  const words = bm.words.slice();
  for (let i = 0; i < arr.cardinality; i++) {
    const v = arr.values[i];
    words[v >>> 5] = (words[v >>> 5] & ~(1 << (v & 31))) >>> 0;
  }
  const out = new BitmapContainer(words);
  out.recount();
  return out.cardinality === 0 ? null : canonicalizeBitmap(out);
}

/* ------------------------------ array x run ------------------------------ */

function arrayRun(
  arr: ArrayContainer,
  run: RunContainer,
  op: SetOp,
  swapped: boolean,
): Container | null {
  if (op === "union" || op === "xor") {
    let intervals = unionPointsRuns(arr, run.runs);
    if (op === "xor") {
      const common = pointsInsideRuns(arr, run.runs);
      intervals = subtractRuns(intervals, common);
    }
    return intervals.length === 0 ? null : canonicalizeRuns(intervals);
  }

  if (op === "intersect") {
    const out = new Uint16Array(arr.cardinality);
    let p = 0;
    let j = 0;
    for (let i = 0; i < arr.cardinality; i++) {
      const v = arr.values[i];
      while (j < run.runs.length && runEnd(run.runs[j]) < v) j++;
      if (j < run.runs.length && v >= run.runs[j].start) out[p++] = v;
    }
    return p === 0 ? null : new ArrayContainer(out.slice(0, p), p);
  }

  // difference
  if (!swapped) {
    // array - run: keep points outside every run.
    const out = new Uint16Array(arr.cardinality);
    let p = 0;
    let j = 0;
    for (let i = 0; i < arr.cardinality; i++) {
      const v = arr.values[i];
      while (j < run.runs.length && runEnd(run.runs[j]) < v) j++;
      if (!(j < run.runs.length && v >= run.runs[j].start)) out[p++] = v;
    }
    return p === 0 ? null : new ArrayContainer(out.slice(0, p), p);
  }
  // run - array: split the runs around the points that fall inside them.
  const result: Run[] = [];
  let i = 0;
  for (const r of run.runs) {
    const re = runEnd(r);
    let cursor = r.start;
    while (i < arr.cardinality && arr.values[i] < r.start) i++;
    while (i < arr.cardinality && arr.values[i] <= re) {
      const v = arr.values[i++];
      if (v > cursor) pushRun(result, cursor, v - 1);
      cursor = v + 1;
    }
    if (cursor <= re) pushRun(result, cursor, re);
  }
  return result.length === 0 ? null : canonicalizeRuns(result);
}

/** Sorted merge of points and runs into maximal disjoint runs. O(n + k). */
function unionPointsRuns(a: ArrayContainer, runs: Run[]): Run[] {
  const out: Run[] = [];
  let i = 0;
  let j = 0;
  while (i < a.cardinality || j < runs.length) {
    let s: number;
    let e: number;
    if (j >= runs.length || (i < a.cardinality && a.values[i] < runs[j].start)) {
      s = a.values[i];
      e = s;
      i++;
    } else {
      s = runs[j].start;
      e = runEnd(runs[j]);
      j++;
    }
    pushRun(out, s, e);
  }
  return out;
}

/** Coalesced runs made of the array points that fall inside the given runs. */
function pointsInsideRuns(a: ArrayContainer, runs: Run[]): Run[] {
  const out: Run[] = [];
  let j = 0;
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.values[i];
    while (j < runs.length && runEnd(runs[j]) < v) j++;
    if (j < runs.length && v >= runs[j].start) pushRun(out, v, v);
  }
  return out;
}

/* ---------------------------- bitmap x bitmap ---------------------------- */

function bitmapBitmap(a: BitmapContainer, b: BitmapContainer, op: SetOp): Container | null {
  const words = new Uint32Array(BITMAP_WORDS);
  for (let w = 0; w < BITMAP_WORDS; w++) {
    const x = a.words[w];
    const y = b.words[w];
    words[w] =
      op === "union"
        ? (x | y) >>> 0
        : op === "intersect"
          ? (x & y) >>> 0
          : op === "xor"
            ? (x ^ y) >>> 0
            : (x & ~y) >>> 0;
  }
  const out = new BitmapContainer(words);
  out.recount();
  return out.cardinality === 0 ? null : canonicalizeBitmap(out);
}

/* ------------------------------ bitmap x run ----------------------------- */

function bitmapRun(
  bm: BitmapContainer,
  run: RunContainer,
  op: SetOp,
  swapped: boolean,
): Container | null {
  if (op === "union" || op === "xor") {
    const words = bm.words.slice();
    applyToRunRanges(run.runs, (w, mask) => {
      words[w] = op === "union" ? (words[w] | mask) >>> 0 : (words[w] ^ mask) >>> 0;
    });
    const out = new BitmapContainer(words);
    out.recount();
    return out.cardinality === 0 ? null : canonicalizeBitmap(out);
  }

  if (op === "intersect" || (swapped && op === "difference")) {
    // Restrict the bitmap to the run ranges.
    const words = new Uint32Array(BITMAP_WORDS);
    applyToRunRanges(run.runs, (w, mask) => {
      words[w] = (words[w] | (bm.words[w] & mask)) >>> 0;
    });
    const out = new BitmapContainer(words);
    out.recount();
    return out.cardinality === 0 ? null : canonicalizeBitmap(out);
  }

  if (!swapped) {
    // bitmap - run: clear the run ranges.
    const words = bm.words.slice();
    applyToRunRanges(run.runs, (w, mask) => {
      words[w] = (words[w] & ~mask) >>> 0;
    });
    const out = new BitmapContainer(words);
    out.recount();
    return out.cardinality === 0 ? null : canonicalizeBitmap(out);
  }

  // run - bitmap: keep the bits inside the runs that are 0 in the bitmap,
  // collected word-slice by word-slice straight into runs.
  const result: Run[] = [];
  for (const r of run.runs) {
    const re = runEnd(r);
    let cursor = r.start;
    while (cursor <= re) {
      const w = cursor >>> 5;
      const base = w * 32;
      const hi = Math.min(re, base + 31);
      const missing = (~bm.words[w] & wordRangeMask(cursor & 31, hi & 31)) >>> 0;
      let bits = missing | 0;
      while (bits !== 0) {
        const v = base + ctz32(bits);
        bits &= bits - 1;
        // Coalesce consecutive set bits within and across word slices.
        let e = v;
        while (bits !== 0 && base + ctz32(bits) === e + 1) {
          e++;
          bits &= bits - 1;
        }
        pushRun(result, v, e);
      }
      cursor = hi + 1;
    }
  }
  return result.length === 0 ? null : canonicalizeRuns(result);
}

/** Invoke fn(wordIndex, maskOfRunBitsInThatWord) for every word a run list touches. */
function applyToRunRanges(runs: Run[], fn: (word: number, mask: number) => void): void {
  for (const r of runs) {
    const re = runEnd(r);
    const first = r.start >>> 5;
    const last = re >>> 5;
    if (first === last) {
      fn(first, wordRangeMask(r.start & 31, re & 31));
    } else {
      fn(first, wordRangeMask(r.start & 31, 31));
      for (let w = first + 1; w < last; w++) fn(w, -1);
      fn(last, wordRangeMask(0, re & 31));
    }
  }
}

/* ------------------------------- run x run ------------------------------- */

function runRun(a: RunContainer, b: RunContainer, op: SetOp): Container | null {
  let result: Run[];
  if (op === "union") result = unionRuns(a.runs, b.runs);
  else if (op === "intersect") result = intersectRuns(a.runs, b.runs);
  else if (op === "difference") result = subtractRuns(a.runs, b.runs);
  else result = subtractRuns(unionRuns(a.runs, b.runs), intersectRuns(a.runs, b.runs));
  return result.length === 0 ? null : canonicalizeRuns(result);
}

function unionRuns(a: Run[], b: Run[]): Run[] {
  const out: Run[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    const next: Run =
      j >= b.length || (i < a.length && a[i].start <= b[j].start) ? a[i++] : b[j++];
    pushRun(out, next.start, runEnd(next));
  }
  return out;
}

function intersectRuns(a: Run[], b: Run[]): Run[] {
  const out: Run[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const s = Math.max(a[i].start, b[j].start);
    const e = Math.min(runEnd(a[i]), runEnd(b[j]));
    if (s <= e) pushRun(out, s, e);
    if (runEnd(a[i]) <= runEnd(b[j])) i++;
    else j++;
  }
  return out;
}

function subtractRuns(a: Run[], b: Run[]): Run[] {
  const out: Run[] = [];
  let j = 0;
  for (const r of a) {
    let cursor = r.start;
    const re = runEnd(r);
    while (j < b.length && runEnd(b[j]) < cursor) j++;
    let k = j;
    while (k < b.length && b[k].start <= re) {
      const s = Math.max(cursor, b[k].start);
      if (s > cursor) pushRun(out, cursor, s - 1);
      cursor = Math.max(cursor, runEnd(b[k]) + 1);
      k++;
    }
    if (cursor <= re) pushRun(out, cursor, re);
  }
  return out;
}

/* --------------------------------- misc ---------------------------------- */

/** Estimated in-memory size in bytes, useful for container selection tests. */
export function containerByteSize(c: Container): number {
  if (c.type === TYPE_ARRAY) return c.cardinality * 2;
  if (c.type === TYPE_BITMAP) return BITMAP_WORDS * 4;
  return c.runs.length * 4;
}

export { ARRAY_LIMIT, SHARD_CAPACITY };
