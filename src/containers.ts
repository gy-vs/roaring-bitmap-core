/**
 * Container layer for a Roaring-style 32-bit integer set.
 *
 * The 32-bit value space is sharded by the high 16 bits; each shard holds one
 * container for its low-16-bit values. Three container types exist:
 *
 *  - ArrayContainer : sorted Uint16 values, used while cardinality <= 4096
 *  - BitmapContainer: 65536-bit bitmap, used when cardinality > 4096
 *  - RunContainer   : sorted, disjoint (start, length) runs, used when it is
 *                     the most compact encoding
 *
 * All containers keep values sorted and track their own cardinality, so
 * conversions between them preserve ordering and cardinality by construction.
 */

export const ARRAY_LIMIT = 4096;
export const BITMAP_WORDS = 2048; // 2048 * 32 bits = 65536 values
export const BITMAP_BYTES = BITMAP_WORDS * 4;
export const MAX_LO = 0xffff;

export type ContainerType = 0 | 1 | 2;
export const ContainerType = {
  Array: 0 as ContainerType,
  Bitmap: 1 as ContainerType,
  Run: 2 as ContainerType,
};

export function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >>> 24;
}

/** Index of the lowest set bit of a non-zero word. */
function lowestBitIndex(word: number): number {
  return 31 - Math.clz32(word & -word);
}

export abstract class Container {
  abstract readonly type: ContainerType;
  abstract cardinality: number;

  abstract contains(x: number): boolean;
  /** May return a different container when a density threshold is crossed. */
  abstract add(x: number): Container;
  /** May return a different container when a density threshold is crossed. */
  abstract remove(x: number): Container;
  /** Number of elements <= x. */
  abstract rank(x: number): number;
  /** Element at 0-based rank i. Callers must guarantee i < cardinality. */
  abstract select(i: number): number;
  abstract min(): number;
  abstract max(): number;
  abstract [Symbol.iterator](): IterableIterator<number>;
  abstract iterateRange(lo: number, hi: number, scratch?: Uint32Array): IterableIterator<number>;
  abstract clone(): Container;
  abstract toArrayContainer(): ArrayContainer;
  abstract toBitmapContainer(): BitmapContainer;
  abstract toRunContainer(): RunContainer;
  abstract serializedSizeInBytes(): number;
  abstract serializeInto(view: DataView, offset: number): void;
}

// ---------------------------------------------------------------------------
// ArrayContainer
// ---------------------------------------------------------------------------

export class ArrayContainer extends Container {
  readonly type = ContainerType.Array;
  cardinality = 0;
  /** Sorted values; data.length is capacity, cardinality is the live count. */
  data: Uint16Array;

  private constructor(capacity: number) {
    super();
    this.data = new Uint16Array(capacity);
  }

  static create(capacity = 4): ArrayContainer {
    return new ArrayContainer(capacity);
  }

  /** Takes ownership of `data`; the first `cardinality` entries must be sorted. */
  static fromSorted(data: Uint16Array, cardinality: number): ArrayContainer {
    const c = new ArrayContainer(0);
    c.data = data;
    c.cardinality = cardinality;
    return c;
  }

  /** Binary search: index of x, or -(insertionPoint + 1) when absent. */
  indexOf(x: number): number {
    let lo = 0;
    let hi = this.cardinality - 1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      const v = this.data[mid];
      if (v < x) lo = mid + 1;
      else if (v > x) hi = mid - 1;
      else return mid;
    }
    return -(lo + 1);
  }

  /** First index whose value is >= x (may equal cardinality). */
  lowerBound(x: number): number {
    let lo = 0;
    let hi = this.cardinality;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.data[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  contains(x: number): boolean {
    return this.indexOf(x) >= 0;
  }

  ensureCapacity(cap: number): void {
    if (cap <= this.data.length) return;
    let next = Math.max(4, this.data.length);
    while (next < cap) next *= 2;
    const grown = new Uint16Array(next);
    grown.set(this.data.subarray(0, this.cardinality));
    this.data = grown;
  }

  add(x: number): Container {
    const idx = this.indexOf(x);
    if (idx >= 0) return this;
    if (this.cardinality >= ARRAY_LIMIT) {
      const bc = this.toBitmapContainer();
      bc.add(x);
      return bc;
    }
    const ins = -idx - 1;
    this.ensureCapacity(this.cardinality + 1);
    this.data.copyWithin(ins + 1, ins, this.cardinality);
    this.data[ins] = x;
    this.cardinality++;
    return this;
  }

  remove(x: number): Container {
    const idx = this.indexOf(x);
    if (idx < 0) return this;
    this.data.copyWithin(idx, idx + 1, this.cardinality);
    this.cardinality--;
    return this;
  }

  rank(x: number): number {
    let lo = 0;
    let hi = this.cardinality;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.data[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  select(i: number): number {
    return this.data[i];
  }

  min(): number {
    return this.data[0];
  }

  max(): number {
    return this.data[this.cardinality - 1];
  }

  *[Symbol.iterator](): IterableIterator<number> {
    for (let i = 0; i < this.cardinality; i++) yield this.data[i];
  }

  *iterateRange(lo: number, hi: number): IterableIterator<number> {
    let i = this.lowerBound(lo);
    while (i < this.cardinality && this.data[i] <= hi) yield this.data[i++];
  }

  clone(): ArrayContainer {
    return ArrayContainer.fromSorted(this.data.slice(0, this.cardinality), this.cardinality);
  }

  toArrayContainer(): ArrayContainer {
    return this;
  }

  toBitmapContainer(): BitmapContainer {
    const bc = BitmapContainer.create();
    for (let i = 0; i < this.cardinality; i++) bc.add(this.data[i]);
    return bc;
  }

  toRunContainer(): RunContainer {
    const starts: number[] = [];
    const lengths: number[] = [];
    let i = 0;
    while (i < this.cardinality) {
      const s = this.data[i];
      let e = s;
      while (i + 1 < this.cardinality && this.data[i + 1] === e + 1) {
        i++;
        e++;
      }
      starts.push(s);
      lengths.push(e - s);
      i++;
    }
    return RunContainer.fromArrays(starts, lengths);
  }

  serializedSizeInBytes(): number {
    return this.cardinality * 2;
  }

  serializeInto(view: DataView, offset: number): void {
    for (let i = 0; i < this.cardinality; i++) {
      view.setUint16(offset + i * 2, this.data[i], true);
    }
  }
}

// ---------------------------------------------------------------------------
// BitmapContainer
// ---------------------------------------------------------------------------

export class BitmapContainer extends Container {
  readonly type = ContainerType.Bitmap;
  cardinality = 0;
  readonly words: Uint32Array;

  private constructor(words: Uint32Array, cardinality: number) {
    super();
    this.words = words;
    this.cardinality = cardinality;
  }

  static create(): BitmapContainer {
    return new BitmapContainer(new Uint32Array(BITMAP_WORDS), 0);
  }

  /** Takes ownership of `words` (must have exactly BITMAP_WORDS entries). */
  static fromWords(words: Uint32Array, cardinality: number): BitmapContainer {
    return new BitmapContainer(words, cardinality);
  }

  contains(x: number): boolean {
    return (this.words[x >>> 5] & (1 << (x & 31))) !== 0;
  }

  add(x: number): Container {
    const w = x >>> 5;
    const bit = 1 << (x & 31);
    if ((this.words[w] & bit) === 0) {
      this.words[w] |= bit;
      this.cardinality++;
    }
    return this;
  }

  remove(x: number): Container {
    const w = x >>> 5;
    const bit = 1 << (x & 31);
    if ((this.words[w] & bit) !== 0) {
      this.words[w] &= ~bit;
      this.cardinality--;
      if (this.cardinality <= ARRAY_LIMIT) return this.toArrayContainer();
    }
    return this;
  }

  rank(x: number): number {
    const w = x >>> 5;
    let sum = 0;
    for (let i = 0; i < w; i++) sum += popcount32(this.words[i]);
    // Keep bits 0..(x&31) of the final word.
    sum += popcount32(this.words[w] & (0xffffffff >>> (31 - (x & 31))));
    return sum;
  }

  select(i: number): number {
    for (let w = 0; w < BITMAP_WORDS; w++) {
      const word = this.words[w];
      const c = popcount32(word);
      if (i < c) {
        let rest = word;
        for (let k = 0; k < i; k++) rest &= rest - 1;
        return (w << 5) + lowestBitIndex(rest);
      }
      i -= c;
    }
    throw new RangeError('select index out of range');
  }

  min(): number {
    for (let w = 0; w < BITMAP_WORDS; w++) {
      if (this.words[w] !== 0) return (w << 5) + lowestBitIndex(this.words[w]);
    }
    throw new RangeError('empty container');
  }

  max(): number {
    for (let w = BITMAP_WORDS - 1; w >= 0; w--) {
      const word = this.words[w];
      if (word !== 0) return (w << 5) + (31 - Math.clz32(word));
    }
    throw new RangeError('empty container');
  }

  *[Symbol.iterator](): IterableIterator<number> {
    for (let w = 0; w < BITMAP_WORDS; w++) {
      let word = this.words[w];
      const base = w << 5;
      while (word !== 0) {
        yield base + lowestBitIndex(word);
        word &= word - 1;
      }
    }
  }

  /**
   * Iterates values in [lo, hi]. When `scratch` is given, the covered words
   * are copied into it first so iteration is isolated from concurrent
   * mutation; the caller owns releasing that buffer.
   */
  *iterateRange(lo: number, hi: number, scratch?: Uint32Array): IterableIterator<number> {
    const w0 = lo >>> 5;
    const w1 = hi >>> 5;
    const n = w1 - w0 + 1;
    let words: Uint32Array;
    if (scratch !== undefined) {
      words = scratch.subarray(0, n);
      words.set(this.words.subarray(w0, w1 + 1));
    } else {
      words = this.words.subarray(w0, w1 + 1);
    }
    for (let i = 0; i < n; i++) {
      let word = words[i];
      if (i === 0) word &= 0xffffffff << (lo & 31);
      if (i === n - 1) word &= 0xffffffff >>> (31 - (hi & 31));
      const base = (w0 + i) << 5;
      while (word !== 0) {
        yield base + lowestBitIndex(word);
        word &= word - 1;
      }
    }
  }

  clone(): BitmapContainer {
    return new BitmapContainer(this.words.slice(), this.cardinality);
  }

  toArrayContainer(): ArrayContainer {
    const out = new Uint16Array(this.cardinality);
    let k = 0;
    for (let w = 0; w < BITMAP_WORDS; w++) {
      let word = this.words[w];
      const base = w << 5;
      while (word !== 0) {
        out[k++] = base + lowestBitIndex(word);
        word &= word - 1;
      }
    }
    return ArrayContainer.fromSorted(out, this.cardinality);
  }

  toBitmapContainer(): BitmapContainer {
    return this;
  }

  toRunContainer(): RunContainer {
    const starts: number[] = [];
    const lengths: number[] = [];
    let prev = -2;
    let runStart = -1;
    for (const v of this) {
      if (v !== prev + 1) {
        if (runStart >= 0) {
          starts.push(runStart);
          lengths.push(prev - runStart);
        }
        runStart = v;
      }
      prev = v;
    }
    if (runStart >= 0) {
      starts.push(runStart);
      lengths.push(prev - runStart);
    }
    return RunContainer.fromArrays(starts, lengths);
  }

  serializedSizeInBytes(): number {
    return BITMAP_BYTES;
  }

  serializeInto(view: DataView, offset: number): void {
    for (let i = 0; i < BITMAP_WORDS; i++) {
      view.setUint32(offset + i * 4, this.words[i], true);
    }
  }
}

// ---------------------------------------------------------------------------
// RunContainer
// ---------------------------------------------------------------------------

export class RunContainer extends Container {
  readonly type = ContainerType.Run;
  cardinality = 0;
  runCount = 0;
  private starts: Uint16Array;
  private lengths: Uint16Array;

  private constructor(capacity: number) {
    super();
    this.starts = new Uint16Array(capacity);
    this.lengths = new Uint16Array(capacity);
  }

  static create(capacity = 4): RunContainer {
    return new RunContainer(capacity);
  }

  /** Single run [start, start + length]. */
  static single(start: number, length: number): RunContainer {
    const c = new RunContainer(1);
    c.starts[0] = start;
    c.lengths[0] = length;
    c.runCount = 1;
    c.cardinality = length + 1;
    return c;
  }

  static fromArrays(starts: number[], lengths: number[]): RunContainer {
    const c = new RunContainer(starts.length);
    let card = 0;
    for (let i = 0; i < starts.length; i++) {
      c.starts[i] = starts[i];
      c.lengths[i] = lengths[i];
      card += lengths[i] + 1;
    }
    c.runCount = starts.length;
    c.cardinality = card;
    return c;
  }

  /** Takes ownership of typed arrays; used by the deserializer. */
  static fromTyped(starts: Uint16Array, lengths: Uint16Array, runCount: number, cardinality: number): RunContainer {
    const c = new RunContainer(0);
    c.starts = starts;
    c.lengths = lengths;
    c.runCount = runCount;
    c.cardinality = cardinality;
    return c;
  }

  startAt(i: number): number {
    return this.starts[i];
  }

  endAt(i: number): number {
    return this.starts[i] + this.lengths[i];
  }

  /** Index of the last run with start <= x, or -1. */
  private lastRunStartingAtOrBefore(x: number): number {
    let lo = 0;
    let hi = this.runCount - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >>> 1;
      if (this.starts[mid] <= x) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return idx;
  }

  contains(x: number): boolean {
    const idx = this.lastRunStartingAtOrBefore(x);
    return idx >= 0 && x - this.starts[idx] <= this.lengths[idx];
  }

  private ensureCapacity(cap: number): void {
    if (cap <= this.starts.length) return;
    let next = Math.max(4, this.starts.length);
    while (next < cap) next *= 2;
    const s = new Uint16Array(next);
    const l = new Uint16Array(next);
    s.set(this.starts.subarray(0, this.runCount));
    l.set(this.lengths.subarray(0, this.runCount));
    this.starts = s;
    this.lengths = l;
  }

  private removeRunAt(idx: number): void {
    this.starts.copyWithin(idx, idx + 1, this.runCount);
    this.lengths.copyWithin(idx, idx + 1, this.runCount);
    this.runCount--;
  }

  private insertRunAt(idx: number, start: number, length: number): void {
    this.ensureCapacity(this.runCount + 1);
    this.starts.copyWithin(idx + 1, idx, this.runCount);
    this.lengths.copyWithin(idx + 1, idx, this.runCount);
    this.starts[idx] = start;
    this.lengths[idx] = length;
    this.runCount++;
  }

  add(x: number): Container {
    const idx = this.lastRunStartingAtOrBefore(x);
    if (idx >= 0) {
      const end = this.starts[idx] + this.lengths[idx];
      if (x <= end) return this; // already present
      if (x === end + 1) {
        // Extend run idx rightwards, merging with the next run if adjacent.
        this.cardinality++;
        const next = idx + 1;
        if (next < this.runCount && this.starts[next] === x + 1) {
          this.lengths[idx] = this.starts[next] + this.lengths[next] - this.starts[idx];
          this.removeRunAt(next);
        } else {
          this.lengths[idx]++;
        }
        return this;
      }
    }
    const next = idx + 1;
    if (next < this.runCount && this.starts[next] === x + 1) {
      // Extend the following run leftwards.
      this.starts[next] = x;
      this.lengths[next]++;
      this.cardinality++;
      return this;
    }
    this.insertRunAt(next, x, 0);
    this.cardinality++;
    return this;
  }

  remove(x: number): Container {
    const idx = this.lastRunStartingAtOrBefore(x);
    if (idx < 0) return this;
    const s = this.starts[idx];
    const e = s + this.lengths[idx];
    if (x > e) return this;
    this.cardinality--;
    if (s === e) {
      this.removeRunAt(idx);
    } else if (x === s) {
      this.starts[idx]++;
      this.lengths[idx]--;
    } else if (x === e) {
      this.lengths[idx]--;
    } else {
      // Split [s, e] into [s, x-1] and [x+1, e].
      const rightStart = x + 1;
      const rightLen = e - x - 1;
      this.lengths[idx] = x - s - 1;
      this.insertRunAt(idx + 1, rightStart, rightLen);
    }
    return this;
  }

  rank(x: number): number {
    const idx = this.lastRunStartingAtOrBefore(x);
    if (idx < 0) return 0;
    let total = 0;
    for (let i = 0; i < idx; i++) total += this.lengths[i] + 1;
    total += Math.min(x - this.starts[idx], this.lengths[idx]) + 1;
    return total;
  }

  select(i: number): number {
    for (let r = 0; r < this.runCount; r++) {
      const c = this.lengths[r] + 1;
      if (i < c) return this.starts[r] + i;
      i -= c;
    }
    throw new RangeError('select index out of range');
  }

  min(): number {
    return this.starts[0];
  }

  max(): number {
    return this.starts[this.runCount - 1] + this.lengths[this.runCount - 1];
  }

  *[Symbol.iterator](): IterableIterator<number> {
    for (let r = 0; r < this.runCount; r++) {
      const end = this.starts[r] + this.lengths[r];
      for (let v = this.starts[r]; v <= end; v++) yield v;
    }
  }

  *iterateRange(lo: number, hi: number): IterableIterator<number> {
    // First run whose end >= lo.
    let loIdx = 0;
    let hiIdx = this.runCount;
    while (loIdx < hiIdx) {
      const mid = (loIdx + hiIdx) >>> 1;
      if (this.starts[mid] + this.lengths[mid] < lo) loIdx = mid + 1;
      else hiIdx = mid;
    }
    for (let r = loIdx; r < this.runCount && this.starts[r] <= hi; r++) {
      const s = Math.max(this.starts[r], lo);
      const e = Math.min(this.starts[r] + this.lengths[r], hi);
      for (let v = s; v <= e; v++) yield v;
    }
  }

  clone(): RunContainer {
    return RunContainer.fromTyped(
      this.starts.slice(0, this.runCount),
      this.lengths.slice(0, this.runCount),
      this.runCount,
      this.cardinality,
    );
  }

  /** Replaces this container's contents with another run container's. */
  adopt(other: RunContainer): void {
    this.starts = other.starts.slice(0, other.runCount);
    this.lengths = other.lengths.slice(0, other.runCount);
    this.runCount = other.runCount;
    this.cardinality = other.cardinality;
  }

  toArrayContainer(): ArrayContainer {
    const out = new Uint16Array(this.cardinality);
    let k = 0;
    for (let r = 0; r < this.runCount; r++) {
      const end = this.starts[r] + this.lengths[r];
      for (let v = this.starts[r]; v <= end; v++) out[k++] = v;
    }
    return ArrayContainer.fromSorted(out, this.cardinality);
  }

  toBitmapContainer(): BitmapContainer {
    const bc = BitmapContainer.create();
    for (const v of this) bc.add(v);
    return bc;
  }

  toRunContainer(): RunContainer {
    return this;
  }

  serializedSizeInBytes(): number {
    return this.runCount * 4;
  }

  serializeInto(view: DataView, offset: number): void {
    for (let r = 0; r < this.runCount; r++) {
      view.setUint16(offset + r * 4, this.starts[r], true);
      view.setUint16(offset + r * 4 + 2, this.lengths[r], true);
    }
  }
}

/** Number of runs the container's values would form. */
export function countRuns(c: Container): number {
  if (c instanceof RunContainer) return c.runCount;
  let runs = 0;
  let prev = -2;
  for (const v of c) {
    if (v !== prev + 1) runs++;
    prev = v;
  }
  return runs;
}

/**
 * Picks the most compact representation for `c` (ties keep the current type)
 * and converts if a different type wins. Sorting and cardinality are
 * preserved by the conversion routines themselves.
 */
export function normalize(c: Container): Container {
  const card = c.cardinality;
  if (card === 0) return c;
  let bestSize = c.serializedSizeInBytes();
  let bestType = c.type;
  if (card <= ARRAY_LIMIT) {
    if (card * 2 < bestSize) {
      bestSize = card * 2;
      bestType = ContainerType.Array;
    }
  } else if (BITMAP_BYTES < bestSize) {
    bestSize = BITMAP_BYTES;
    bestType = ContainerType.Bitmap;
  }
  const runSize = countRuns(c) * 4;
  if (runSize < bestSize) {
    bestType = ContainerType.Run;
  }
  if (bestType === c.type) return c;
  if (bestType === ContainerType.Array) return c.toArrayContainer();
  if (bestType === ContainerType.Bitmap) return c.toBitmapContainer();
  return c.toRunContainer();
}
