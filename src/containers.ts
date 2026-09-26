import { popcount32, ctz32, wordRangeMask } from "./bits";

/** Maximum cardinality of an array container before it converts to a bitmap. */
export const ARRAY_LIMIT = 4096;
/** Number of 32-bit words in a bitmap container (2^16 bits). */
export const BITMAP_WORDS = 2048;
/** Size of a bitmap container in bytes. */
export const BITMAP_BYTES = BITMAP_WORDS * 4;
/** Values covered by one shard (the low 16 bits). */
export const SHARD_CAPACITY = 1 << 16;

export const TYPE_ARRAY = 1;
export const TYPE_BITMAP = 2;
export const TYPE_RUN = 3;
export type ContainerType = typeof TYPE_ARRAY | typeof TYPE_BITMAP | typeof TYPE_RUN;

/** A half-closed run stored as inclusive start + length (length >= 1). */
export interface Run {
  start: number; // 0..65535, inclusive
  length: number; // >= 1, start + length - 1 <= 65535
}

export function runEnd(r: Run): number {
  return r.start + r.length - 1;
}

/* ------------------------------------------------------------------ */
/* Array container: sorted Uint16Array, cardinality <= ARRAY_LIMIT.    */
/* ------------------------------------------------------------------ */

export class ArrayContainer {
  readonly type: typeof TYPE_ARRAY = TYPE_ARRAY;
  values: Uint16Array;
  cardinality: number;

  constructor(values?: Uint16Array, cardinality?: number) {
    this.values = values ?? new Uint16Array(4);
    this.cardinality = cardinality ?? 0;
  }

  static of(v: number): ArrayContainer {
    const c = new ArrayContainer(new Uint16Array(4), 1);
    c.values[0] = v;
    return c;
  }

  /** Index of the first value >= v. */
  lowerBound(v: number): number {
    let lo = 0;
    let hi = this.cardinality;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.values[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Index of the first value >= v, searching from position `from`. */
  lowerBoundAt(from: number, v: number): number {
    let lo = from;
    let hi = this.cardinality;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.values[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  /** Index of the first value > v. */
  upperBound(v: number): number {
    let lo = 0;
    let hi = this.cardinality;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.values[mid] <= v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  has(v: number): boolean {
    const i = this.lowerBound(v);
    return i < this.cardinality && this.values[i] === v;
  }

  add(v: number): Container {
    const i = this.lowerBound(v);
    if (i < this.cardinality && this.values[i] === v) return this;
    if (this.cardinality === this.values.length) {
      const grown = new Uint16Array(Math.max(4, this.values.length * 2));
      grown.set(this.values.subarray(0, this.cardinality));
      this.values = grown;
    }
    this.values.copyWithin(i + 1, i, this.cardinality);
    this.values[i] = v;
    this.cardinality++;
    if (this.cardinality > ARRAY_LIMIT) return this.toBitmap();
    return this;
  }

  delete(v: number): Container {
    const i = this.lowerBound(v);
    if (i >= this.cardinality || this.values[i] !== v) return this;
    this.values.copyWithin(i, i + 1, this.cardinality);
    this.cardinality--;
    return this;
  }

  /** Number of values <= v. */
  rank(v: number): number {
    return this.upperBound(v);
  }

  select(i: number): number {
    return i >= 0 && i < this.cardinality ? this.values[i] : -1;
  }

  addRange(start: number, end: number): Container {
    const lo = this.lowerBound(start);
    const hi = this.upperBound(end);
    const rangeCard = end - start + 1;
    const newCard = lo + rangeCard + (this.cardinality - hi);
    if (newCard <= ARRAY_LIMIT) {
      const out = new Uint16Array(Math.max(4, newCard));
      out.set(this.values.subarray(0, lo), 0);
      for (let v = start, p = lo; v <= end; v++, p++) out[p] = v;
      out.set(this.values.subarray(hi, this.cardinality), lo + rangeCard);
      this.values = out;
      this.cardinality = newCard;
      return canonicalizeArray(this);
    }
    const bm = this.toBitmap();
    return bm.addRange(start, end);
  }

  deleteRange(start: number, end: number): Container {
    const lo = this.lowerBound(start);
    const hi = this.upperBound(end);
    if (lo < hi) {
      this.values.copyWithin(lo, hi, this.cardinality);
      this.cardinality -= hi - lo;
    }
    return this;
  }

  runCount(): number {
    let n = 0;
    for (let i = 0; i < this.cardinality; i++) {
      if (i === 0 || this.values[i] !== this.values[i - 1] + 1) n++;
    }
    return n;
  }

  toBitmap(): BitmapContainer {
    const bm = new BitmapContainer(new Uint32Array(BITMAP_WORDS), this.cardinality);
    for (let i = 0; i < this.cardinality; i++) {
      const v = this.values[i];
      bm.words[v >>> 5] = (bm.words[v >>> 5] | (1 << (v & 31))) >>> 0;
    }
    return bm;
  }

  toRun(): RunContainer {
    const runs: Run[] = [];
    if (this.cardinality > 0) {
      let start = this.values[0];
      let prev = start;
      for (let i = 1; i < this.cardinality; i++) {
        const v = this.values[i];
        if (v !== prev + 1) {
          runs.push({ start, length: prev - start + 1 });
          start = v;
        }
        prev = v;
      }
      runs.push({ start, length: prev - start + 1 });
    }
    return new RunContainer(runs, this.cardinality);
  }

  clone(): ArrayContainer {
    return new ArrayContainer(this.values.slice(0, this.cardinality), this.cardinality);
  }
}

/* ------------------------------------------------------------------ */
/* Bitmap container: 2048 x 32-bit words, cardinality > ARRAY_LIMIT.   */
/* ------------------------------------------------------------------ */

export class BitmapContainer {
  readonly type: typeof TYPE_BITMAP = TYPE_BITMAP;
  words: Uint32Array;
  cardinality: number;

  constructor(words?: Uint32Array, cardinality?: number) {
    this.words = words ?? new Uint32Array(BITMAP_WORDS);
    this.cardinality = cardinality ?? 0;
  }

  has(v: number): boolean {
    return (this.words[v >>> 5] & (1 << (v & 31))) !== 0;
  }

  add(v: number): Container {
    const w = v >>> 5;
    const m = 1 << (v & 31);
    if ((this.words[w] & m) === 0) {
      this.words[w] = (this.words[w] | m) >>> 0;
      this.cardinality++;
    }
    return this;
  }

  delete(v: number): Container {
    const w = v >>> 5;
    const m = 1 << (v & 31);
    if ((this.words[w] & m) !== 0) {
      this.words[w] = (this.words[w] & ~m) >>> 0;
      this.cardinality--;
      if (this.cardinality <= ARRAY_LIMIT) return this.toArray();
    }
    return this;
  }

  /** Number of values <= v. */
  rank(v: number): number {
    const w = v >>> 5;
    let sum = 0;
    for (let i = 0; i < w; i++) sum += popcount32(this.words[i]);
    sum += popcount32(this.words[w] & wordRangeMask(0, v & 31));
    return sum;
  }

  select(i: number): number {
    if (i < 0 || i >= this.cardinality) return -1;
    let remaining = i;
    for (let w = 0; w < BITMAP_WORDS; w++) {
      const word = this.words[w];
      const pc = popcount32(word);
      if (remaining < pc) {
        let bits = word | 0;
        while (remaining-- > 0) bits &= bits - 1;
        return w * 32 + ctz32(bits);
      }
      remaining -= pc;
    }
    return -1;
  }

  addRange(start: number, end: number): Container {
    const firstWord = start >>> 5;
    const lastWord = end >>> 5;
    if (firstWord === lastWord) {
      this.words[firstWord] = (this.words[firstWord] | wordRangeMask(start & 31, end & 31)) >>> 0;
    } else {
      this.words[firstWord] = (this.words[firstWord] | wordRangeMask(start & 31, 31)) >>> 0;
      for (let w = firstWord + 1; w < lastWord; w++) this.words[w] = 0xffffffff;
      this.words[lastWord] = (this.words[lastWord] | wordRangeMask(0, end & 31)) >>> 0;
    }
    this.recount();
    return canonicalizeBitmap(this);
  }

  deleteRange(start: number, end: number): Container {
    const firstWord = start >>> 5;
    const lastWord = end >>> 5;
    if (firstWord === lastWord) {
      this.words[firstWord] = (this.words[firstWord] & ~wordRangeMask(start & 31, end & 31)) >>> 0;
    } else {
      this.words[firstWord] = (this.words[firstWord] & ~wordRangeMask(start & 31, 31)) >>> 0;
      for (let w = firstWord + 1; w < lastWord; w++) this.words[w] = 0;
      this.words[lastWord] = (this.words[lastWord] & ~wordRangeMask(0, end & 31)) >>> 0;
    }
    this.recount();
    return canonicalizeBitmap(this);
  }

  recount(): void {
    let sum = 0;
    for (let i = 0; i < BITMAP_WORDS; i++) sum += popcount32(this.words[i]);
    this.cardinality = sum;
  }

  toArray(): ArrayContainer {
    const out = new Uint16Array(Math.max(4, this.cardinality));
    let p = 0;
    for (let w = 0; w < BITMAP_WORDS; w++) {
      let bits = this.words[w] | 0;
      while (bits !== 0) {
        out[p++] = w * 32 + ctz32(bits);
        bits &= bits - 1;
      }
    }
    return new ArrayContainer(out, this.cardinality);
  }

  toRun(): RunContainer {
    const runs: Run[] = [];
    let start = -1;
    let prev = -2;
    for (let w = 0; w < BITMAP_WORDS; w++) {
      let bits = this.words[w] | 0;
      while (bits !== 0) {
        const v = w * 32 + ctz32(bits);
        bits &= bits - 1;
        if (v !== prev + 1) {
          if (start >= 0) runs.push({ start, length: prev - start + 1 });
          start = v;
        }
        prev = v;
      }
    }
    if (start >= 0) runs.push({ start, length: prev - start + 1 });
    return new RunContainer(runs, this.cardinality);
  }

  clone(): BitmapContainer {
    return new BitmapContainer(this.words.slice(), this.cardinality);
  }
}

/* ------------------------------------------------------------------ */
/* Run container: sorted, disjoint, non-adjacent runs.                 */
/* ------------------------------------------------------------------ */

export class RunContainer {
  readonly type: typeof TYPE_RUN = TYPE_RUN;
  runs: Run[];
  cardinality: number;

  constructor(runs?: Run[], cardinality?: number) {
    this.runs = runs ?? [];
    this.cardinality = cardinality ?? 0;
  }

  has(v: number): boolean {
    let lo = 0;
    let hi = this.runs.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const r = this.runs[mid];
      if (v < r.start) hi = mid;
      else if (v > runEnd(r)) lo = mid + 1;
      else return true;
    }
    return false;
  }

  /** Number of values <= v. */
  rank(v: number): number {
    let sum = 0;
    for (const r of this.runs) {
      if (r.start > v) break;
      sum += Math.min(v, runEnd(r)) - r.start + 1;
    }
    return sum;
  }

  select(i: number): number {
    if (i < 0 || i >= this.cardinality) return -1;
    let remaining = i;
    for (const r of this.runs) {
      if (remaining < r.length) return r.start + remaining;
      remaining -= r.length;
    }
    return -1;
  }

  add(v: number): Container {
    // Find the last run with start <= v (index i), or -1.
    let lo = 0;
    let hi = this.runs.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.runs[mid].start <= v) lo = mid + 1;
      else hi = mid;
    }
    const i = lo - 1;
    if (i >= 0) {
      const r = this.runs[i];
      if (v <= runEnd(r)) return this; // already present
      if (v === runEnd(r) + 1) {
        r.length++;
        const next = this.runs[i + 1];
        if (next && next.start === v + 1) {
          r.length += next.length;
          this.runs.splice(i + 1, 1);
        }
        this.recount();
        return this.finishAdd();
      }
    }
    const next = this.runs[i + 1];
    if (next && next.start === v + 1) {
      next.start = v;
      next.length++;
      this.recount();
      return this.finishAdd();
    }
    this.runs.splice(i + 1, 0, { start: v, length: 1 });
    this.recount();
    return this.finishAdd();
  }

  delete(v: number): Container {
    for (let i = 0; i < this.runs.length; i++) {
      const r = this.runs[i];
      if (v < r.start) break;
      if (v > runEnd(r)) continue;
      if (r.length === 1) {
        this.runs.splice(i, 1);
      } else if (v === r.start) {
        r.start++;
        r.length--;
      } else if (v === runEnd(r)) {
        r.length--;
      } else {
        const right: Run = { start: v + 1, length: runEnd(r) - v };
        r.length = v - r.start;
        this.runs.splice(i + 1, 0, right);
      }
      this.recount();
      return this.finishDelete();
    }
    return this;
  }

  addRange(start: number, end: number): Container {
    const out: Run[] = [];
    let ns = start;
    let ne = end;
    let inserted = false;
    for (const r of this.runs) {
      const re = runEnd(r);
      if (re + 1 < ns) {
        out.push(r);
      } else if (r.start > ne + 1) {
        if (!inserted) {
          out.push({ start: ns, length: ne - ns + 1 });
          inserted = true;
        }
        out.push(r);
      } else {
        ns = Math.min(ns, r.start);
        ne = Math.max(ne, re);
      }
    }
    if (!inserted) out.push({ start: ns, length: ne - ns + 1 });
    this.runs = out;
    this.recount();
    return this.finishAdd();
  }

  deleteRange(start: number, end: number): Container {
    const out: Run[] = [];
    for (const r of this.runs) {
      const re = runEnd(r);
      if (re < start || r.start > end) {
        out.push(r);
        continue;
      }
      if (r.start < start) out.push({ start: r.start, length: start - r.start });
      if (re > end) out.push({ start: end + 1, length: re - end });
    }
    this.runs = out;
    this.recount();
    return this.finishDelete();
  }

  recount(): void {
    let sum = 0;
    for (const r of this.runs) sum += r.length;
    this.cardinality = sum;
  }

  /** Convert to the densest encoding after growth. */
  private finishAdd(): Container {
    if (this.cardinality <= ARRAY_LIMIT && this.runs.length * 2 >= this.cardinality) {
      return this.toArray();
    }
    if (this.runs.length * 4 >= BITMAP_BYTES) return canonicalizeBitmap(this.toBitmap());
    return this;
  }

  private finishDelete(): Container {
    if (this.runs.length * 4 >= BITMAP_BYTES) return canonicalizeBitmap(this.toBitmap());
    if (this.cardinality <= ARRAY_LIMIT && this.runs.length * 2 >= this.cardinality) {
      return this.toArray();
    }
    return this;
  }

  toArray(): ArrayContainer {
    const out = new Uint16Array(Math.max(4, this.cardinality));
    let p = 0;
    for (const r of this.runs) {
      for (let v = r.start; v <= runEnd(r); v++) out[p++] = v;
    }
    return new ArrayContainer(out, this.cardinality);
  }

  toBitmap(): BitmapContainer {
    const bm = new BitmapContainer(new Uint32Array(BITMAP_WORDS), this.cardinality);
    for (const r of this.runs) {
      const re = runEnd(r);
      const firstWord = r.start >>> 5;
      const lastWord = re >>> 5;
      if (firstWord === lastWord) {
        bm.words[firstWord] = (bm.words[firstWord] | wordRangeMask(r.start & 31, re & 31)) >>> 0;
      } else {
        bm.words[firstWord] = (bm.words[firstWord] | wordRangeMask(r.start & 31, 31)) >>> 0;
        for (let w = firstWord + 1; w < lastWord; w++) bm.words[w] = 0xffffffff;
        bm.words[lastWord] = (bm.words[lastWord] | wordRangeMask(0, re & 31)) >>> 0;
      }
    }
    return bm;
  }

  clone(): RunContainer {
    return new RunContainer(this.runs.map((r) => ({ ...r })), this.cardinality);
  }
}

/* ------------------------------------------------------------------ */
/* Union type + canonicalization helpers.                              */
/* ------------------------------------------------------------------ */

export type Container = ArrayContainer | BitmapContainer | RunContainer;

/** Array containers stay arrays unless the run encoding is strictly denser. */
export function canonicalizeArray(ac: ArrayContainer): Container {
  if (ac.cardinality > ARRAY_LIMIT) return canonicalizeBitmap(ac.toBitmap());
  if (ac.cardinality >= 2 && ac.runCount() * 2 < ac.cardinality) return ac.toRun();
  return ac;
}

/** Bitmap containers convert down to arrays at the threshold, and to runs when that is smaller. */
export function canonicalizeBitmap(bm: BitmapContainer): Container {
  if (bm.cardinality <= ARRAY_LIMIT) return canonicalizeArray(bm.toArray());
  const rc = bm.toRun();
  if (rc.runs.length * 4 < BITMAP_BYTES) return rc;
  return bm;
}

/** Pick the densest encoding for a run list produced by a set operation. */
export function canonicalizeRuns(runs: Run[]): Container {
  let card = 0;
  for (const r of runs) card += r.length;
  const rc = new RunContainer(runs, card);
  if (card <= ARRAY_LIMIT && runs.length * 2 >= card) return rc.toArray();
  if (runs.length * 4 >= BITMAP_BYTES) return rc.toBitmap();
  return rc;
}

/** Iterate the low-16-bit values of a container in ascending order. */
export function* containerValues(c: Container): Generator<number> {
  if (c.type === TYPE_ARRAY) {
    for (let i = 0; i < c.cardinality; i++) yield c.values[i];
  } else if (c.type === TYPE_BITMAP) {
    for (let w = 0; w < BITMAP_WORDS; w++) {
      let bits = c.words[w] | 0;
      while (bits !== 0) {
        yield w * 32 + ctz32(bits);
        bits &= bits - 1;
      }
    }
  } else {
    for (const run of c.runs) {
      for (let v = run.start; v <= runEnd(run); v++) yield v;
    }
  }
}

/** Logical equality: same values, regardless of the chosen container encoding. */
export function containerEquals(a: Container, b: Container): boolean {
  if (a.cardinality !== b.cardinality) return false;
  if (a.type === b.type) {
    if (a.type === TYPE_ARRAY && b.type === TYPE_ARRAY) {
      for (let i = 0; i < a.cardinality; i++) if (a.values[i] !== b.values[i]) return false;
      return true;
    }
    if (a.type === TYPE_BITMAP && b.type === TYPE_BITMAP) {
      for (let i = 0; i < BITMAP_WORDS; i++) if (a.words[i] !== b.words[i]) return false;
      return true;
    }
    if (a.type === TYPE_RUN && b.type === TYPE_RUN) {
      if (a.runs.length !== b.runs.length) return false;
      for (let i = 0; i < a.runs.length; i++) {
        if (a.runs[i].start !== b.runs[i].start || a.runs[i].length !== b.runs[i].length) return false;
      }
      return true;
    }
  }
  const ia = containerValues(a);
  const ib = containerValues(b);
  for (;;) {
    const na = ia.next();
    const nb = ib.next();
    if (na.done !== nb.done) return false;
    if (na.done) return true;
    if (na.value !== nb.value) return false;
  }
}
