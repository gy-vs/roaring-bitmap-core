/**
 * Container-level set operations. Every operation dispatches on the concrete
 * pair of container types and uses an algorithm specialised for that pair
 * (merge joins for sorted arrays, word-wise ops for bitmaps, interval
 * algebra for runs) instead of materialising both sides as arrays.
 *
 * The `containerI*` variants are the in-place counterparts: they reuse the
 * left operand's storage where the representation allows it (word-wise
 * bitmap ops, in-place array merges/filters, run-list replacement) and fall
 * back to the functional version otherwise. They never mutate the right
 * operand.
 */

import {
  ARRAY_LIMIT,
  BITMAP_WORDS,
  ArrayContainer,
  BitmapContainer,
  Container,
  RunContainer,
  normalize,
  popcount32,
} from './containers';

// ---------------------------------------------------------------------------
// Word-range helpers (operate on raw bitmap words)
// ---------------------------------------------------------------------------

function setRangeBits(words: Uint32Array, s: number, e: number): void {
  const w0 = s >>> 5;
  const w1 = e >>> 5;
  if (w0 === w1) {
    words[w0] |= (0xffffffff << (s & 31)) & (0xffffffff >>> (31 - (e & 31)));
    return;
  }
  words[w0] |= 0xffffffff << (s & 31);
  for (let w = w0 + 1; w < w1; w++) words[w] = 0xffffffff;
  words[w1] |= 0xffffffff >>> (31 - (e & 31));
}

function clearRangeBits(words: Uint32Array, s: number, e: number): void {
  const w0 = s >>> 5;
  const w1 = e >>> 5;
  if (w0 === w1) {
    words[w0] &= ~((0xffffffff << (s & 31)) & (0xffffffff >>> (31 - (e & 31))));
    return;
  }
  words[w0] &= ~(0xffffffff << (s & 31));
  for (let w = w0 + 1; w < w1; w++) words[w] = 0;
  words[w1] &= ~(0xffffffff >>> (31 - (e & 31)));
}

function flipRangeBits(words: Uint32Array, s: number, e: number): void {
  const w0 = s >>> 5;
  const w1 = e >>> 5;
  if (w0 === w1) {
    words[w0] ^= (0xffffffff << (s & 31)) & (0xffffffff >>> (31 - (e & 31)));
    return;
  }
  words[w0] ^= 0xffffffff << (s & 31);
  for (let w = w0 + 1; w < w1; w++) words[w] ^= 0xffffffff;
  words[w1] ^= 0xffffffff >>> (31 - (e & 31));
}

function recount(words: Uint32Array): number {
  let card = 0;
  for (let i = 0; i < BITMAP_WORDS; i++) card += popcount32(words[i]);
  return card;
}

// ---------------------------------------------------------------------------
// Intersection primitives
// ---------------------------------------------------------------------------

function andArrayArray(a: ArrayContainer, b: ArrayContainer): Container | null {
  const n1 = a.cardinality;
  const n2 = b.cardinality;
  const d1 = a.data;
  const d2 = b.data;
  const out = new Uint16Array(Math.min(n1, n2));
  let i = 0;
  let j = 0;
  let k = 0;
  while (i < n1 && j < n2) {
    const x = d1[i];
    const y = d2[j];
    if (x < y) i++;
    else if (x > y) j++;
    else {
      out[k++] = x;
      i++;
      j++;
    }
  }
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andArrayBitmap(a: ArrayContainer, bm: BitmapContainer): Container | null {
  const out = new Uint16Array(a.cardinality);
  let k = 0;
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.data[i];
    if (bm.contains(v)) out[k++] = v;
  }
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andArrayRun(a: ArrayContainer, r: RunContainer): Container | null {
  const out = new Uint16Array(a.cardinality);
  const d = a.data;
  const n = a.cardinality;
  let k = 0;
  let idx = 0;
  for (let run = 0; run < r.runCount && idx < n; run++) {
    const s = r.startAt(run);
    const e = r.endAt(run);
    while (idx < n && d[idx] < s) idx++;
    while (idx < n && d[idx] <= e) out[k++] = d[idx++];
  }
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andBitmapBitmap(a: BitmapContainer, b: BitmapContainer): Container | null {
  const words = new Uint32Array(BITMAP_WORDS);
  let card = 0;
  for (let i = 0; i < BITMAP_WORDS; i++) {
    const w = a.words[i] & b.words[i];
    words[i] = w;
    card += popcount32(w);
  }
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function andBitmapRun(bm: BitmapContainer, r: RunContainer): Container | null {
  const words = new Uint32Array(BITMAP_WORDS);
  let card = 0;
  for (let i = 0; i < r.runCount; i++) {
    const s = r.startAt(i);
    const e = r.endAt(i);
    const w0 = s >>> 5;
    const w1 = e >>> 5;
    if (w0 === w1) {
      const mask = (0xffffffff << (s & 31)) & (0xffffffff >>> (31 - (e & 31)));
      const w = bm.words[w0] & mask;
      words[w0] |= w;
      card += popcount32(w);
    } else {
      const wf = bm.words[w0] & (0xffffffff << (s & 31));
      words[w0] |= wf;
      card += popcount32(wf);
      for (let w = w0 + 1; w < w1; w++) {
        const ww = bm.words[w];
        words[w] = ww;
        card += popcount32(ww);
      }
      const wl = bm.words[w1] & (0xffffffff >>> (31 - (e & 31)));
      words[w1] |= wl;
      card += popcount32(wl);
    }
  }
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function andRunRun(a: RunContainer, b: RunContainer): Container | null {
  const starts: number[] = [];
  const lengths: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.runCount && j < b.runCount) {
    const s = Math.max(a.startAt(i), b.startAt(j));
    const e = Math.min(a.endAt(i), b.endAt(j));
    if (s <= e) {
      starts.push(s);
      lengths.push(e - s);
    }
    if (a.endAt(i) < b.endAt(j)) i++;
    else j++;
  }
  return starts.length === 0 ? null : RunContainer.fromArrays(starts, lengths);
}

// ---------------------------------------------------------------------------
// Union primitives
// ---------------------------------------------------------------------------

function orArrayArray(a: ArrayContainer, b: ArrayContainer): Container {
  const n1 = a.cardinality;
  const n2 = b.cardinality;
  const d1 = a.data;
  const d2 = b.data;
  const out = new Uint16Array(Math.min(65536, n1 + n2));
  let i = 0;
  let j = 0;
  let k = 0;
  while (i < n1 && j < n2) {
    const x = d1[i];
    const y = d2[j];
    if (x < y) out[k++] = d1[i++];
    else if (x > y) out[k++] = d2[j++];
    else {
      out[k++] = x;
      i++;
      j++;
    }
  }
  while (i < n1) out[k++] = d1[i++];
  while (j < n2) out[k++] = d2[j++];
  if (k > ARRAY_LIMIT) {
    const bc = BitmapContainer.create();
    for (let m = 0; m < k; m++) bc.add(out[m]);
    return bc;
  }
  return ArrayContainer.fromSorted(out, k);
}

function orBitmapBitmap(a: BitmapContainer, b: BitmapContainer): Container {
  const words = new Uint32Array(BITMAP_WORDS);
  let card = 0;
  for (let i = 0; i < BITMAP_WORDS; i++) {
    const w = a.words[i] | b.words[i];
    words[i] = w;
    card += popcount32(w);
  }
  return BitmapContainer.fromWords(words, card);
}

function orArrayBitmap(a: ArrayContainer, bm: BitmapContainer): Container {
  const words = bm.words.slice();
  let card = bm.cardinality;
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.data[i];
    const w = v >>> 5;
    const bit = 1 << (v & 31);
    if ((words[w] & bit) === 0) {
      words[w] |= bit;
      card++;
    }
  }
  return BitmapContainer.fromWords(words, card);
}

function orRunBitmap(r: RunContainer, bm: BitmapContainer): Container {
  const words = bm.words.slice();
  for (let i = 0; i < r.runCount; i++) setRangeBits(words, r.startAt(i), r.endAt(i));
  return BitmapContainer.fromWords(words, recount(words));
}

function orRunRun(a: RunContainer, b: RunContainer): Container {
  const starts: number[] = [];
  const lengths: number[] = [];
  let i = 0;
  let j = 0;
  let cs = -1;
  let ce = -1;
  const absorb = (s: number, e: number): void => {
    if (cs === -1) {
      cs = s;
      ce = e;
    } else if (s <= ce + 1) {
      if (e > ce) ce = e;
    } else {
      starts.push(cs);
      lengths.push(ce - cs);
      cs = s;
      ce = e;
    }
  };
  while (i < a.runCount && j < b.runCount) {
    if (a.startAt(i) <= b.startAt(j)) {
      absorb(a.startAt(i), a.endAt(i));
      i++;
    } else {
      absorb(b.startAt(j), b.endAt(j));
      j++;
    }
  }
  while (i < a.runCount) {
    absorb(a.startAt(i), a.endAt(i));
    i++;
  }
  while (j < b.runCount) {
    absorb(b.startAt(j), b.endAt(j));
    j++;
  }
  if (cs !== -1) {
    starts.push(cs);
    lengths.push(ce - cs);
  }
  return RunContainer.fromArrays(starts, lengths);
}

function orArrayRun(a: ArrayContainer, r: RunContainer): Container {
  const starts: number[] = [];
  const lengths: number[] = [];
  let cs = -1;
  let ce = -1;
  const absorb = (s: number, e: number): void => {
    if (cs === -1) {
      cs = s;
      ce = e;
    } else if (s <= ce + 1) {
      if (e > ce) ce = e;
    } else {
      starts.push(cs);
      lengths.push(ce - cs);
      cs = s;
      ce = e;
    }
  };
  const d = a.data;
  const n = a.cardinality;
  let i = 0;
  let j = 0;
  while (i < n && j < r.runCount) {
    if (d[i] < r.startAt(j)) absorb(d[i++], d[i - 1]);
    else {
      absorb(r.startAt(j), r.endAt(j));
      j++;
    }
  }
  while (i < n) {
    absorb(d[i], d[i]);
    i++;
  }
  while (j < r.runCount) {
    absorb(r.startAt(j), r.endAt(j));
    j++;
  }
  if (cs !== -1) {
    starts.push(cs);
    lengths.push(ce - cs);
  }
  return RunContainer.fromArrays(starts, lengths);
}

// ---------------------------------------------------------------------------
// Difference primitives (a \ b)
// ---------------------------------------------------------------------------

function andNotArrayArray(a: ArrayContainer, b: ArrayContainer): Container | null {
  const d1 = a.data;
  const d2 = b.data;
  const n1 = a.cardinality;
  const n2 = b.cardinality;
  const out = new Uint16Array(n1);
  let i = 0;
  let j = 0;
  let k = 0;
  while (i < n1 && j < n2) {
    const x = d1[i];
    const y = d2[j];
    if (x < y) out[k++] = d1[i++];
    else if (x > y) j++;
    else {
      i++;
      j++;
    }
  }
  while (i < n1) out[k++] = d1[i++];
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andNotArrayBitmap(a: ArrayContainer, bm: BitmapContainer): Container | null {
  const out = new Uint16Array(a.cardinality);
  let k = 0;
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.data[i];
    if (!bm.contains(v)) out[k++] = v;
  }
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andNotArrayRun(a: ArrayContainer, r: RunContainer): Container | null {
  const out = new Uint16Array(a.cardinality);
  const d = a.data;
  let k = 0;
  let run = 0;
  for (let i = 0; i < a.cardinality; i++) {
    const v = d[i];
    while (run < r.runCount && r.endAt(run) < v) run++;
    if (run < r.runCount && r.startAt(run) <= v) continue;
    out[k++] = v;
  }
  return k === 0 ? null : ArrayContainer.fromSorted(out, k);
}

function andNotBitmapBitmap(a: BitmapContainer, b: BitmapContainer): Container | null {
  const words = new Uint32Array(BITMAP_WORDS);
  let card = 0;
  for (let i = 0; i < BITMAP_WORDS; i++) {
    const w = a.words[i] & ~b.words[i];
    words[i] = w;
    card += popcount32(w);
  }
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function andNotBitmapRun(bm: BitmapContainer, r: RunContainer): Container | null {
  const words = bm.words.slice();
  for (let i = 0; i < r.runCount; i++) clearRangeBits(words, r.startAt(i), r.endAt(i));
  const card = recount(words);
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function andNotRunBitmap(r: RunContainer, bm: BitmapContainer): Container | null {
  const out = new Uint16Array(r.cardinality);
  let k = 0;
  for (let i = 0; i < r.runCount; i++) {
    const end = r.endAt(i);
    for (let v = r.startAt(i); v <= end; v++) {
      if (!bm.contains(v)) out[k++] = v;
    }
  }
  if (k === 0) return null;
  if (k > ARRAY_LIMIT) {
    const bc = BitmapContainer.create();
    for (let m = 0; m < k; m++) bc.add(out[m]);
    return bc;
  }
  return ArrayContainer.fromSorted(out, k);
}

function andNotRunArray(r: RunContainer, a: ArrayContainer): Container | null {
  const starts: number[] = [];
  const lengths: number[] = [];
  const d = a.data;
  const n = a.cardinality;
  let idx = 0;
  for (let i = 0; i < r.runCount; i++) {
    const s = r.startAt(i);
    const e = r.endAt(i);
    while (idx < n && d[idx] < s) idx++;
    let cur = s;
    while (idx < n && d[idx] <= e) {
      if (d[idx] > cur) {
        starts.push(cur);
        lengths.push(d[idx] - 1 - cur);
      }
      cur = d[idx] + 1;
      idx++;
    }
    if (cur <= e) {
      starts.push(cur);
      lengths.push(e - cur);
    }
  }
  return starts.length === 0 ? null : RunContainer.fromArrays(starts, lengths);
}

function andNotRunRun(a: RunContainer, b: RunContainer): Container | null {
  const starts: number[] = [];
  const lengths: number[] = [];
  let j = 0;
  for (let i = 0; i < a.runCount; i++) {
    const s = a.startAt(i);
    const e = a.endAt(i);
    while (j < b.runCount && b.endAt(j) < s) j++;
    let cur = s;
    let k = j;
    while (k < b.runCount && b.startAt(k) <= e) {
      if (b.startAt(k) > cur) {
        starts.push(cur);
        lengths.push(b.startAt(k) - 1 - cur);
      }
      cur = Math.max(cur, b.endAt(k) + 1);
      if (cur > e) break;
      k++;
    }
    if (cur <= e) {
      starts.push(cur);
      lengths.push(e - cur);
    }
  }
  return starts.length === 0 ? null : RunContainer.fromArrays(starts, lengths);
}

// ---------------------------------------------------------------------------
// Symmetric-difference primitives
// ---------------------------------------------------------------------------

function xorArrayArray(a: ArrayContainer, b: ArrayContainer): Container | null {
  const d1 = a.data;
  const d2 = b.data;
  const n1 = a.cardinality;
  const n2 = b.cardinality;
  const out = new Uint16Array(Math.min(65536, n1 + n2));
  let i = 0;
  let j = 0;
  let k = 0;
  while (i < n1 && j < n2) {
    const x = d1[i];
    const y = d2[j];
    if (x < y) out[k++] = d1[i++];
    else if (x > y) out[k++] = d2[j++];
    else {
      i++;
      j++;
    }
  }
  while (i < n1) out[k++] = d1[i++];
  while (j < n2) out[k++] = d2[j++];
  if (k === 0) return null;
  if (k > ARRAY_LIMIT) {
    const bc = BitmapContainer.create();
    for (let m = 0; m < k; m++) bc.add(out[m]);
    return bc;
  }
  return ArrayContainer.fromSorted(out, k);
}

function xorArrayBitmap(a: ArrayContainer, bm: BitmapContainer): Container | null {
  const words = bm.words.slice();
  let card = bm.cardinality;
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.data[i];
    const w = v >>> 5;
    const bit = 1 << (v & 31);
    if ((words[w] & bit) === 0) {
      words[w] |= bit;
      card++;
    } else {
      words[w] &= ~bit;
      card--;
    }
  }
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function xorBitmapBitmap(a: BitmapContainer, b: BitmapContainer): Container | null {
  const words = new Uint32Array(BITMAP_WORDS);
  let card = 0;
  for (let i = 0; i < BITMAP_WORDS; i++) {
    const w = a.words[i] ^ b.words[i];
    words[i] = w;
    card += popcount32(w);
  }
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

function xorRunBitmap(r: RunContainer, bm: BitmapContainer): Container | null {
  const words = bm.words.slice();
  for (let i = 0; i < r.runCount; i++) flipRangeBits(words, r.startAt(i), r.endAt(i));
  const card = recount(words);
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

// ---------------------------------------------------------------------------
// Dispatchers
// ---------------------------------------------------------------------------

export function containerAnd(a: Container, b: Container): Container | null {
  let r: Container | null;
  if (a instanceof ArrayContainer) {
    if (b instanceof ArrayContainer) r = andArrayArray(a, b);
    else if (b instanceof BitmapContainer) r = andArrayBitmap(a, b);
    else r = andArrayRun(a, b as RunContainer);
  } else if (a instanceof BitmapContainer) {
    if (b instanceof ArrayContainer) r = andArrayBitmap(b, a);
    else if (b instanceof BitmapContainer) r = andBitmapBitmap(a, b);
    else r = andBitmapRun(a, b as RunContainer);
  } else {
    const ra = a as RunContainer;
    if (b instanceof ArrayContainer) r = andArrayRun(b, ra);
    else if (b instanceof BitmapContainer) r = andBitmapRun(b, ra);
    else r = andRunRun(ra, b as RunContainer);
  }
  return r === null || r.cardinality === 0 ? null : normalize(r);
}

export function containerOr(a: Container, b: Container): Container {
  let r: Container;
  if (a instanceof ArrayContainer) {
    if (b instanceof ArrayContainer) r = orArrayArray(a, b);
    else if (b instanceof BitmapContainer) r = orArrayBitmap(a, b);
    else r = orArrayRun(a, b as RunContainer);
  } else if (a instanceof BitmapContainer) {
    if (b instanceof ArrayContainer) r = orArrayBitmap(b, a);
    else if (b instanceof BitmapContainer) r = orBitmapBitmap(a, b);
    else r = orRunBitmap(b as RunContainer, a);
  } else {
    const ra = a as RunContainer;
    if (b instanceof ArrayContainer) r = orArrayRun(b, ra);
    else if (b instanceof BitmapContainer) r = orRunBitmap(ra, b);
    else r = orRunRun(ra, b as RunContainer);
  }
  return normalize(r);
}

export function containerAndNot(a: Container, b: Container): Container | null {
  let r: Container | null;
  if (a instanceof ArrayContainer) {
    if (b instanceof ArrayContainer) r = andNotArrayArray(a, b);
    else if (b instanceof BitmapContainer) r = andNotArrayBitmap(a, b);
    else r = andNotArrayRun(a, b as RunContainer);
  } else if (a instanceof BitmapContainer) {
    if (b instanceof ArrayContainer) r = andNotBitmapArray(a, b);
    else if (b instanceof BitmapContainer) r = andNotBitmapBitmap(a, b);
    else r = andNotBitmapRun(a, b as RunContainer);
  } else {
    const ra = a as RunContainer;
    if (b instanceof ArrayContainer) r = andNotRunArray(ra, b);
    else if (b instanceof BitmapContainer) r = andNotRunBitmap(ra, b);
    else r = andNotRunRun(ra, b as RunContainer);
  }
  return r === null || r.cardinality === 0 ? null : normalize(r);
}

/** bitmap \ array: clear the array's bits in a copy of the bitmap. */
function andNotBitmapArray(bm: BitmapContainer, a: ArrayContainer): Container | null {
  const words = bm.words.slice();
  for (let i = 0; i < a.cardinality; i++) {
    const v = a.data[i];
    words[v >>> 5] &= ~(1 << (v & 31));
  }
  const card = recount(words);
  return card === 0 ? null : BitmapContainer.fromWords(words, card);
}

export function containerXor(a: Container, b: Container): Container | null {
  let r: Container | null;
  if (a instanceof ArrayContainer) {
    if (b instanceof ArrayContainer) r = xorArrayArray(a, b);
    else if (b instanceof BitmapContainer) r = xorArrayBitmap(a, b);
    else r = xorViaHalves(a, b as RunContainer);
  } else if (a instanceof BitmapContainer) {
    if (b instanceof ArrayContainer) r = xorArrayBitmap(b, a);
    else if (b instanceof BitmapContainer) r = xorBitmapBitmap(a, b);
    else r = xorRunBitmap(b as RunContainer, a);
  } else {
    const ra = a as RunContainer;
    if (b instanceof ArrayContainer) r = xorViaHalves(b, ra);
    else if (b instanceof BitmapContainer) r = xorRunBitmap(ra, b);
    else r = xorRunRun(ra, b as RunContainer);
  }
  return r === null || r.cardinality === 0 ? null : normalize(r);
}

/** array △ run = (array \ run) ∪ (run \ array), both cheap specialisations. */
function xorViaHalves(a: ArrayContainer, r: RunContainer): Container | null {
  const left = andNotArrayRun(a, r);
  const right = andNotRunArray(r, a);
  if (left === null) return right;
  if (right === null) return left;
  return orArrayRun(left as ArrayContainer, right as RunContainer);
}

/** run △ run = (a \ b) ∪ (b \ a). */
function xorRunRun(a: RunContainer, b: RunContainer): Container | null {
  const left = andNotRunRun(a, b);
  const right = andNotRunRun(b, a);
  if (left === null) return right;
  if (right === null) return left;
  return orRunRun(left as RunContainer, right as RunContainer);
}

// ---------------------------------------------------------------------------
// In-place variants. The left container's storage is reused where the
// representation allows; the result (possibly a converted container, or null
// when empty) is returned and the caller must store it back.
// ---------------------------------------------------------------------------

export function containerIOr(a: Container, b: Container): Container {
  if (a instanceof BitmapContainer) {
    if (b instanceof BitmapContainer) {
      for (let i = 0; i < BITMAP_WORDS; i++) a.words[i] |= b.words[i];
      a.cardinality = recount(a.words);
      return a;
    }
    if (b instanceof ArrayContainer) {
      for (let i = 0; i < b.cardinality; i++) a.add(b.data[i]);
      return a;
    }
    const r = b as RunContainer;
    for (let i = 0; i < r.runCount; i++) setRangeBits(a.words, r.startAt(i), r.endAt(i));
    a.cardinality = recount(a.words);
    return a;
  }
  if (a instanceof ArrayContainer && b instanceof ArrayContainer) {
    // Backward in-place merge of two sorted arrays with dedup. Duplicates
    // leave a gap between the unplaced prefix and the placed suffix, so the
    // suffix is compacted down afterwards.
    const n1 = a.cardinality;
    const n2 = b.cardinality;
    a.ensureCapacity(n1 + n2);
    let i = n1 - 1;
    let j = n2 - 1;
    let k = n1 + n2 - 1;
    while (i >= 0 && j >= 0) {
      const x = a.data[i];
      const y = b.data[j];
      if (x > y) {
        a.data[k--] = x;
        i--;
      } else if (x < y) {
        a.data[k--] = y;
        j--;
      } else {
        a.data[k--] = x;
        i--;
        j--;
      }
    }
    let rem: number;
    if (j >= 0) {
      // a exhausted first: copy b's remaining prefix in front.
      while (j >= 0) a.data[k--] = b.data[j--];
      rem = k + 1;
    } else {
      rem = i + 1;
    }
    const placed = n1 + n2 - 1 - k;
    if (k + 1 !== rem) a.data.copyWithin(rem, k + 1, n1 + n2);
    a.cardinality = rem + placed;
    if (a.cardinality > ARRAY_LIMIT) return a.toBitmapContainer();
    return a;
  }
  if (a instanceof RunContainer && b instanceof RunContainer) {
    a.adopt(orRunRun(a, b) as RunContainer);
    return a;
  }
  return containerOr(a, b);
}

export function containerIAnd(a: Container, b: Container): Container | null {
  if (a instanceof BitmapContainer && b instanceof BitmapContainer) {
    let card = 0;
    for (let i = 0; i < BITMAP_WORDS; i++) {
      const w = a.words[i] & b.words[i];
      a.words[i] = w;
      card += popcount32(w);
    }
    a.cardinality = card;
    if (card === 0) return null;
    return card <= ARRAY_LIMIT ? a.toArrayContainer() : a;
  }
  if (a instanceof ArrayContainer) {
    const d = a.data;
    let k = 0;
    if (b instanceof ArrayContainer) {
      const d2 = b.data;
      const n2 = b.cardinality;
      let i = 0;
      let j = 0;
      while (i < a.cardinality && j < n2) {
        const x = d[i];
        const y = d2[j];
        if (x < y) i++;
        else if (x > y) j++;
        else {
          d[k++] = x;
          i++;
          j++;
        }
      }
    } else if (b instanceof BitmapContainer) {
      for (let i = 0; i < a.cardinality; i++) {
        if (b.contains(d[i])) d[k++] = d[i];
      }
    } else {
      const r = b as RunContainer;
      let run = 0;
      for (let i = 0; i < a.cardinality; i++) {
        const v = d[i];
        while (run < r.runCount && r.endAt(run) < v) run++;
        if (run < r.runCount && r.startAt(run) <= v) d[k++] = v;
      }
    }
    a.cardinality = k;
    return k === 0 ? null : a;
  }
  if (a instanceof RunContainer && b instanceof RunContainer) {
    const r = andRunRun(a, b);
    if (r === null) return null;
    a.adopt(r as RunContainer);
    return a;
  }
  return containerAnd(a, b);
}

export function containerIAndNot(a: Container, b: Container): Container | null {
  if (a instanceof BitmapContainer) {
    if (b instanceof BitmapContainer) {
      for (let i = 0; i < BITMAP_WORDS; i++) a.words[i] &= ~b.words[i];
    } else if (b instanceof ArrayContainer) {
      for (let i = 0; i < b.cardinality; i++) {
        const v = b.data[i];
        a.words[v >>> 5] &= ~(1 << (v & 31));
      }
    } else {
      const r = b as RunContainer;
      for (let i = 0; i < r.runCount; i++) clearRangeBits(a.words, r.startAt(i), r.endAt(i));
    }
    a.cardinality = recount(a.words);
    if (a.cardinality === 0) return null;
    return a.cardinality <= ARRAY_LIMIT ? a.toArrayContainer() : a;
  }
  if (a instanceof ArrayContainer) {
    const d = a.data;
    let k = 0;
    if (b instanceof ArrayContainer) {
      const d2 = b.data;
      const n2 = b.cardinality;
      let i = 0;
      let j = 0;
      while (i < a.cardinality) {
        const x = d[i];
        while (j < n2 && d2[j] < x) j++;
        if (j < n2 && d2[j] === x) {
          i++;
          continue;
        }
        d[k++] = d[i++];
      }
    } else {
      for (let i = 0; i < a.cardinality; i++) {
        if (!b.contains(d[i])) d[k++] = d[i];
      }
    }
    a.cardinality = k;
    return k === 0 ? null : a;
  }
  if (a instanceof RunContainer && b instanceof RunContainer) {
    const r = andNotRunRun(a, b);
    if (r === null) return null;
    a.adopt(r as RunContainer);
    return a;
  }
  return containerAndNot(a, b);
}

export function containerIXor(a: Container, b: Container): Container | null {
  if (a instanceof BitmapContainer) {
    if (b instanceof BitmapContainer) {
      for (let i = 0; i < BITMAP_WORDS; i++) a.words[i] ^= b.words[i];
    } else if (b instanceof ArrayContainer) {
      for (let i = 0; i < b.cardinality; i++) {
        const v = b.data[i];
        a.words[v >>> 5] ^= 1 << (v & 31);
      }
    } else {
      const r = b as RunContainer;
      for (let i = 0; i < r.runCount; i++) flipRangeBits(a.words, r.startAt(i), r.endAt(i));
    }
    a.cardinality = recount(a.words);
    if (a.cardinality === 0) return null;
    return a.cardinality <= ARRAY_LIMIT ? a.toArrayContainer() : a;
  }
  if (a instanceof ArrayContainer && b instanceof ArrayContainer) {
    // Backward in-place symmetric-difference merge; same compaction as ior.
    const n1 = a.cardinality;
    const n2 = b.cardinality;
    a.ensureCapacity(n1 + n2);
    let i = n1 - 1;
    let j = n2 - 1;
    let k = n1 + n2 - 1;
    while (i >= 0 && j >= 0) {
      const x = a.data[i];
      const y = b.data[j];
      if (x > y) {
        a.data[k--] = x;
        i--;
      } else if (x < y) {
        a.data[k--] = y;
        j--;
      } else {
        i--;
        j--;
      }
    }
    let rem: number;
    if (j >= 0) {
      while (j >= 0) a.data[k--] = b.data[j--];
      rem = k + 1;
    } else {
      rem = i + 1;
    }
    const placed = n1 + n2 - 1 - k;
    if (k + 1 !== rem) a.data.copyWithin(rem, k + 1, n1 + n2);
    a.cardinality = rem + placed;
    if (a.cardinality === 0) return null;
    if (a.cardinality > ARRAY_LIMIT) return a.toBitmapContainer();
    return a;
  }
  if (a instanceof RunContainer && b instanceof RunContainer) {
    const r = xorRunRun(a, b);
    if (r === null) return null;
    a.adopt(r as RunContainer);
    return a;
  }
  return containerXor(a, b);
}
