/**
 * RoaringSet: a set of unsigned 32-bit integers.
 *
 * Values are sharded by their high 16 bits; each shard holds one container
 * (array / bitmap / run) chosen by density. Shards are kept in two parallel
 * arrays sorted by key.
 *
 * Also implements the versioned binary format:
 *
 *   offset  size  field
 *   0       4     magic "RB32" (0x32334252, little-endian)
 *   4       4     format version (currently 1)
 *   8       4     container count N
 *   12      4     flags (reserved, must be 0)
 *   16      2N    shard keys (u16, strictly increasing)
 *   +2N     2N    cardinalities minus one (u16)
 *   +4N     N     container types (u8: 0=array, 1=bitmap, 2=run)
 *   (pad to 4)
 *   ...     4N    absolute payload offsets (u32)
 *   ...     ...   container payloads
 *
 * Payloads: array = card x u16 sorted values; bitmap = 2048 x u32 words;
 * run = runCount x (u16 start, u16 length). Decoding validates the header,
 * every offset and length, and each payload's cardinality and ordering
 * before building the set.
 */

import {
  ARRAY_LIMIT,
  BITMAP_BYTES,
  BITMAP_WORDS,
  ArrayContainer,
  BitmapContainer,
  Container,
  ContainerType,
  MAX_LO,
  RunContainer,
  normalize,
  popcount32,
} from './containers';
import {
  containerAnd,
  containerAndNot,
  containerIAnd,
  containerIAndNot,
  containerIOr,
  containerIXor,
  containerOr,
  containerXor,
} from './ops';

export const MAX_U32 = 0xffffffff;

const MAGIC = 0x32334252; // "RB32"
const FORMAT_VERSION = 1;
const HEADER_SIZE = 16;

export class DeserializationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'DeserializationError';
  }
}

/**
 * Pool of scratch word buffers used by range iteration over bitmap
 * containers. Buffers are released when an iterator finishes or is cancelled
 * by the caller (generator `return`, e.g. `break` in a for-of loop).
 */
export const wordPool = {
  free: [] as Uint32Array[],
  totalAcquired: 0,
  totalReleased: 0,
  acquire(): Uint32Array {
    this.totalAcquired++;
    return this.free.pop() ?? new Uint32Array(BITMAP_WORDS);
  },
  release(buf: Uint32Array): void {
    this.totalReleased++;
    if (this.free.length < 16) this.free.push(buf);
  },
};

function checkUint32(x: number, name: string): void {
  if (!Number.isInteger(x) || x < 0 || x > MAX_U32) {
    throw new RangeError(`${name} must be an integer in [0, 2^32 - 1], got ${x}`);
  }
}

export class RoaringSet implements Iterable<number> {
  private keys: number[] = [];
  private containers: Container[] = [];
  private _size = 0;
  /**
   * Prefix cardinalities: prefix[k] = total cardinality of shards [0, k),
   * prefix[0] = 0. Lazily rebuilt after structural or cardinality mutations
   * so rank/select can binary-search shards instead of summing them.
   */
  private prefix: number[] | null = null;

  private invalidatePrefix(): void {
    this.prefix = null;
  }

  private ensurePrefix(): number[] {
    if (this.prefix === null) {
      const p = new Array<number>(this.containers.length + 1);
      let sum = 0;
      p[0] = 0;
      for (let i = 0; i < this.containers.length; i++) {
        sum += this.containers[i].cardinality;
        p[i + 1] = sum;
      }
      this.prefix = p;
    }
    return this.prefix;
  }

  /** Number of elements in the set. */
  get size(): number {
    return this._size;
  }

  get isEmpty(): boolean {
    return this._size === 0;
  }

  /** Number of high-16-bit shards currently present. */
  get containerCount(): number {
    return this.keys.length;
  }

  static of(...values: number[]): RoaringSet {
    return RoaringSet.from(values);
  }

  static from(values: Iterable<number>): RoaringSet {
    return new RoaringSet().addMany(values);
  }

  /** Index of `hi` in keys, or -(insertionPoint + 1) when absent. */
  private findKey(hi: number): number {
    let lo = 0;
    let hi_ = this.keys.length - 1;
    while (lo <= hi_) {
      const mid = (lo + hi_) >>> 1;
      const k = this.keys[mid];
      if (k < hi) lo = mid + 1;
      else if (k > hi) hi_ = mid - 1;
      else return mid;
    }
    return -(lo + 1);
  }

  private insertContainerAt(idx: number, key: number, c: Container): void {
    this.keys.splice(idx, 0, key);
    this.containers.splice(idx, 0, c);
    this._size += c.cardinality;
  }

  private removeContainerAt(idx: number): void {
    this._size -= this.containers[idx].cardinality;
    this.keys.splice(idx, 1);
    this.containers.splice(idx, 1);
  }

  // -------------------------------------------------------------------------
  // Mutation
  // -------------------------------------------------------------------------

  /** Adds x; returns true when the set changed. */
  add(x: number): boolean {
    checkUint32(x, 'value');
    this.invalidatePrefix();
    const hi = x >>> 16;
    const lo = x & 0xffff;
    const idx = this.findKey(hi);
    if (idx >= 0) {
      const c = this.containers[idx];
      const before = c.cardinality;
      const nc = c.add(lo);
      this.containers[idx] = nc;
      if (nc.cardinality === before) return false;
      this._size++;
      return true;
    }
    const c = ArrayContainer.create();
    c.add(lo);
    this.insertContainerAt(-idx - 1, hi, c);
    return true;
  }

  /** Adds every value of the iterable; bulk-merged shard by shard. */
  addMany(values: Iterable<number>): this {
    this.invalidatePrefix();
    const arr: number[] = [];
    for (const v of values) {
      checkUint32(v, 'value');
      arr.push(v);
    }
    arr.sort((a, b) => a - b);
    let i = 0;
    while (i < arr.length) {
      const hi = arr[i] >>> 16;
      let j = i + 1;
      while (j < arr.length && arr[j] >>> 16 === hi) j++;
      const los: number[] = [];
      let prev = -1;
      for (let k = i; k < j; k++) {
        const lo = arr[k] & 0xffff;
        if (lo !== prev) {
          los.push(lo);
          prev = lo;
        }
      }
      const src = normalize(ArrayContainer.fromSorted(Uint16Array.from(los), los.length));
      const idx = this.findKey(hi);
      if (idx >= 0) {
        const old = this.containers[idx];
        const before = old.cardinality;
        const merged = containerIOr(old, src);
        this._size += merged.cardinality - before;
        this.containers[idx] = merged;
      } else {
        this.insertContainerAt(-idx - 1, hi, src);
      }
      i = j;
    }
    return this;
  }

  /** Adds every value in the inclusive range [start, end]. */
  addRange(start: number, end: number): this {
    checkUint32(start, 'start');
    this.invalidatePrefix();
    checkUint32(end, 'end');
    if (start > end) throw new RangeError(`empty range [${start}, ${end}]`);
    const hi0 = start >>> 16;
    const hi1 = end >>> 16;
    for (let hi = hi0; hi <= hi1; hi++) {
      const lo0 = hi === hi0 ? start & 0xffff : 0;
      const lo1 = hi === hi1 ? end & 0xffff : 0xffff;
      const src = normalize(RunContainer.single(lo0, lo1 - lo0));
      const idx = this.findKey(hi);
      if (idx >= 0) {
        const old = this.containers[idx];
        const before = old.cardinality;
        const merged = containerIOr(old, src);
        this._size += merged.cardinality - before;
        this.containers[idx] = merged;
      } else {
        this.insertContainerAt(-idx - 1, hi, src);
      }
    }
    return this;
  }

  /** Removes x; returns true when the set changed. */
  remove(x: number): boolean {
    checkUint32(x, 'value');
    this.invalidatePrefix();
    const hi = x >>> 16;
    const lo = x & 0xffff;
    const idx = this.findKey(hi);
    if (idx < 0) return false;
    const c = this.containers[idx];
    const before = c.cardinality;
    const nc = c.remove(lo);
    if (nc.cardinality === before) return false;
    this._size--;
    if (nc.cardinality === 0) {
      this.keys.splice(idx, 1);
      this.containers.splice(idx, 1);
    } else {
      this.containers[idx] = nc;
    }
    return true;
  }

  clear(): void {
    this.invalidatePrefix();
    this.keys = [];
    this.containers = [];
    this._size = 0;
  }

  // -------------------------------------------------------------------------
  // Queries
  // -------------------------------------------------------------------------

  has(x: number): boolean {
    checkUint32(x, 'value');
    const idx = this.findKey(x >>> 16);
    return idx >= 0 && this.containers[idx].contains(x & 0xffff);
  }

  /** Number of elements <= x. */
  rank(x: number): number {
    checkUint32(x, 'value');
    const hi = x >>> 16;
    const lo = x & 0xffff;
    const idx = this.findKey(hi);
    const prefix = this.ensurePrefix();
    if (idx >= 0) return prefix[idx] + this.containers[idx].rank(lo);
    return prefix[-idx - 1];
  }

  /** Element with 0-based rank i, or undefined when out of range. */
  select(i: number): number | undefined {
    if (!Number.isInteger(i) || i < 0 || i >= this._size) return undefined;
    const prefix = this.ensurePrefix();
    // First shard whose prefix end exceeds i.
    let lo = 0;
    let hi = this.keys.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (prefix[mid + 1] <= i) lo = mid + 1;
      else hi = mid;
    }
    return this.keys[lo] * 0x10000 + this.containers[lo].select(i - prefix[lo]);
  }

  minimum(): number | undefined {
    if (this._size === 0) return undefined;
    return this.keys[0] * 0x10000 + this.containers[0].min();
  }

  maximum(): number | undefined {
    if (this._size === 0) return undefined;
    const last = this.keys.length - 1;
    return this.keys[last] * 0x10000 + this.containers[last].max();
  }

  // -------------------------------------------------------------------------
  // Iteration
  // -------------------------------------------------------------------------

  *[Symbol.iterator](): IterableIterator<number> {
    for (let k = 0; k < this.keys.length; k++) {
      const base = this.keys[k] * 0x10000;
      for (const lo of this.containers[k]) yield base + lo;
    }
  }

  /**
   * Iterates the values in [start, end] in ascending order. A scratch word
   * buffer is checked out of the pool for bitmap shards and released when
   * the iterator completes or is cancelled early by the caller.
   */
  *iterateRange(start: number, end: number): IterableIterator<number> {
    checkUint32(start, 'start');
    checkUint32(end, 'end');
    if (start > end) throw new RangeError(`empty range [${start}, ${end}]`);
    const scratch = wordPool.acquire();
    try {
      const hi0 = start >>> 16;
      const hi1 = end >>> 16;
      const first = this.findKey(hi0);
      let i = first >= 0 ? first : -first - 1;
      for (; i < this.keys.length && this.keys[i] <= hi1; i++) {
        const key = this.keys[i];
        const lo0 = key === hi0 ? start & 0xffff : 0;
        const lo1 = key === hi1 ? end & 0xffff : 0xffff;
        const base = key * 0x10000;
        const c = this.containers[i];
        if (c instanceof BitmapContainer) {
          for (const lo of c.iterateRange(lo0, lo1, scratch)) yield base + lo;
        } else {
          for (const lo of c.iterateRange(lo0, lo1)) yield base + lo;
        }
      }
    } finally {
      wordPool.release(scratch);
    }
  }

  /** Sorted array of all values. */
  toArray(): number[] {
    const out = new Array<number>(this._size);
    let i = 0;
    for (const v of this) out[i++] = v;
    return out;
  }

  // -------------------------------------------------------------------------
  // Set operations (non-mutating)
  // -------------------------------------------------------------------------

  union(other: RoaringSet): RoaringSet {
    const out = new RoaringSet();
    let i = 0;
    let j = 0;
    while (i < this.keys.length && j < other.keys.length) {
      const ka = this.keys[i];
      const kb = other.keys[j];
      if (ka < kb) {
        out.insertContainerAt(out.keys.length, ka, this.containers[i].clone());
        i++;
      } else if (ka > kb) {
        out.insertContainerAt(out.keys.length, kb, other.containers[j].clone());
        j++;
      } else {
        out.insertContainerAt(out.keys.length, ka, containerOr(this.containers[i], other.containers[j]));
        i++;
        j++;
      }
    }
    while (i < this.keys.length) {
      out.insertContainerAt(out.keys.length, this.keys[i], this.containers[i].clone());
      i++;
    }
    while (j < other.keys.length) {
      out.insertContainerAt(out.keys.length, other.keys[j], other.containers[j].clone());
      j++;
    }
    return out;
  }

  intersection(other: RoaringSet): RoaringSet {
    const out = new RoaringSet();
    let i = 0;
    let j = 0;
    while (i < this.keys.length && j < other.keys.length) {
      const ka = this.keys[i];
      const kb = other.keys[j];
      if (ka < kb) i++;
      else if (ka > kb) j++;
      else {
        const c = containerAnd(this.containers[i], other.containers[j]);
        if (c !== null) out.insertContainerAt(out.keys.length, ka, c);
        i++;
        j++;
      }
    }
    return out;
  }

  difference(other: RoaringSet): RoaringSet {
    const out = new RoaringSet();
    let i = 0;
    let j = 0;
    while (i < this.keys.length && j < other.keys.length) {
      const ka = this.keys[i];
      const kb = other.keys[j];
      if (ka < kb) {
        out.insertContainerAt(out.keys.length, ka, this.containers[i].clone());
        i++;
      } else if (ka > kb) {
        j++;
      } else {
        const c = containerAndNot(this.containers[i], other.containers[j]);
        if (c !== null) out.insertContainerAt(out.keys.length, ka, c);
        i++;
        j++;
      }
    }
    while (i < this.keys.length) {
      out.insertContainerAt(out.keys.length, this.keys[i], this.containers[i].clone());
      i++;
    }
    return out;
  }

  symmetricDifference(other: RoaringSet): RoaringSet {
    const out = new RoaringSet();
    let i = 0;
    let j = 0;
    while (i < this.keys.length && j < other.keys.length) {
      const ka = this.keys[i];
      const kb = other.keys[j];
      if (ka < kb) {
        out.insertContainerAt(out.keys.length, ka, this.containers[i].clone());
        i++;
      } else if (ka > kb) {
        out.insertContainerAt(out.keys.length, kb, other.containers[j].clone());
        j++;
      } else {
        const c = containerXor(this.containers[i], other.containers[j]);
        if (c !== null) out.insertContainerAt(out.keys.length, ka, c);
        i++;
        j++;
      }
    }
    while (i < this.keys.length) {
      out.insertContainerAt(out.keys.length, this.keys[i], this.containers[i].clone());
      i++;
    }
    while (j < other.keys.length) {
      out.insertContainerAt(out.keys.length, other.keys[j], other.containers[j].clone());
      j++;
    }
    return out;
  }

  // -------------------------------------------------------------------------
  // Set operations (in place)
  // -------------------------------------------------------------------------

  unionInPlace(other: RoaringSet): this {
    this.invalidatePrefix();
    if (other === this) return this;
    for (let j = 0; j < other.keys.length; j++) {
      const key = other.keys[j];
      const idx = this.findKey(key);
      if (idx < 0) {
        this.insertContainerAt(-idx - 1, key, other.containers[j].clone());
      } else {
        const old = this.containers[idx];
        const before = old.cardinality;
        const merged = containerIOr(old, other.containers[j]);
        this._size += merged.cardinality - before;
        this.containers[idx] = merged;
      }
    }
    return this;
  }

  intersectionInPlace(other: RoaringSet): this {
    this.invalidatePrefix();
    if (other === this) return this;
    for (let i = this.keys.length - 1; i >= 0; i--) {
      const idx = other.findKey(this.keys[i]);
      if (idx < 0) {
        this.removeContainerAt(i);
        continue;
      }
      const old = this.containers[i];
      const before = old.cardinality;
      const c = containerIAnd(old, other.containers[idx]);
      if (c === null || c.cardinality === 0) {
        this._size -= before; // old may have been mutated empty in place
        this.keys.splice(i, 1);
        this.containers.splice(i, 1);
      } else {
        this._size += c.cardinality - before;
        this.containers[i] = c;
      }
    }
    return this;
  }

  differenceInPlace(other: RoaringSet): this {
    this.invalidatePrefix();
    if (other === this) {
      this.clear();
      return this;
    }
    for (let i = this.keys.length - 1; i >= 0; i--) {
      const idx = other.findKey(this.keys[i]);
      if (idx < 0) continue;
      const old = this.containers[i];
      const before = old.cardinality;
      const c = containerIAndNot(old, other.containers[idx]);
      if (c === null || c.cardinality === 0) {
        this._size -= before;
        this.keys.splice(i, 1);
        this.containers.splice(i, 1);
      } else {
        this._size += c.cardinality - before;
        this.containers[i] = c;
      }
    }
    return this;
  }

  symmetricDifferenceInPlace(other: RoaringSet): this {
    this.invalidatePrefix();
    if (other === this) {
      this.clear();
      return this;
    }
    for (let j = 0; j < other.keys.length; j++) {
      const key = other.keys[j];
      const idx = this.findKey(key);
      if (idx < 0) {
        this.insertContainerAt(-idx - 1, key, other.containers[j].clone());
      } else {
        const old = this.containers[idx];
        const before = old.cardinality;
        const c = containerIXor(old, other.containers[j]);
        if (c === null || c.cardinality === 0) {
          this._size -= before;
          this.keys.splice(idx, 1);
          this.containers.splice(idx, 1);
        } else {
          this._size += c.cardinality - before;
          this.containers[idx] = c;
        }
      }
    }
    return this;
  }

  // -------------------------------------------------------------------------
  // Misc
  // -------------------------------------------------------------------------

  equals(other: RoaringSet): boolean {
    if (this._size !== other._size || this.keys.length !== other.keys.length) return false;
    for (let i = 0; i < this.keys.length; i++) {
      if (this.keys[i] !== other.keys[i]) return false;
      if (containerXor(this.containers[i], other.containers[i]) !== null) return false;
    }
    return true;
  }

  clone(): RoaringSet {
    const out = new RoaringSet();
    out.keys = this.keys.slice();
    out.containers = this.containers.map((c) => c.clone());
    out._size = this._size;
    return out;
  }

  /** Converts shards to run containers where that encoding is smaller. */
  runOptimize(): boolean {
    let changed = false;
    for (let i = 0; i < this.containers.length; i++) {
      const c = this.containers[i];
      if (c.type === ContainerType.Run) continue;
      const nc = normalize(c);
      if (nc !== c) {
        this.containers[i] = nc;
        changed = true;
      }
    }
    return changed;
  }

  /** Trims over-allocated container capacity. */
  shrinkToFit(): void {
    for (let i = 0; i < this.containers.length; i++) {
      const c = this.containers[i];
      if (c instanceof ArrayContainer && c.data.length > c.cardinality) {
        this.containers[i] = ArrayContainer.fromSorted(c.data.slice(0, c.cardinality), c.cardinality);
      } else if (c instanceof RunContainer) {
        this.containers[i] = c.clone();
      }
    }
  }

  /** Diagnostic: per-shard container type names, in key order. */
  debugContainerTypes(): string[] {
    return this.containers.map((c) => ['array', 'bitmap', 'run'][c.type]);
  }

  // -------------------------------------------------------------------------
  // Serialization
  // -------------------------------------------------------------------------

  serializedSizeInBytes(): number {
    const n = this.keys.length;
    let total = headerPaddedSize(n) + n * 4;
    for (const c of this.containers) total += c.serializedSizeInBytes();
    return total;
  }

  serialize(): Uint8Array {
    const n = this.keys.length;
    const buf = new Uint8Array(this.serializedSizeInBytes());
    const view = new DataView(buf.buffer);
    view.setUint32(0, MAGIC, true);
    view.setUint32(4, FORMAT_VERSION, true);
    view.setUint32(8, n, true);
    view.setUint32(12, 0, true);
    let p = HEADER_SIZE;
    for (const k of this.keys) {
      view.setUint16(p, k, true);
      p += 2;
    }
    for (const c of this.containers) {
      view.setUint16(p, c.cardinality - 1, true);
      p += 2;
    }
    for (const c of this.containers) {
      view.setUint8(p, c.type);
      p += 1;
    }
    const offsetsPos = headerPaddedSize(n);
    let offset = offsetsPos + n * 4;
    for (let i = 0; i < n; i++) {
      view.setUint32(offsetsPos + i * 4, offset, true);
      offset += this.containers[i].serializedSizeInBytes();
    }
    let dataOffset = offsetsPos + n * 4;
    for (const c of this.containers) {
      c.serializeInto(view, dataOffset);
      dataOffset += c.serializedSizeInBytes();
    }
    return buf;
  }

  static deserialize(buf: Uint8Array): RoaringSet {
    if (!(buf instanceof Uint8Array)) {
      throw new TypeError('deserialize expects a Uint8Array');
    }
    const len = buf.byteLength;
    const view = new DataView(buf.buffer, buf.byteOffset, len);
    const fail = (msg: string): never => {
      throw new DeserializationError(msg);
    };
    if (len < HEADER_SIZE) fail(`buffer too small for header: ${len} bytes`);
    if (view.getUint32(0, true) !== MAGIC) fail('bad magic number');
    const version = view.getUint32(4, true);
    if (version !== FORMAT_VERSION) fail(`unsupported format version ${version}`);
    if (view.getUint32(12, true) !== 0) fail('unsupported flags');
    const n = view.getUint32(8, true);
    if (n > 0x10000) fail(`invalid container count ${n}`);
    const offsetsPos = headerPaddedSize(n);
    const dataStart = offsetsPos + n * 4;
    if (dataStart > len) fail('truncated header');

    const keys = new Array<number>(n);
    const cards = new Array<number>(n);
    const types = new Array<number>(n);
    let p = HEADER_SIZE;
    for (let i = 0; i < n; i++) {
      keys[i] = view.getUint16(p, true);
      p += 2;
      if (i > 0 && keys[i] <= keys[i - 1]) fail('shard keys not strictly increasing');
    }
    for (let i = 0; i < n; i++) {
      cards[i] = view.getUint16(p, true) + 1;
      p += 2;
    }
    for (let i = 0; i < n; i++) {
      types[i] = view.getUint8(p);
      p += 1;
      if (types[i] > 2) fail(`unknown container type ${types[i]}`);
    }

    // Validate offsets and per-container payload lengths before decoding.
    const offsets = new Array<number>(n);
    for (let i = 0; i < n; i++) offsets[i] = view.getUint32(offsetsPos + i * 4, true);
    if (n > 0 && offsets[0] !== dataStart) fail('first payload offset does not match header size');
    for (let i = 0; i < n; i++) {
      const end = i + 1 < n ? offsets[i + 1] : len;
      if (end < offsets[i] || end > len) fail(`payload ${i} out of bounds`);
      const payloadLen = end - offsets[i];
      if (types[i] === ContainerType.Array) {
        if (payloadLen !== cards[i] * 2) {
          fail(`array payload length ${payloadLen} does not match cardinality ${cards[i]}`);
        }
        if (cards[i] > ARRAY_LIMIT) fail(`array container exceeds ${ARRAY_LIMIT} elements`);
      } else if (types[i] === ContainerType.Bitmap) {
        if (payloadLen !== BITMAP_BYTES) fail(`bitmap payload length ${payloadLen} != ${BITMAP_BYTES}`);
        if (cards[i] <= ARRAY_LIMIT) fail('bitmap container below minimum cardinality');
      } else {
        if (payloadLen < 4 || payloadLen % 4 !== 0) fail(`invalid run payload length ${payloadLen}`);
      }
    }

    // Decode payloads, validating content against the declared cardinality.
    const containers = new Array<Container>(n);
    let size = 0;
    for (let i = 0; i < n; i++) {
      const off = offsets[i];
      if (types[i] === ContainerType.Array) {
        const data = new Uint16Array(cards[i]);
        for (let k = 0; k < cards[i]; k++) {
          data[k] = view.getUint16(off + k * 2, true);
          if (k > 0 && data[k] <= data[k - 1]) fail(`array payload ${i} not strictly increasing`);
        }
        containers[i] = ArrayContainer.fromSorted(data, cards[i]);
      } else if (types[i] === ContainerType.Bitmap) {
        const words = new Uint32Array(BITMAP_WORDS);
        let card = 0;
        for (let w = 0; w < BITMAP_WORDS; w++) {
          words[w] = view.getUint32(off + w * 4, true);
          card += popcount32(words[w]);
        }
        if (card !== cards[i]) {
          fail(`bitmap cardinality mismatch: declared ${cards[i]}, actual ${card}`);
        }
        containers[i] = BitmapContainer.fromWords(words, card);
      } else {
        const payloadLen = (i + 1 < n ? offsets[i + 1] : len) - off;
        const runs = payloadLen / 4;
        const starts = new Uint16Array(runs);
        const lengths = new Uint16Array(runs);
        let card = 0;
        let prevEnd = -2;
        for (let r = 0; r < runs; r++) {
          const s = view.getUint16(off + r * 4, true);
          const l = view.getUint16(off + r * 4 + 2, true);
          if (s + l > MAX_LO) fail(`run payload ${i} extends past 65535`);
          if (s <= prevEnd + 1) fail(`run payload ${i} overlaps or is not sorted`);
          starts[r] = s;
          lengths[r] = l;
          card += l + 1;
          prevEnd = s + l;
        }
        if (card !== cards[i]) {
          fail(`run cardinality mismatch: declared ${cards[i]}, actual ${card}`);
        }
        containers[i] = RunContainer.fromTyped(starts, lengths, runs, card);
      }
      size += cards[i];
    }

    const out = new RoaringSet();
    out.keys = keys;
    out.containers = containers;
    out._size = size;
    return out;
  }
}

/** Header size including the type-table padding, excluding the offsets. */
function headerPaddedSize(n: number): number {
  return (HEADER_SIZE + n * 5 + 3) & ~3;
}
