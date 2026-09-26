import { Buffer } from "node:buffer";
import {
  ArrayContainer,
  BitmapContainer,
  RunContainer,
  Container,
  Run,
  TYPE_ARRAY,
  TYPE_BITMAP,
  TYPE_RUN,
  ARRAY_LIMIT,
  BITMAP_WORDS,
  BITMAP_BYTES,
  SHARD_CAPACITY,
} from "./containers";
import { popcount32 } from "./bits";

/**
 * Versioned binary format (all integers little-endian):
 *
 *   header (20 bytes)
 *     0  u32  magic "R32S" (0x52333253)
 *     4  u16  format version (currently 1)
 *     6  u16  flags (reserved, must be 0)
 *     8  u32  shard count
 *    12  u64  total cardinality
 *   descriptor table (shardCount x 16 bytes)
 *    +0  u32  shard key (high 16 bits), strictly increasing
 *    +4  u8   container type (1 = array, 2 = bitmap, 3 = run)
 *    +5  3    reserved, must be 0
 *    +8  u32  container cardinality
 *   +12  u32  payload offset from buffer start; payloads are contiguous
 *             and gapless in descriptor order
 *   payloads
 *     array  : cardinality x u16, strictly increasing
 *     bitmap : 2048 x u32 words; popcount must equal the cardinality
 *     run    : u32 runCount, then runCount x (u16 start, u16 length-1, 4 reserved)
 *
 * Decoding validates every offset, length and cardinality (plus ordering
 * and encoding invariants) before materialising any container.
 */

export const SERIALIZATION_VERSION = 1;
export const SERIALIZATION_MAGIC = 0x52333253; // "R32S"

export const HEADER_SIZE = 20;
export const DESCRIPTOR_SIZE = 16;
/** Maximum number of non-adjacent runs in a 16-bit shard (singletons at even positions). */
export const MAX_RUNS = SHARD_CAPACITY / 2;

export class DeserializationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DeserializationError";
  }
}

function corrupt(message: string): never {
  throw new DeserializationError(message);
}

export function encodeSet(shards: Map<number, Container>, cardinality: number): Buffer {
  const keys = Array.from(shards.keys()).sort((a, b) => a - b);
  const header = Buffer.alloc(HEADER_SIZE + keys.length * DESCRIPTOR_SIZE);
  header.writeUInt32LE(SERIALIZATION_MAGIC, 0);
  header.writeUInt16LE(SERIALIZATION_VERSION, 4);
  header.writeUInt16LE(0, 6); // flags
  header.writeUInt32LE(keys.length, 8);
  header.writeBigUInt64LE(BigInt(cardinality), 12);

  const payloads: Buffer[] = [];
  let offset = HEADER_SIZE + keys.length * DESCRIPTOR_SIZE;
  for (let i = 0; i < keys.length; i++) {
    const c = shards.get(keys[i])!;
    const payload = encodeContainer(c);
    const d = HEADER_SIZE + i * DESCRIPTOR_SIZE;
    header.writeUInt32LE(keys[i], d);
    header.writeUInt8(c.type, d + 4);
    // d + 5..7 reserved: already zero
    header.writeUInt32LE(c.cardinality, d + 8);
    header.writeUInt32LE(offset, d + 12);
    offset += payload.length;
    payloads.push(payload);
  }
  return Buffer.concat([header, ...payloads]);
}

function encodeContainer(c: Container): Buffer {
  if (c.type === TYPE_ARRAY) {
    const buf = Buffer.alloc(c.cardinality * 2);
    for (let i = 0; i < c.cardinality; i++) buf.writeUInt16LE(c.values[i], i * 2);
    return buf;
  }
  if (c.type === TYPE_BITMAP) {
    const buf = Buffer.alloc(BITMAP_BYTES);
    for (let i = 0; i < BITMAP_WORDS; i++) buf.writeUInt32LE(c.words[i], i * 4);
    return buf;
  }
  const buf = Buffer.alloc(4 + c.runs.length * 8);
  buf.writeUInt32LE(c.runs.length, 0);
  for (let i = 0; i < c.runs.length; i++) {
    buf.writeUInt16LE(c.runs[i].start, 4 + i * 8);
    buf.writeUInt16LE(c.runs[i].length - 1, 4 + i * 8 + 2);
    // 4 bytes reserved per entry: already zero
  }
  return buf;
}

interface Descriptor {
  key: number;
  type: number;
  card: number;
  offset: number;
  length: number;
}

export function decodeSet(input: Buffer | Uint8Array): {
  shards: Map<number, Container>;
  cardinality: number;
} {
  const buf = Buffer.isBuffer(input)
    ? input
    : Buffer.from(input.buffer, input.byteOffset, input.byteLength);

  // ---- structural validation: header, offsets, lengths, cardinalities ----
  if (buf.length < HEADER_SIZE) corrupt(`buffer too small for header: ${buf.length} bytes`);
  if (buf.readUInt32LE(0) !== SERIALIZATION_MAGIC) corrupt("bad magic number");
  const version = buf.readUInt16LE(4);
  if (version !== SERIALIZATION_VERSION) corrupt(`unsupported format version ${version}`);
  if (buf.readUInt16LE(6) !== 0) corrupt("reserved header flags must be zero");
  const shardCount = buf.readUInt32LE(8);
  if (shardCount > SHARD_CAPACITY) corrupt(`shard count ${shardCount} out of range`);
  const declaredCardinality = Number(buf.readBigUInt64LE(12));
  if (!Number.isSafeInteger(declaredCardinality) || declaredCardinality > 0x100000000) {
    corrupt(`total cardinality ${declaredCardinality} out of range`);
  }
  const payloadBase = HEADER_SIZE + shardCount * DESCRIPTOR_SIZE;
  if (buf.length < payloadBase) corrupt("buffer too small for the descriptor table");

  const descriptors: Descriptor[] = [];
  let expectedOffset = payloadBase;
  let totalCardinality = 0;
  let prevKey = -1;
  for (let i = 0; i < shardCount; i++) {
    const d = HEADER_SIZE + i * DESCRIPTOR_SIZE;
    const key = buf.readUInt32LE(d);
    const type = buf.readUInt8(d + 4);
    const card = buf.readUInt32LE(d + 8);
    const offset = buf.readUInt32LE(d + 12);

    if (key > 0xffff) corrupt(`descriptor ${i}: shard key ${key} out of range`);
    if (key <= prevKey) corrupt(`descriptor ${i}: shard keys not strictly increasing`);
    prevKey = key;
    if (buf.readUInt8(d + 5) !== 0 || buf.readUInt8(d + 6) !== 0 || buf.readUInt8(d + 7) !== 0) {
      corrupt(`descriptor ${i}: reserved bytes must be zero`);
    }

    let length: number;
    if (type === TYPE_ARRAY) {
      if (card < 1 || card > ARRAY_LIMIT) {
        corrupt(`descriptor ${i}: array cardinality ${card} out of range [1, ${ARRAY_LIMIT}]`);
      }
      length = card * 2;
    } else if (type === TYPE_BITMAP) {
      if (card <= ARRAY_LIMIT || card > SHARD_CAPACITY) {
        corrupt(
          `descriptor ${i}: bitmap cardinality ${card} out of range (${ARRAY_LIMIT}, ${SHARD_CAPACITY}]`,
        );
      }
      length = BITMAP_BYTES;
    } else if (type === TYPE_RUN) {
      if (card < 1 || card > SHARD_CAPACITY) {
        corrupt(`descriptor ${i}: run cardinality ${card} out of range [1, ${SHARD_CAPACITY}]`);
      }
      if (offset + 4 > buf.length) corrupt(`descriptor ${i}: run header out of bounds`);
      const runCount = buf.readUInt32LE(offset);
      if (runCount < 1 || runCount > MAX_RUNS) {
        corrupt(`descriptor ${i}: run count ${runCount} out of range [1, ${MAX_RUNS}]`);
      }
      if (card < runCount) corrupt(`descriptor ${i}: cardinality ${card} < run count ${runCount}`);
      if (card > runCount * SHARD_CAPACITY) {
        corrupt(`descriptor ${i}: cardinality ${card} infeasible for ${runCount} runs`);
      }
      length = 4 + runCount * 8;
    } else {
      corrupt(`descriptor ${i}: unknown container type ${type}`);
    }

    if (offset !== expectedOffset) {
      corrupt(`descriptor ${i}: payload offset ${offset}, expected ${expectedOffset} (contiguous)`);
    }
    if (offset + length > buf.length) {
      corrupt(`descriptor ${i}: payload [${offset}, ${offset + length}) exceeds buffer ${buf.length}`);
    }
    expectedOffset = offset + length;
    totalCardinality += card;
    descriptors.push({ key, type, card, offset, length });
  }
  if (expectedOffset !== buf.length) {
    corrupt(`${buf.length - expectedOffset} trailing bytes after the last payload`);
  }
  if (totalCardinality !== declaredCardinality) {
    corrupt(`cardinality mismatch: header says ${declaredCardinality}, containers sum to ${totalCardinality}`);
  }

  // ---- validation passed: decode with per-container content checks ----
  const shards = new Map<number, Container>();
  for (const d of descriptors) shards.set(d.key, decodeContainer(buf, d));
  return { shards, cardinality: totalCardinality };
}

function decodeContainer(buf: Buffer, d: Descriptor): Container {
  if (d.type === TYPE_ARRAY) {
    const values = new Uint16Array(Math.max(4, d.card));
    let prev = -1;
    for (let i = 0; i < d.card; i++) {
      const v = buf.readUInt16LE(d.offset + i * 2);
      if (v <= prev) corrupt("array payload is not strictly increasing");
      values[i] = v;
      prev = v;
    }
    return new ArrayContainer(values, d.card);
  }
  if (d.type === TYPE_BITMAP) {
    const words = new Uint32Array(BITMAP_WORDS);
    let card = 0;
    for (let i = 0; i < BITMAP_WORDS; i++) {
      const w = buf.readUInt32LE(d.offset + i * 4);
      words[i] = w;
      card += popcount32(w);
    }
    if (card !== d.card) {
      corrupt(`bitmap popcount ${card} does not match descriptor cardinality ${d.card}`);
    }
    return new BitmapContainer(words, card);
  }
  // run container
  const runCount = buf.readUInt32LE(d.offset);
  const runs: Run[] = [];
  let card = 0;
  let prevEnd = -2;
  for (let i = 0; i < runCount; i++) {
    const start = buf.readUInt16LE(d.offset + 4 + i * 8);
    const length = buf.readUInt16LE(d.offset + 4 + i * 8 + 2) + 1;
    const end = start + length - 1;
    if (end > 0xffff) corrupt(`run ${i} extends past 65535`);
    if (start <= prevEnd + 1) corrupt(`run ${i} overlaps or is adjacent to the previous run`);
    runs.push({ start, length });
    card += length;
    prevEnd = end;
  }
  if (card !== d.card) {
    corrupt(`run lengths sum to ${card}, descriptor cardinality is ${d.card}`);
  }
  return new RunContainer(runs, card);
}
