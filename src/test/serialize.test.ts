import { test } from "node:test";
import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
import { Roaring32, DeserializationError, SERIALIZATION_VERSION } from "../index";
import { HEADER_SIZE, DESCRIPTOR_SIZE } from "../serialize";
import { mulberry32, randInt } from "./helpers";

function roundtrip(s: Roaring32): Roaring32 {
  const buf = s.serialize();
  const back = Roaring32.deserialize(buf);
  assert.ok(back.equals(s), "round-trip equality");
  assert.deepEqual(back.toArray(), s.toArray(), "round-trip values");
  assert.equal(back.size, s.size);
  return back;
}

/** A set with all three container types plus boundary shards. */
function mixedSet(): Roaring32 {
  const s = new Roaring32();
  s.addAll([1, 2, 3, 70000]); // array shards
  for (let i = 0; i < 5000; i++) s.add(200000 + i * 3); // bitmap shard
  s.addRange(1_000_000, 1_100_000); // run shards
  s.add(0);
  s.add(0xffffffff); // top shard
  return s;
}

test("round-trip: empty, singletons, extremes", () => {
  roundtrip(new Roaring32());
  roundtrip(new Roaring32([0]));
  roundtrip(new Roaring32([0xffffffff]));
  roundtrip(new Roaring32([0, 0xffffffff]));
});

test("round-trip: mixed container types", () => {
  const back = roundtrip(mixedSet());
  const st = back.stats();
  assert.ok(st.array >= 1 && st.bitmap >= 1 && st.run >= 1);
});

test("round-trip: one million dense values", () => {
  const s = new Roaring32();
  s.addRange(0, 999_999);
  roundtrip(s);
});

test("round-trip: random sparse values", () => {
  const rng = mulberry32(1);
  const s = new Roaring32();
  for (let i = 0; i < 50000; i++) s.add(randInt(rng, 0, 0xffffffff));
  roundtrip(s);
});

test("deserialize accepts a plain Uint8Array view", () => {
  const s = mixedSet();
  const buf = s.serialize();
  const view = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  assert.ok(Roaring32.deserialize(view).equals(s));
});

test("rejects truncated buffers", () => {
  const good = mixedSet().serialize();
  assert.throws(() => Roaring32.deserialize(Buffer.alloc(0)), DeserializationError);
  assert.throws(() => Roaring32.deserialize(good.subarray(0, HEADER_SIZE - 1)), DeserializationError);
  assert.throws(() => Roaring32.deserialize(good.subarray(0, HEADER_SIZE)), DeserializationError);
  assert.throws(() => Roaring32.deserialize(good.subarray(0, good.length - 1)), DeserializationError);
  // cut in the middle of the descriptor table
  assert.throws(
    () => Roaring32.deserialize(good.subarray(0, HEADER_SIZE + 3)),
    DeserializationError,
  );
});

test("rejects bad magic and unsupported version", () => {
  const good = mixedSet().serialize();
  const badMagic = Buffer.from(good);
  badMagic.writeUInt32LE(0xdeadbeef, 0);
  assert.throws(() => Roaring32.deserialize(badMagic), DeserializationError);

  const badVersion = Buffer.from(good);
  badVersion.writeUInt16LE(SERIALIZATION_VERSION + 1, 4);
  assert.throws(() => Roaring32.deserialize(badVersion), DeserializationError);

  const badFlags = Buffer.from(good);
  badFlags.writeUInt16LE(1, 6);
  assert.throws(() => Roaring32.deserialize(badFlags), DeserializationError);
});

test("rejects trailing garbage and header cardinality mismatch", () => {
  const good = mixedSet().serialize();
  assert.throws(
    () => Roaring32.deserialize(Buffer.concat([good, Buffer.from([0])])),
    DeserializationError,
  );
  const badCard = Buffer.from(good);
  badCard.writeBigUInt64LE(999999n, 12);
  assert.throws(() => Roaring32.deserialize(badCard), DeserializationError);
});

test("rejects corrupt descriptors", () => {
  const good = mixedSet().serialize();
  const firstDesc = HEADER_SIZE;

  // unknown container type
  const badType = Buffer.from(good);
  badType.writeUInt8(99, firstDesc + 4);
  assert.throws(() => Roaring32.deserialize(badType), DeserializationError);

  // reserved bytes must be zero
  const badReserved = Buffer.from(good);
  badReserved.writeUInt8(1, firstDesc + 5);
  assert.throws(() => Roaring32.deserialize(badReserved), DeserializationError);

  // array cardinality above the array limit (first shard holds 4 values)
  const badArrayCard = Buffer.from(good);
  badArrayCard.writeUInt32LE(5000, firstDesc + 8);
  assert.throws(() => Roaring32.deserialize(badArrayCard), DeserializationError);

  // zero cardinality
  const zeroCard = Buffer.from(good);
  zeroCard.writeUInt32LE(0, firstDesc + 8);
  assert.throws(() => Roaring32.deserialize(zeroCard), DeserializationError);

  // offset pointing outside the buffer
  const badOffset = Buffer.from(good);
  badOffset.writeUInt32LE(0xfffff000, firstDesc + 12);
  assert.throws(() => Roaring32.deserialize(badOffset), DeserializationError);

  // overlapping offsets: second descriptor points at the first payload
  const overlap = Buffer.from(good);
  const firstOffset = good.readUInt32LE(firstDesc + 12);
  overlap.writeUInt32LE(firstOffset, firstDesc + DESCRIPTOR_SIZE + 12);
  assert.throws(() => Roaring32.deserialize(overlap), DeserializationError);

  // shard key out of range
  const badKey = Buffer.from(good);
  badKey.writeUInt32LE(0x10000, firstDesc);
  assert.throws(() => Roaring32.deserialize(badKey), DeserializationError);
});

test("rejects corrupt array payload (unsorted values)", () => {
  const s = new Roaring32([10, 20, 30, 40]);
  const good = s.serialize();
  const payload = good.readUInt32LE(HEADER_SIZE + 12);
  const corrupted = Buffer.from(good);
  corrupted.writeUInt16LE(30, payload); // was 10 -> now 30, 20, 30, 40
  corrupted.writeUInt16LE(20, payload + 4); // -> 30, 20, 30, 40? make it strictly unsorted
  assert.throws(() => Roaring32.deserialize(corrupted), DeserializationError);
});

test("rejects corrupt bitmap payload (popcount mismatch)", () => {
  const s = new Roaring32();
  for (let i = 0; i < 5000; i++) s.add(i * 3);
  const good = s.serialize();
  const payload = good.readUInt32LE(HEADER_SIZE + 12);
  const corrupted = Buffer.from(good);
  corrupted.writeUInt32LE(corrupted.readUInt32LE(payload) ^ 1, payload); // flip one bit
  assert.throws(() => Roaring32.deserialize(corrupted), DeserializationError);
});

test("rejects corrupt run payloads", () => {
  const s = new Roaring32();
  s.addRange(100, 5000); // 4901 values -> single run [100, 5000] in shard 0
  const good = s.serialize();
  const payload = good.readUInt32LE(HEADER_SIZE + 12);

  // run extending past 65535
  const overflow = Buffer.from(good);
  overflow.writeUInt16LE(65535, payload);
  overflow.writeUInt16LE(5, payload + 2); // length 6 -> end 65540
  assert.throws(() => Roaring32.deserialize(overflow), DeserializationError);

  // adjacent runs: [100,3000] and [3001,5000] should have been coalesced
  const headerAndDesc = good.subarray(0, HEADER_SIZE + DESCRIPTOR_SIZE);
  const crafted = Buffer.alloc(4 + 2 * 8);
  crafted.writeUInt32LE(2, 0);
  crafted.writeUInt16LE(100, 4);
  crafted.writeUInt16LE(2900, 6); // length 2901 -> [100,3000]
  crafted.writeUInt16LE(3001, 12);
  crafted.writeUInt16LE(1999, 14); // length 2000 -> [3001,5000]
  const adjacent = Buffer.concat([Buffer.from(headerAndDesc), crafted]);
  assert.throws(() => Roaring32.deserialize(adjacent), DeserializationError);

  // run lengths that do not sum to the descriptor cardinality
  const badSum = Buffer.from(good);
  badSum.writeUInt16LE(50, payload + 2); // length 51 instead of 4901
  assert.throws(() => Roaring32.deserialize(badSum), DeserializationError);
});

test("rejects bitmap cardinality below the threshold in descriptor", () => {
  const s = new Roaring32();
  for (let i = 0; i < 5000; i++) s.add(i * 3);
  const good = s.serialize();
  const corrupted = Buffer.from(good);
  corrupted.writeUInt32LE(100, HEADER_SIZE + 8); // bitmap with card 100 is non-canonical
  assert.throws(() => Roaring32.deserialize(corrupted), DeserializationError);
});

test("serialized size is reasonable for compressed runs", () => {
  const s = new Roaring32();
  s.addRange(0, 999_999);
  const buf = s.serialize();
  // 16 shards: 15 full runs + 1 partial run -> tiny payload
  assert.ok(buf.length < 1024, `expected compact encoding, got ${buf.length} bytes`);
});
