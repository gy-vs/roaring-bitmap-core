# roaring-bitmap-core

A 32-bit unsigned integer set for TypeScript / Node.js with **no external
bitmap dependencies**. Values are sharded by their high 16 bits, and every
shard stores its low 16 bits in one of three container encodings that is
selected automatically based on density.

```ts
import { Roaring32 } from "./dist/index.js";

const s = new Roaring32();
s.add(0);
s.add(0xffffffff);
s.addRange(1000, 1_000_000);

s.rank(0xffff);        // number of elements <= 65535
s.select(0);           // smallest element
for (const v of s.range(5_000, 90_000)) console.log(v);

const u = s.union(other);        // functional (non-mutating)
s.addAll(other);                 // in-place
const buf: Buffer = s.serialize();
const back = Roaring32.deserialize(buf);
```

## Design

### Sharding

Every value is split into a **key** (high 16 bits) and a **low** value
(low 16 bits). The set is a `Map<key, Container>`; shards are iterated in
strictly increasing key order and empty shards are deleted.

### Containers

| Container | Encoding | When used |
|-----------|----------|-----------|
| `ArrayContainer` | sorted `Uint16Array` | cardinality ≤ 4096 |
| `BitmapContainer` | 2048 × 32-bit words (8 KiB) | cardinality > 4096, or when runs would be larger |
| `RunContainer` | sorted, disjoint, non-adjacent `[start, length]` runs | when 4·runCount is smaller than the alternatives |

Transitions happen on every mutation, not only at the array/bitmap
threshold:

- array cardinality crosses **4097** → bitmap; bitmap cardinality falls
  back to **4096** → array;
- adding/deleting a range builds runs and is kept when it is denser;
- fragmenting a run (e.g. deleting every other value) converts it to a
  bitmap once its runs grow past 8 KiB, and to an array once cardinality
  is ≤ 4096;
- point inserts/deletes on a run extend, merge or split runs in place;
- `optimize()` re-selects encodings for the current content (e.g. a dense
  bitmap becomes a run).

Every conversion preserves **sorted order** and **cardinality**.

### Set operations

`union`, `intersect`, `difference`, `xor` are dispatched per shard pair
and then per *container-type pair* — values are not always expanded into
arrays:

- array × array: sorted merge with galloping for intersection;
- array × bitmap: bitmap word probes / word set-or-clear;
- array × run: binary-searched run membership and run splitting;
- bitmap × bitmap: one pass of word-wise AND/OR/XOR/ANDNOT;
- bitmap × run: word-mask range application, or word-slice extraction into
  runs for `run − bitmap`;
- run × run: interval sweep (union / intersection / subtraction).

Each result is canonicalised through the density rules above, so a sparse
bitmap intersection collapses back to an array or run container.

All four operations are available:

- functionally: `union`, `intersect`, `difference`, `xor` (inputs untouched);
- in place: `addAll`, `intersectAll`, `deleteAll`, `xorAll`.
  `addAll`/`deleteAll` also accept any `Iterable<number>`.

### Rank / select / iteration

- `rank(v)` — count of elements `≤ v`;
- `select(i)` — the element at 0-based sorted position `i`;
- `range(start, end)` — ascending iterator over the inclusive interval,
  also used by `[Symbol.iterator]()`;
- `min()`, `max()`, `keys()`, `values()`, `entries()`, `forEach()`,
  `toArray()`.

Bitmap shards are iterated in chunks through a small scratch buffer. It is
released in a `finally` block, so cancelling a range iteration with
`break`, `return` or a thrown exception cannot leak it (observable through
`pendingIteratorBuffers`).

### Serialization

`serialize()` / `deserialize()` use a little-endian, **versioned** binary
format (magic `R32S`, version `1`): a 20-byte header, a fixed-size
descriptor table (key, type, cardinality, payload offset per shard) and
contiguous type-specific payloads (strictly increasing `u16` values, 2048
`u32` bitmap words, or `u32`-prefixed `(start, length-1)` runs).

Before materialising anything, the decoder validates:

- magic, version and reserved flags;
- shard count and buffer length (header + table + every payload);
- descriptor types, cardinality ranges, gapless/ordered offsets and that
  there are no trailing bytes;
- strictly increasing shard keys and zero reserved bytes;
- sum-of-container cardinalities against the header;
- array ordering, bitmap popcounts and run ordering / disjointness /
  bounds / length sums.

Any failure throws `DeserializationError`. Deserialized containers are
kept exactly in their declared (canonical) encoding.

See `src/serialize.ts` for the exact layout.

## API

```ts
class Roaring32 implements Iterable<number> {
  constructor(values?: Iterable<number>);
  static from(values: Iterable<number>): Roaring32;

  readonly size: number;
  readonly isEmpty: boolean;
  readonly shardCount: number;
  stats(): { shards: number; array: number; bitmap: number; run: number; cardinality: number };

  add(v: number): boolean;
  addRange(start: number, end: number): void;
  delete(v: number): boolean;
  deleteRange(start: number, end: number): void;
  clear(): void;

  has(v: number): boolean;
  rank(v: number): number;
  select(i: number): number;
  min(): number;
  max(): number;

  range(start: number, end: number): IterableIterator<number>;
  [Symbol.iterator](): IterableIterator<number>;
  keys(); values(); entries(); forEach(cb, thisArg?);
  toArray(): number[];

  union(other: Roaring32): Roaring32;
  intersect(other: Roaring32): Roaring32;
  difference(other: Roaring32): Roaring32;
  xor(other: Roaring32): Roaring32;

  addAll(other: Roaring32 | Iterable<number>): this;
  intersectAll(other: Roaring32): this;
  deleteAll(other: Roaring32 | Iterable<number>): this;
  xorAll(other: Roaring32): this;

  equals(other: Roaring32): boolean;
  clone(): Roaring32;
  optimize(): void;

  serialize(): Buffer;
  static deserialize(data: Buffer | Uint8Array): Roaring32;
}
```

## Layout

```
src/
  bits.ts         # popcount / trailing-zero / bit-mask helpers
  containers.ts   # ArrayContainer, BitmapContainer, RunContainer + transitions
  ops.ts          # container-pair dispatch for union/intersect/difference/xor
  serialize.ts    # versioned binary encode/decode with strict validation
  index.ts        # Roaring32
  test/           # node:test suites, differential against a naive set
```

## Build and test

```bash
npm install
npm run build
npm test
```

The test suite (51 tests, no test framework beyond `node:test`) covers:

- container threshold transitions in both directions and run ↔ array/bitmap
  selection under mutation;
- `0` and `2³²−1`, the full 2³²-element universe, long contiguous ranges;
- all nine container-type combinations for every set operation, plus
  sparse intersections over the whole 32-bit domain;
- functional vs in-place operations and input non-mutation;
- iterator buffer release on `break` and on thrown consumer errors;
- truncated / bad-magic / wrong-version / overlapping-offset / wrong-cardinality
  / unsorted / popcount-mismatched / malformed-run inputs;
- one-million-element sets (contiguous and scattered), with `rank`/`select`
  and iteration differentially checked against a naive sorted array;
- randomized fuzz of mutation and operation sequences.
