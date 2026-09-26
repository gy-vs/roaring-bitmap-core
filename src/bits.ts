/** 32-bit bit helpers shared by the containers. */

/** Population count of a 32-bit word (signed or unsigned interpretation). */
export function popcount32(x: number): number {
  x = x | 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return (x * 0x01010101) >>> 24;
}

/** Count of trailing zeros of a non-zero 32-bit word. */
export function ctz32(x: number): number {
  // x & -x isolates the lowest set bit, even for negative (two's complement) inputs.
  return 31 - Math.clz32((x & -x) | 0);
}

/** Mask with bits `lo..hi` (inclusive, 0 <= lo <= hi <= 31) set. */
export function wordRangeMask(lo: number, hi: number): number {
  const loMask = (0xffffffff << lo) | 0;
  const hiMask = hi === 31 ? -1 : ((1 << (hi + 1)) - 1) | 0;
  return (loMask & hiMask) | 0;
}
