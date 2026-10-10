/**
 * IEEE half floats, as onnxruntime-web carries them (`Uint16Array` bits): a graph exported in
 * float16 throughout takes and returns those.
 */

const view = new DataView(new ArrayBuffer(4));

export function halfToFloat(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction ? NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

/** Round to nearest even, as NumPy's `astype(float16)`. */
export function floatToHalf(value: number): number {
  view.setFloat32(0, value);
  const x = view.getUint32(0);
  const sign = (x >>> 16) & 0x8000;
  const exponent = (x >>> 23) & 0xff;
  let mantissa = x & 0x7fffff;
  if (exponent === 0xff) return sign | 0x7c00 | (mantissa ? 0x200 : 0);
  let e = exponent - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mantissa |= 0x800000;
    const shift = 14 - e;
    const half = 1 << (shift - 1);
    let m = mantissa >> shift;
    const rest = mantissa & ((1 << shift) - 1);
    if (rest > half || (rest === half && m & 1)) m += 1;
    return sign | m;
  }
  let m = mantissa >> 13;
  const rest = mantissa & 0x1fff;
  if (rest > 0x1000 || (rest === 0x1000 && m & 1)) {
    m += 1;
    if (m === 0x400) {
      m = 0;
      e += 1;
      if (e >= 0x1f) return sign | 0x7c00;
    }
  }
  return sign | (e << 10) | m;
}

export function toHalf(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = floatToHalf(values[i]!);
  return out;
}

export function fromHalf(bits: ArrayLike<number>): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i += 1) out[i] = halfToFloat(bits[i]!);
  return out;
}
