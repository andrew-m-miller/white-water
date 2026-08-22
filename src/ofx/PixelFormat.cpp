// Vendored from warp-drive 887a123, src/ofx/PixelFormat.cpp.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "ofx/PixelFormat.h"

#include <cmath>
#include <cstring>

#include "core/image/Image.h"

namespace whitewater {
namespace ofx {

namespace {

std::uint32_t floatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bitsToFloat(std::uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Discards the low `shift` bits of value, rounding to nearest and breaking ties towards an
// even result. Ties-to-even rather than ties-away because that is what IEEE arithmetic
// does everywhere else, and a conversion that rounded differently would put a systematic
// upward bias into every half-float render.
std::uint32_t shiftRoundToNearestEven(std::uint32_t value, int shift) {
  const std::uint32_t truncated = value >> shift;
  const std::uint32_t half = std::uint32_t(1) << (shift - 1);
  const std::uint32_t remainder = value & ((std::uint32_t(1) << shift) - 1);
  if (remainder > half || (remainder == half && (truncated & 1) != 0)) {
    return truncated + 1;
  }
  return truncated;
}

// Integer channel scaling. The multiply-then-round on the way out is what makes the round
// trip exact: v/255 is not representable, but it is within half an ulp of a value whose
// product with 255 rounds back to v.
inline float fromUnsignedByte(std::uint8_t value) {
  return static_cast<float>(value) * (1.0f / 255.0f);
}
inline std::uint8_t toUnsignedByte(float value) {
  if (!(value > 0.0f)) return 0;  // catches NaN as well as negatives
  if (value >= 1.0f) return 255;
  return static_cast<std::uint8_t>(value * 255.0f + 0.5f);
}
inline float fromUnsignedShort(std::uint16_t value) {
  return static_cast<float>(value) * (1.0f / 65535.0f);
}
inline std::uint16_t toUnsignedShort(float value) {
  if (!(value > 0.0f)) return 0;
  if (value >= 1.0f) return 65535;
  return static_cast<std::uint16_t>(value * 65535.0f + 0.5f);
}

// How many channels a host layout carries, and where alpha sits in it.
int componentCount(OFX::PixelComponentEnum components) {
  switch (components) {
    case OFX::ePixelComponentRGBA: return 4;
    case OFX::ePixelComponentRGB: return 3;
    case OFX::ePixelComponentAlpha: return 1;
    default: return 0;
  }
}

int depthBytes(OFX::BitDepthEnum depth) {
  switch (depth) {
    case OFX::eBitDepthUByte: return 1;
    case OFX::eBitDepthUShort: return 2;
    case OFX::eBitDepthHalf: return 2;
    case OFX::eBitDepthFloat: return 4;
    default: return 0;
  }
}

// Reads one host channel value as a float. Kept as a template over the storage type so the
// per-row loops below are one loop each rather than one per depth.
template <typename Storage>
float readChannel(const Storage *source, int index);

template <>
float readChannel<std::uint8_t>(const std::uint8_t *source, int index) {
  return fromUnsignedByte(source[index]);
}
template <>
float readChannel<std::uint16_t>(const std::uint16_t *source, int index) {
  return fromUnsignedShort(source[index]);
}
template <>
float readChannel<float>(const float *source, int index) {
  return source[index];
}

// Half is stored in a uint16_t like the short depth is, so it cannot be distinguished by
// type alone; it gets its own loop rather than a fourth specialization.
template <typename Storage>
void readTypedRow(const Storage *source, int channels, int pixelCount, float *rgba) {
  for (int x = 0; x < pixelCount; ++x) {
    float *out = rgba + static_cast<std::ptrdiff_t>(x) * kImageChannels;
    const Storage *in = source + static_cast<std::ptrdiff_t>(x) * channels;
    if (channels == 1) {
      out[0] = 0.0f;
      out[1] = 0.0f;
      out[2] = 0.0f;
      out[3] = readChannel<Storage>(in, 0);
    } else {
      out[0] = readChannel<Storage>(in, 0);
      out[1] = readChannel<Storage>(in, 1);
      out[2] = readChannel<Storage>(in, 2);
      out[3] = channels == 4 ? readChannel<Storage>(in, 3) : 1.0f;
    }
  }
}

void readHalfRow(const std::uint16_t *source, int channels, int pixelCount, float *rgba) {
  for (int x = 0; x < pixelCount; ++x) {
    float *out = rgba + static_cast<std::ptrdiff_t>(x) * kImageChannels;
    const std::uint16_t *in = source + static_cast<std::ptrdiff_t>(x) * channels;
    if (channels == 1) {
      out[0] = 0.0f;
      out[1] = 0.0f;
      out[2] = 0.0f;
      out[3] = halfToFloat(in[0]);
    } else {
      out[0] = halfToFloat(in[0]);
      out[1] = halfToFloat(in[1]);
      out[2] = halfToFloat(in[2]);
      out[3] = channels == 4 ? halfToFloat(in[3]) : 1.0f;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Half float
// ---------------------------------------------------------------------------

float halfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  const std::uint32_t exponent = (value >> 10) & 0x1fu;
  const std::uint32_t mantissa = value & 0x3ffu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return bitsToFloat(sign);  // signed zero
    }
    // Subnormal. Every half subnormal is a normal float, so rather than renormalizing by
    // hand the value is formed directly: the smallest subnormal is 2^-24 and the mantissa
    // counts multiples of it.
    const float magnitude = static_cast<float>(mantissa) * 5.9604644775390625e-08f;
    return sign != 0 ? -magnitude : magnitude;
  }
  if (exponent == 31) {
    return bitsToFloat(sign | 0x7f800000u | (mantissa << 13));  // infinity or NaN
  }
  return bitsToFloat(sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13));
}

std::uint16_t floatToHalf(float value) {
  const std::uint32_t bits = floatBits(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t rawExponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;

  if (rawExponent == 0xff) {
    // A NaN must stay a NaN: truncating its mantissa to zero would turn it into an
    // infinity, so a payload that would vanish is replaced by the canonical quiet NaN.
    if (mantissa != 0) {
      const std::uint32_t truncated = mantissa >> 13;
      return static_cast<std::uint16_t>(sign | 0x7c00u | (truncated != 0 ? truncated : 1u));
    }
    return static_cast<std::uint16_t>(sign | 0x7c00u);
  }

  const int exponent = static_cast<int>(rawExponent) - 127 + 15;
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00u);  // overflow to infinity
  }
  if (exponent <= 0) {
    // Subnormal, or too small to represent at all. The implicit leading one is restored
    // and the whole significand is shifted down to the fixed 2^-24 grid of a half
    // subnormal; a shift past the end of the significand rounds to zero on its own.
    const int shift = 14 - exponent;
    if (shift > 25 || rawExponent == 0) {
      return sign;
    }
    const std::uint32_t significand = mantissa | 0x800000u;
    const std::uint32_t rounded = shiftRoundToNearestEven(significand, shift);
    return static_cast<std::uint16_t>(sign | rounded);
  }

  // Normal. Rounding may carry into the exponent, and because the exponent field sits
  // directly above the mantissa field the carry propagates by plain addition -- including
  // the case where it pushes the exponent to 31, which is exactly the infinity encoding.
  const std::uint32_t rounded = shiftRoundToNearestEven(mantissa, 13);
  return static_cast<std::uint16_t>(sign | ((static_cast<std::uint32_t>(exponent) << 10) +
                                            rounded));
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

int hostPixelBytes(OFX::BitDepthEnum depth, OFX::PixelComponentEnum components) {
  return depthBytes(depth) * componentCount(components);
}

bool isDirectlyUsableLayout(OFX::BitDepthEnum depth, OFX::PixelComponentEnum components,
                            int rowBytes) {
  // The row stride is carried in floats by ImageView, so a stride that is not a whole
  // number of floats cannot be expressed. No real host produces one, but a silent
  // truncation here would corrupt every row after the first.
  return depth == OFX::eBitDepthFloat && components == OFX::ePixelComponentRGBA &&
         rowBytes % static_cast<int>(sizeof(float)) == 0;
}

void readHostRow(const void *source, OFX::BitDepthEnum depth,
                 OFX::PixelComponentEnum components, int pixelCount, float *rgba) {
  const int channels = componentCount(components);
  if (channels == 0 || pixelCount <= 0) {
    return;
  }
  switch (depth) {
    case OFX::eBitDepthUByte:
      readTypedRow(static_cast<const std::uint8_t *>(source), channels, pixelCount, rgba);
      return;
    case OFX::eBitDepthUShort:
      readTypedRow(static_cast<const std::uint16_t *>(source), channels, pixelCount, rgba);
      return;
    case OFX::eBitDepthHalf:
      readHalfRow(static_cast<const std::uint16_t *>(source), channels, pixelCount, rgba);
      return;
    case OFX::eBitDepthFloat:
      readTypedRow(static_cast<const float *>(source), channels, pixelCount, rgba);
      return;
    default:
      return;
  }
}

void writeHostRow(const float *rgba, int pixelCount, OFX::BitDepthEnum depth,
                  OFX::PixelComponentEnum components, void *destination) {
  const int channels = componentCount(components);
  if (channels == 0 || pixelCount <= 0) {
    return;
  }
  // An alpha-only destination takes the rendered alpha and nothing else, so the source
  // channel it reads is channel 3 rather than channel 0.
  const int firstChannel = channels == 1 ? 3 : 0;

  for (int x = 0; x < pixelCount; ++x) {
    const float *in = rgba + static_cast<std::ptrdiff_t>(x) * kImageChannels;
    const std::ptrdiff_t offset = static_cast<std::ptrdiff_t>(x) * channels;
    switch (depth) {
      case OFX::eBitDepthUByte: {
        std::uint8_t *out = static_cast<std::uint8_t *>(destination) + offset;
        for (int c = 0; c < channels; ++c) out[c] = toUnsignedByte(in[firstChannel + c]);
        break;
      }
      case OFX::eBitDepthUShort: {
        std::uint16_t *out = static_cast<std::uint16_t *>(destination) + offset;
        for (int c = 0; c < channels; ++c) out[c] = toUnsignedShort(in[firstChannel + c]);
        break;
      }
      case OFX::eBitDepthHalf: {
        std::uint16_t *out = static_cast<std::uint16_t *>(destination) + offset;
        for (int c = 0; c < channels; ++c) out[c] = floatToHalf(in[firstChannel + c]);
        break;
      }
      case OFX::eBitDepthFloat: {
        float *out = static_cast<float *>(destination) + offset;
        for (int c = 0; c < channels; ++c) out[c] = in[firstChannel + c];
        break;
      }
      default:
        return;
    }
  }
}

}  // namespace ofx
}  // namespace whitewater
