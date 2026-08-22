// Vendored from warp-drive 887a123, src/ofx/PixelFormat.h.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// Converting between a host's pixels and the float RGBA the resampler works on.
//
// A host may hand us any of four depths and any of three component layouts, and Flame
// offers all four depths including half float -- so this is not a theoretical matrix, it
// is one we will be handed. Everything here converts a single row at a time, because that
// is the unit the render threads already work in: a worker converts the rows it renders
// and no other, so there is no serial pass over the frame on either side.
//
// Scaling follows the OFX convention: an 8-bit value covers 0..1 as v/255, a 16-bit value
// as v/65535, and float and half are used as they are. Both integer conversions round to
// nearest on the way out, which makes them exact round trips -- read a byte, write it back
// unchanged and you get the same byte -- and that exactness is what lets an identity warp
// be checked bit for bit at every depth rather than only in float.
//
// Half float is implemented here rather than pulled in from a library. Ilmbase is not a
// dependency this project wants in a plugin that must load into Flame's process with a
// glibc 2.28 baseline, and the conversion is thirty lines of well-understood bit
// manipulation. Both directions are exact where exactness is possible: half -> float
// always, and float -> half for any value that came from a half, which is what makes the
// round trip above hold.

#ifndef WHITEWATER_OFX_PIXELFORMAT_H
#define WHITEWATER_OFX_PIXELFORMAT_H

#include <cstdint>

#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

// ---------------------------------------------------------------------------
// Half float
// ---------------------------------------------------------------------------

// IEEE 754 binary16 -> binary32. Exact for every input, infinities, NaNs and subnormals
// included: every half is representable as a float.
float halfToFloat(std::uint16_t value);

// binary32 -> binary16, round to nearest even. Values too large become infinity and values
// too small become zero, both with the sign preserved, which is what every other
// implementation of this does and what a downstream host expects.
std::uint16_t floatToHalf(float value);

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

// True when the host's own layout is exactly what the resampler wants, so the row
// converters can be skipped and the host's buffer used directly. This is the Flame case
// worth caring about: at float RGBA a warp costs no copy at all.
bool isDirectlyUsableLayout(OFX::BitDepthEnum depth, OFX::PixelComponentEnum components,
                            int rowBytes);

// Bytes per pixel for a host layout, or 0 for one we cannot handle.
int hostPixelBytes(OFX::BitDepthEnum depth, OFX::PixelComponentEnum components);

// Expands pixelCount host pixels into pixelCount interleaved float RGBA values.
//
// RGB gains an opaque alpha, because a three-channel image has full coverage everywhere
// and giving it zero alpha would make the unpremultiplied filter divide by nothing. Alpha
// -only gains black colour: only the alpha channel is written back, so the colour is never
// read, and zero is the cheapest thing to put there.
void readHostRow(const void *source, OFX::BitDepthEnum depth,
                 OFX::PixelComponentEnum components, int pixelCount, float *rgba);

// The inverse. Channels the host layout does not have are dropped.
void writeHostRow(const float *rgba, int pixelCount, OFX::BitDepthEnum depth,
                  OFX::PixelComponentEnum components, void *destination);

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_PIXELFORMAT_H
