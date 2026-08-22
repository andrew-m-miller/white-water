// Derived from warp-drive 887a123, src/ofx/FrameCapture.h.
// Host-free owned frame value produced by an OFX capture boundary.
//
// The OFX image lifetime and pixel conversion belong in src/ofx/FrameCapture.{h,cpp}.
// This value deliberately lives in core so inference code can consume an immutable,
// host-independent frame without including an OFX header or support library.

#ifndef WHITEWATER_CORE_IMAGE_OWNEDFRAME_H
#define WHITEWATER_CORE_IMAGE_OWNEDFRAME_H

#include <cstddef>
#include <vector>

#include "core/geom/Vec2.h"
#include "core/image/Image.h"

namespace whitewater {

enum class CapturedPixelDepth {
  kUnknown,
  kByte,
  kShort,
  kHalf,
  kFloat,
  kCustom,
};

enum class CapturedPixelComponents {
  kUnknown,
  kRGBA,
  kRGB,
  kAlpha,
  kCustom,
};

enum class CapturedAlphaAssociation {
  kOpaque,
  kPremultiplied,
  kUnpremultiplied,
};

struct CapturedPixelFormat {
  CapturedPixelDepth depth = CapturedPixelDepth::kUnknown;
  CapturedPixelComponents components = CapturedPixelComponents::kUnknown;
  CapturedAlphaAssociation alpha = CapturedAlphaAssociation::kUnpremultiplied;
};

struct CapturedPixelBounds {
  int x1 = 0;
  int y1 = 0;
  int x2 = 0;
  int y2 = 0;

  int width() const { return x2 - x1; }
  int height() const { return y2 - y1; }
  bool isEmpty() const { return width() <= 0 || height() <= 0; }
};

inline bool operator==(const CapturedPixelBounds &a, const CapturedPixelBounds &b) {
  return a.x1 == b.x1 && a.y1 == b.y1 && a.x2 == b.x2 && a.y2 == b.y2;
}

// A complete source frame whose pixels outlive the host image that supplied them.
// Pixels are packed RGBA floats in rows whose first row is bounds.y1; origin is the
// coordinate of pixel (0, 0) in that view. The source format remains explicit because
// converting to float RGBA does not erase whether the source was byte/half/float,
// RGB/RGBA/alpha, or premultiplied.
struct OwnedFrame {
  std::vector<float> rgba;
  CapturedPixelBounds bounds;
  Vec2 origin;
  std::ptrdiff_t rowStride = 0;  // float elements, not bytes
  CapturedPixelFormat sourceFormat;
  double pixelAspectRatio = 1.0;

  int width() const { return bounds.width(); }
  int height() const { return bounds.height(); }
  bool isEmpty() const { return bounds.isEmpty() || rgba.empty(); }

  const float *row(int y) const {
    return rgba.data() + static_cast<std::ptrdiff_t>(y) * rowStride;
  }
  const float *pixel(int x, int y) const {
    return row(y) + static_cast<std::ptrdiff_t>(x) * kImageChannels;
  }
};

// Keep the Phase 0 capture vocabulary source-compatible while new inference code can use
// the architectural name that makes ownership explicit.
using CapturedFrame = OwnedFrame;

}  // namespace whitewater

#endif  // WHITEWATER_CORE_IMAGE_OWNEDFRAME_H
