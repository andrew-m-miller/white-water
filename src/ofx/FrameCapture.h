// Vendored from warp-drive 887a123, src/ofx/FrameCapture.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// A host-side source frame pulled for the external editor.
//
// The capture value owns its pixels and carries the image facts that a later transport
// boundary must not infer: pixel bounds/origin, the source layout and alpha association,
// and pixel aspect ratio.  It deliberately has no OFX or filesystem state.  The one OFX
// operation in this slice is captureSourceFrame(), whose only job is to fetch and release
// an image around this value conversion.

#ifndef WHITEWATER_OFX_FRAMECAPTURE_H
#define WHITEWATER_OFX_FRAMECAPTURE_H

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "core/geom/Vec2.h"
#include "core/warp/Resampler.h"
#include "ofx/HostImage.h"
#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

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

// A complete, host-free source frame.  Pixels are packed RGBA floats in rows whose first
// row is the image's bounds.y1 row; origin is the coordinate of pixel (0, 0) in that view.
// The source format remains explicit because converting to float RGBA does not erase whether
// colour was originally byte/half/float, RGB/RGBA/alpha, or premultiplied.
struct CapturedFrame {
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

struct FrameCaptureResult {
  std::optional<CapturedFrame> frame;
  std::string error;

  explicit operator bool() const { return frame.has_value(); }
};

// Converts an already-described host image through HostSourceImage's shared conversion
// machinery and copies every row before the caller can release the host image.  This is the
// discriminating, host-free half of the seam and is also useful to a future frame service.
FrameCaptureResult captureHostImage(const HostImageData &image);

// Pulls the clip's complete image at one exact time.  The returned value owns all pixels;
// the OfxImage is released before this function returns.  No request/response or file naming
// policy belongs here.
FrameCaptureResult captureSourceFrame(OFX::Clip &sourceClip, double time);

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_FRAMECAPTURE_H
