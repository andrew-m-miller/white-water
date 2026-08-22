// Vendored from warp-drive 887a123, src/ofx/FrameCapture.h.
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
#include "core/image/OwnedFrame.h"
#include "ofx/HostImage.h"
#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

// The frame value and its format/bounds types are defined in core/image/OwnedFrame.h so
// inference code can include them without pulling OFX into the host-free layer. Re-export
// the old names in this namespace for callers that used the Phase 0 boundary API.
using ::whitewater::CapturedAlphaAssociation;
using ::whitewater::CapturedFrame;
using ::whitewater::CapturedPixelBounds;
using ::whitewater::CapturedPixelComponents;
using ::whitewater::CapturedPixelDepth;
using ::whitewater::CapturedPixelFormat;

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
