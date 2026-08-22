// Vendored from warp-drive 887a123, src/ofx/HostImage.h.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// Presenting a host's image to the resampler, and taking the result back.
//
// The resampler works on interleaved float RGBA with a row stride. When a host hands us
// exactly that -- which is the case worth optimizing, because it is what Flame does at
// float depth -- these adapters point straight at the host's memory and no pixel is copied
// in either direction. Any other depth or component layout is converted through scratch
// storage, per row, by the thread that renders that row.
//
// Coordinates are the point of this file. OFX gives an image a bounds rectangle in pixel
// coordinates and a data pointer that addresses the pixel at (bounds.x1, bounds.y1); the
// warp is solved in that same pixel space. So each adapter reports the coordinate of its
// row 0 and column 0, and the caller hands those to ResampleGeometry -- which is precisely
// what ResampleGeometry exists for. Nothing here computes an offset of its own, and the
// render path contains no coordinate arithmetic at all.
//
// The destination adapter is deliberately narrowed to the render window rather than the
// whole image. Tiles are on in Flame, so the window is routinely a strip of the frame, and
// a scratch buffer the size of the frame would be wasted on every tile.

#ifndef WHITEWATER_OFX_HOSTIMAGE_H
#define WHITEWATER_OFX_HOSTIMAGE_H

#include <string>
#include <vector>

#include "core/geom/Vec2.h"
#include "core/image/Image.h"
#include "core/warp/Resampler.h"
#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

// The subset of an OFX image descriptor needed by the float-RGBA adapter.  Keeping this
// small value separate from OFX::Image lets frame-capture tests model a host's
// clipGetImage result without copying any conversion rules out of HostImage.
struct HostImageData {
  const void *pixels = nullptr;
  OfxRectI bounds = {0, 0, 0, 0};
  int rowBytes = 0;
  OFX::BitDepthEnum depth = OFX::eBitDepthNone;
  OFX::PixelComponentEnum components = OFX::ePixelComponentNone;
  OFX::PreMultiplicationEnum premultiplication = OFX::eImageUnPreMultiplied;
  double pixelAspectRatio = 1.0;
};

// How the host says the pixels are stored. There is no default: the resampler refuses one
// too, for the same reason -- either answer is silently wrong for half the callers and the
// symptom is a fringe nobody notices until a grade.
AlphaMode alphaModeFor(OFX::PreMultiplicationEnum premultiplication);

// A host image read as float RGBA.
class HostSourceImage {
 public:
  // Returns false, with a reason, for a layout we cannot read at all. An image the host
  // failed to give us is the caller's problem, not this one's: pass a valid image.
  bool attach(const OFX::Image &image, std::string *error);

  // The same adapter path, with the image properties already extracted.  This is useful at
  // another host boundary (for example a supervised frame service) and deliberately shares
  // the implementation above rather than reimplementing depth/component conversion.
  bool attach(const HostImageData &image, std::string *error);

  ConstImageView view() const { return view_; }
  bool isBorrowed() const { return storage_.empty(); }

  // The warp-space position of the lower-left corner of pixel (0, 0) of the view.
  Vec2 origin() const;

 private:
  std::vector<float> storage_;
  ConstImageView view_;
  OfxRectI bounds_ = {0, 0, 0, 0};
};

// The render window of a host image, written as float RGBA.
class HostDestinationImage {
 public:
  bool attach(OFX::Image &image, const OfxRectI &window, std::string *error);

  ImageView view() const { return view_; }
  bool needsWriteBack() const { return !storage_.empty(); }

  Vec2 origin() const;

  // Converts rows [firstRow, firstRow + rowCount) of the view back into the host's buffer.
  // Rows are indexed from the bottom of the render window, matching the row range the
  // resampler was given, so a worker can write back exactly what it rendered. A no-op when
  // the host's buffer was written through directly.
  void writeBackRows(int firstRow, int rowCount) const;

 private:
  std::vector<float> storage_;
  ImageView view_;
  OfxRectI window_ = {0, 0, 0, 0};

  // Where row 0, column 0 of the window lives in the host's buffer, and how to step.
  void *hostRowZero_ = nullptr;
  int hostRowBytes_ = 0;
  OFX::BitDepthEnum depth_ = OFX::eBitDepthNone;
  OFX::PixelComponentEnum components_ = OFX::ePixelComponentNone;
};

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_HOSTIMAGE_H
