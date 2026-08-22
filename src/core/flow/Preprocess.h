// Host-free image preparation for the pairwise flow boundary.
//
// This file intentionally contains no model names and no input-curve vocabulary.  A
// model's tensor contract belongs above this layer; the values here describe only
// coordinate transforms that remain valid when the estimator is replaced.

#ifndef WHITEWATER_CORE_FLOW_PREPROCESS_H
#define WHITEWATER_CORE_FLOW_PREPROCESS_H

#include <cstdint>

#include "core/flow/Field.h"
#include "core/image/Image.h"
#include "core/image/OwnedFrame.h"

namespace whitewater {

// The rectangle of the unpadded analysis image inside a prepared image.  x/y are local
// pixel indices in the prepared buffer, not source-image coordinates.  The four explicit
// sides matter: callers can ask for a non-zero left/bottom margin and the amount needed to
// reach a multiple is then recorded on the opposite side.
struct CropMetadata {
  int x = 0;
  int y = 0;
  int width = 0;
  int height = 0;
  int paddedWidth = 0;
  int paddedHeight = 0;

  int padLeft = 0;
  int padRight = 0;
  int padBottom = 0;
  int padTop = 0;

  bool isEmpty() const { return width <= 0 || height <= 0; }
  int right() const { return x + width; }
  int top() const { return y + height; }
};

// Geometry before the model-required reflect padding.  `analysisWidth` and
// `analysisHeight` are storage-pixel dimensions after PAR normalization and the optional
// megapixel cap.  scaleX/Y map source pixel distances to those analysis pixels; they are
// deliberately independent because PAR and integer rounding make a single scale lossy.
struct AnalysisGeometry {
  CapturedPixelBounds sourceBounds;
  int sourceWidth = 0;
  int sourceHeight = 0;

  int analysisWidth = 0;
  int analysisHeight = 0;
  int paddedWidth = 0;
  int paddedHeight = 0;

  double pixelAspectRatio = 1.0;
  double canonicalWidth = 0.0;
  double canonicalHeight = 0.0;
  double scaleX = 1.0;
  double scaleY = 1.0;

  CropMetadata crop;

  // Lattice nodes for a cropped model output, expressed in full-resolution storage-pixel
  // coordinates. A model displacement is multiplied by these independent spacings on the
  // way into a FlowField, after which no consumer needs to know PAR or analysis scale.
  FieldGeometry fieldGeometry() const;
};

// Configuration for the model-independent transform.  A zero megapixel cap means "do
// not reduce"; a positive cap is in megapixels (for example 2.0 means two million
// canonical square pixels).  padRight/padTop are minimum requested sides; any additional
// pixels required to make the final extent divisible by padMultiple are added there.  This
// gives a deterministic, asymmetric right/top default while still allowing callers to
// request a left/bottom margin.
struct PreprocessConfig {
  bool premultiplyByMatte = false;
  double megapixelCap = 0.0;

  int padMultiple = 1;
  int padLeft = 0;
  int padRight = 0;
  int padBottom = 0;
  int padTop = 0;

  // The token is intentionally opaque to this phase.  It lets a caller carry a stable
  // conditioning identity into a cache key without this layer naming or ordering any
  // bake-off candidate.
  std::uint64_t transformToken = 0;
};

// The result owns the padded pixels and retains all metadata needed to crop the model
// result back into analysis geometry.  `image` is packed RGBA float, bottom-row first,
// just like core::Image.
struct PreparedImage {
  Image image;
  AnalysisGeometry geometry;
  bool premultiplied = false;
  std::uint64_t transformToken = 0;

  bool isEmpty() const { return image.isEmpty() || geometry.crop.isEmpty(); }

  // A view of the unpadded analysis image.  It aliases `image`; no copy is made.
  ConstImageView analysisView() const;

  // Copy the unpadded image.  This is useful at an inference boundary that does not want
  // to retain the padded storage while a result is consumed.
  Image crop() const;
};

// Computes PAR-normalized and cap-limited analysis dimensions and the exact crop/pad
// record.  Invalid PAR, cap, padding, or source bounds raise std::invalid_argument;
// malformed/empty frames are represented by an empty geometry instead.
AnalysisGeometry analysisGeometry(const OwnedFrame &frame, const PreprocessConfig &config);

// Converts an owned frame to a packed RGBA image, optionally multiplying RGB by its alpha
// matte.  The source alpha association is respected when the option is enabled: already
// premultiplied or opaque input is not multiplied a second time.
Image frameImage(const OwnedFrame &frame, bool premultiplyByMatte = false);

// Reflect-pad a packed image.  The returned crop identifies the exact location of the
// input image in the padded result.  Padding reflects about edge pixels without repeating
// them: for source indices 0,1,2 the samples immediately outside are 1 and then 2.
Image reflectPad(const Image &source, const CropMetadata &crop);

// Remove the padding described by `crop` and return a packed copy.  Invalid crop metadata
// yields an empty image rather than reading outside the input.
Image cropImage(const Image &padded, const CropMetadata &crop);

// Full model-independent preparation: frame conversion, square-pixel sizing, bilinear
// resize, and reflect padding.  The output image is always the padded image.
PreparedImage preprocess(const OwnedFrame &frame, const PreprocessConfig &config = {});

// Naming aliases for callers that prefer the architectural noun.
inline PreparedImage prepareImage(const OwnedFrame &frame, const PreprocessConfig &config = {}) {
  return preprocess(frame, config);
}

inline PreparedImage prepare(const OwnedFrame &frame, const PreprocessConfig &config = {}) {
  return preprocess(frame, config);
}

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_PREPROCESS_H
