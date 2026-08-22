// Conversion of a backward displacement field to Flame's float ST-map conventions.

#ifndef WHITEWATER_CORE_FLOW_STMAP_H
#define WHITEWATER_CORE_FLOW_STMAP_H

#include "core/flow/Field.h"
#include "core/image/Image.h"
#include "core/image/OwnedFrame.h"

namespace whitewater {

enum class StMapMode {
  kAbsoluteUV,
  kAbsolute = kAbsoluteUV,
  kRelativePixels,
  kRelative = kRelativePixels,
};

enum class StMapOrigin {
  kBottomLeft,
  kTopLeft,
};

struct StMapOptions {
  StMapMode mode = StMapMode::kAbsoluteUV;
  StMapOrigin origin = StMapOrigin::kBottomLeft;

  // The output destination and the source image can have different bounds.  Empty bounds
  // are filled from the other one; if both are empty, the field's node extent is used with
  // a lower-left origin inferred from its first node.
  CapturedPixelBounds sourceBounds;
  CapturedPixelBounds destinationBounds;
};

// Writes U/V in R/G, leaves B at zero, and writes alpha one.  The destination image is
// packed and has the dimensions of options.destinationBounds.  The field values are
// backward real-pixel displacements sampled at destination pixel centres.
Image fieldToStMap(const FlowField &field, const StMapOptions &options = {});

// View form for a caller that owns its output storage.  The view dimensions must match the
// resolved destination bounds in options; a mismatch raises std::invalid_argument.
void fieldToStMap(const FlowField &field, const StMapOptions &options,
                  const ImageView &destination);

// Convenience form for an output and source image extent beginning at (0, 0).
Image fieldToStMap(const FlowField &field, int width, int height,
                  StMapMode mode = StMapMode::kAbsoluteUV,
                  StMapOrigin origin = StMapOrigin::kBottomLeft);

// Convenience form that preserves non-zero/negative destination and source bounds.
Image fieldToStMap(const FlowField &field, const CapturedPixelBounds &sourceBounds,
                  const CapturedPixelBounds &destinationBounds, StMapMode mode,
                  StMapOrigin origin = StMapOrigin::kBottomLeft);

// Descriptive aliases used by the flow and CLI layers.
inline Image flowToStMap(const FlowField &field, const StMapOptions &options = {}) {
  return fieldToStMap(field, options);
}

inline Image makeStMap(const FlowField &field, const StMapOptions &options = {}) {
  return fieldToStMap(field, options);
}

inline Image convertFlowToStMap(const FlowField &field, const StMapOptions &options = {}) {
  return fieldToStMap(field, options);
}

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_STMAP_H
