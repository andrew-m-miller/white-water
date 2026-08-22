// Explicit straight/premultiplied alpha compositing for host-free image paths.

#ifndef WHITEWATER_CORE_FLOW_COMPOSITE_H
#define WHITEWATER_CORE_FLOW_COMPOSITE_H

#include "core/image/Image.h"

namespace whitewater {

enum class CompositeAlphaMode {
  kStraight,
  kUnpremultiplied = kStraight,
  kPremultiplied,
};

using CompositeMode = CompositeAlphaMode;

// Source/foreground is placed over background.  All images are packed RGBA float views
// with equal dimensions.  In straight mode RGB is independent of alpha at the inputs and
// output; in premultiplied mode RGB already carries coverage.  The output is written in the
// same association as the selected mode.
void over(const ConstImageView &foreground, const ConstImageView &background,
          const ImageView &destination, CompositeAlphaMode mode);

Image over(const Image &foreground, const Image &background, CompositeAlphaMode mode);

inline void compositeOver(const ConstImageView &foreground, const ConstImageView &background,
                          const ImageView &destination, CompositeAlphaMode mode) {
  over(foreground, background, destination, mode);
}

inline Image compositeOver(const Image &foreground, const Image &background,
                           CompositeAlphaMode mode) {
  return over(foreground, background, mode);
}

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_COMPOSITE_H
