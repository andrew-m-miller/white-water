// Explicit straight/premultiplied alpha compositing for host-free image paths.

#ifndef WHITEWATER_CORE_FLOW_COMPOSITE_H
#define WHITEWATER_CORE_FLOW_COMPOSITE_H

#include "core/image/Image.h"

namespace whitewater {

enum class CompositeAlphaMode {
  kStraight,
  kPremultiplied,
};

// Source/foreground is placed over background.  All images are packed RGBA float views
// with equal dimensions.  In straight mode RGB is independent of alpha at the inputs and
// output; in premultiplied mode RGB already carries coverage.  The output is written in the
// same association as the selected mode.
void over(const ConstImageView &foreground, const ConstImageView &background,
          const ImageView &destination, CompositeAlphaMode mode);

Image over(const Image &foreground, const Image &background, CompositeAlphaMode mode);

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_COMPOSITE_H
