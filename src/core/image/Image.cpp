// Vendored from warp-drive 887a123, src/core/image/Image.cpp.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "core/image/Image.h"

namespace whitewater {

Image::Image(int width, int height) {
  if (width <= 0 || height <= 0) {
    return;
  }
  width_ = width;
  height_ = height;
  pixels_.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                     static_cast<std::size_t>(kImageChannels),
                 0.0f);
}

void Image::fill(float r, float g, float b, float a) {
  for (std::size_t i = 0; i + kImageChannels <= pixels_.size(); i += kImageChannels) {
    pixels_[i] = r;
    pixels_[i + 1] = g;
    pixels_[i + 2] = b;
    pixels_[i + 3] = a;
  }
}

}  // namespace whitewater
