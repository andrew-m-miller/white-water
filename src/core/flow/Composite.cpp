#include "core/flow/Composite.h"

#include <algorithm>
#include <stdexcept>

namespace whitewater {

void over(const ConstImageView &foreground, const ConstImageView &background,
          const ImageView &destination, CompositeAlphaMode mode) {
  if (foreground.isEmpty() || background.isEmpty() || destination.isEmpty()) return;
  if (foreground.width() != background.width() || foreground.height() != background.height() ||
      destination.width() != foreground.width() || destination.height() != foreground.height()) {
    throw std::invalid_argument("over requires equally sized RGBA images");
  }

  for (int y = 0; y < destination.height(); ++y) {
    const float *front = foreground.row(y);
    const float *back = background.row(y);
    float *out = destination.row(y);
    for (int x = 0; x < destination.width(); ++x) {
      const double frontAlpha = static_cast<double>(front[3]);
      const double backAlpha = static_cast<double>(back[3]);
      const double inverseFront = 1.0 - frontAlpha;
      const double outputAlpha = frontAlpha + backAlpha * inverseFront;

      if (mode == CompositeAlphaMode::kPremultiplied) {
        out[0] = static_cast<float>(static_cast<double>(front[0]) +
                                    static_cast<double>(back[0]) * inverseFront);
        out[1] = static_cast<float>(static_cast<double>(front[1]) +
                                    static_cast<double>(back[1]) * inverseFront);
        out[2] = static_cast<float>(static_cast<double>(front[2]) +
                                    static_cast<double>(back[2]) * inverseFront);
      } else if (outputAlpha > 0.0) {
        const double inverseOutput = 1.0 / outputAlpha;
        out[0] = static_cast<float>((static_cast<double>(front[0]) * frontAlpha +
                                     static_cast<double>(back[0]) * backAlpha * inverseFront) *
                                    inverseOutput);
        out[1] = static_cast<float>((static_cast<double>(front[1]) * frontAlpha +
                                     static_cast<double>(back[1]) * backAlpha * inverseFront) *
                                    inverseOutput);
        out[2] = static_cast<float>((static_cast<double>(front[2]) * frontAlpha +
                                     static_cast<double>(back[2]) * backAlpha * inverseFront) *
                                    inverseOutput);
      } else {
        // A fully transparent straight result has no recoverable colour.  Clearing it also
        // avoids propagating hidden RGB from a transparent matte into a later operation.
        out[0] = 0.0f;
        out[1] = 0.0f;
        out[2] = 0.0f;
      }
      out[3] = static_cast<float>(outputAlpha);

      front += kImageChannels;
      back += kImageChannels;
      out += kImageChannels;
    }
  }
}

Image over(const Image &foreground, const Image &background, CompositeAlphaMode mode) {
  if (foreground.isEmpty() || background.isEmpty()) return Image();
  if (foreground.width() != background.width() || foreground.height() != background.height()) {
    throw std::invalid_argument("over requires equally sized RGBA images");
  }
  Image output(foreground.width(), foreground.height());
  over(foreground.view(), background.view(), output.view(), mode);
  return output;
}

}  // namespace whitewater
