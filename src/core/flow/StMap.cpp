#include "core/flow/StMap.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace whitewater {
namespace {

struct ResolvedBounds {
  CapturedPixelBounds source;
  CapturedPixelBounds destination;
};

void validateBounds(const CapturedPixelBounds &bounds, const char *name) {
  if (bounds.width() <= 0 || bounds.height() <= 0) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
}

ResolvedBounds resolveBounds(const FlowField &field, const StMapOptions &options) {
  ResolvedBounds resolved{options.sourceBounds, options.destinationBounds};
  const bool sourceRequested = !resolved.source.isEmpty();
  const bool destinationRequested = !resolved.destination.isEmpty();

  if (!sourceRequested && destinationRequested) resolved.source = resolved.destination;
  if (!destinationRequested && sourceRequested) resolved.destination = resolved.source;
  if (!sourceRequested && !destinationRequested) {
    const Vec2 origin = field.geometry().origin;
    const int x = static_cast<int>(std::floor(origin.x - 0.5));
    const int y = static_cast<int>(std::floor(origin.y - 0.5));
    resolved.destination = {x, y, x + field.columns(), y + field.rows()};
    resolved.source = resolved.destination;
  }

  validateBounds(resolved.source, "source bounds");
  validateBounds(resolved.destination, "destination bounds");
  return resolved;
}

void writeStMap(const FlowField &field, StMapMode mode, StMapOrigin origin,
                const ResolvedBounds &bounds, const ImageView &destination) {
  if (destination.isEmpty() || destination.width() != bounds.destination.width() ||
      destination.height() != bounds.destination.height()) {
    throw std::invalid_argument("ST destination dimensions do not match destination bounds");
  }

  const double sourceWidth = static_cast<double>(bounds.source.width());
  const double sourceHeight = static_cast<double>(bounds.source.height());
  for (int y = 0; y < destination.height(); ++y) {
    const double destinationY = static_cast<double>(bounds.destination.y1 + y) + 0.5;
    float *out = destination.row(y);
    for (int x = 0; x < destination.width(); ++x) {
      const double destinationX = static_cast<double>(bounds.destination.x1 + x) + 0.5;
      const Vec2 displacement = sampleFlow(field, Vec2(destinationX, destinationY));
      double u = 0.0;
      double v = 0.0;
      if (mode == StMapMode::kRelativePixels) {
        u = displacement.x;
        v = origin == StMapOrigin::kTopLeft ? -displacement.y : displacement.y;
      } else {
        const Vec2 sourcePoint = Vec2(destinationX + displacement.x,
                                      destinationY + displacement.y);
        u = (sourcePoint.x - static_cast<double>(bounds.source.x1)) / sourceWidth;
        const double bottomLeftV =
            (sourcePoint.y - static_cast<double>(bounds.source.y1)) / sourceHeight;
        v = origin == StMapOrigin::kTopLeft ? 1.0 - bottomLeftV : bottomLeftV;
      }

      out[0] = static_cast<float>(u);
      out[1] = static_cast<float>(v);
      out[2] = 0.0f;
      out[3] = 1.0f;
      out += kImageChannels;
    }
  }
}

}  // namespace

Image fieldToStMap(const FlowField &field, const StMapOptions &options) {
  const ResolvedBounds bounds = resolveBounds(field, options);
  Image output(bounds.destination.width(), bounds.destination.height());
  writeStMap(field, options.mode, options.origin, bounds, output.view());
  return output;
}

void fieldToStMap(const FlowField &field, const StMapOptions &options,
                  const ImageView &destination) {
  const ResolvedBounds bounds = resolveBounds(field, options);
  writeStMap(field, options.mode, options.origin, bounds, destination);
}

Image fieldToStMap(const FlowField &field, int width, int height, StMapMode mode,
                  StMapOrigin origin) {
  if (width <= 0 || height <= 0) return Image();
  StMapOptions options;
  options.mode = mode;
  options.origin = origin;
  options.sourceBounds = {0, 0, width, height};
  options.destinationBounds = options.sourceBounds;
  return fieldToStMap(field, options);
}

Image fieldToStMap(const FlowField &field, const CapturedPixelBounds &sourceBounds,
                  const CapturedPixelBounds &destinationBounds, StMapMode mode,
                  StMapOrigin origin) {
  StMapOptions options;
  options.mode = mode;
  options.origin = origin;
  options.sourceBounds = sourceBounds;
  options.destinationBounds = destinationBounds;
  return fieldToStMap(field, options);
}

}  // namespace whitewater
