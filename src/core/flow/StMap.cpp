#include "core/flow/StMap.h"

#include <cmath>
#include <string>
#include <stdexcept>

namespace whitewater {
namespace {

CapturedPixelBounds resolveBounds(const FlowField &field, CapturedPixelBounds source,
                                  CapturedPixelBounds destination, bool sourceRequested,
                                  bool destinationRequested) {
  if (!sourceRequested && destinationRequested) source = destination;
  if (!destinationRequested && sourceRequested) destination = source;
  if (!sourceRequested && !destinationRequested) {
    const Vec2 origin = field.geometry().origin;
    const int x = static_cast<int>(std::floor(origin.x - 0.5));
    const int y = static_cast<int>(std::floor(origin.y - 0.5));
    destination = {x, y, x + field.columns(), y + field.rows()};
    source = destination;
  }
  return destination;
}

void validateBounds(const CapturedPixelBounds &bounds, const char *name) {
  if (bounds.width() <= 0 || bounds.height() <= 0) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
}

}  // namespace

Image fieldToStMap(const FlowField &field, const StMapOptions &options) {
  const bool sourceRequested = !options.sourceBounds.isEmpty();
  const bool destinationRequested = !options.destinationBounds.isEmpty();
  CapturedPixelBounds destination =
      resolveBounds(field, options.sourceBounds, options.destinationBounds, sourceRequested,
                    destinationRequested);
  CapturedPixelBounds source = options.sourceBounds;
  if (!sourceRequested) source = destination;
  validateBounds(source, "source bounds");
  validateBounds(destination, "destination bounds");

  Image output(destination.width(), destination.height());
  fieldToStMap(field, options, output.view());
  return output;
}

void fieldToStMap(const FlowField &field, const StMapOptions &options,
                  const ImageView &destination) {
  const bool sourceRequested = !options.sourceBounds.isEmpty();
  const bool destinationRequested = !options.destinationBounds.isEmpty();
  CapturedPixelBounds resolvedDestination =
      resolveBounds(field, options.sourceBounds, options.destinationBounds, sourceRequested,
                    destinationRequested);
  CapturedPixelBounds source = options.sourceBounds;
  if (!sourceRequested) source = resolvedDestination;
  validateBounds(source, "source bounds");
  validateBounds(resolvedDestination, "destination bounds");
  if (destination.isEmpty() || destination.width() != resolvedDestination.width() ||
      destination.height() != resolvedDestination.height()) {
    throw std::invalid_argument("ST destination dimensions do not match destination bounds");
  }

  const double sourceWidth = static_cast<double>(source.width());
  const double sourceHeight = static_cast<double>(source.height());
  for (int y = 0; y < destination.height(); ++y) {
    const double destinationY = static_cast<double>(resolvedDestination.y1 + y) + 0.5;
    float *out = destination.row(y);
    for (int x = 0; x < destination.width(); ++x) {
      const double destinationX = static_cast<double>(resolvedDestination.x1 + x) + 0.5;
      const Vec2 displacement = sampleFlow(field, Vec2(destinationX, destinationY));
      double u = 0.0;
      double v = 0.0;
      if (options.mode == StMapMode::kRelativePixels) {
        u = displacement.x;
        v = options.origin == StMapOrigin::kTopLeft ? -displacement.y : displacement.y;
      } else {
        const Vec2 sourcePoint = Vec2(destinationX + displacement.x,
                                      destinationY + displacement.y);
        u = (sourcePoint.x - static_cast<double>(source.x1)) / sourceWidth;
        const double bottomLeftV =
            (sourcePoint.y - static_cast<double>(source.y1)) / sourceHeight;
        v = options.origin == StMapOrigin::kTopLeft ? 1.0 - bottomLeftV : bottomLeftV;
      }

      out[0] = static_cast<float>(u);
      out[1] = static_cast<float>(v);
      out[2] = 0.0f;
      out[3] = 1.0f;
      out += kImageChannels;
    }
  }
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
