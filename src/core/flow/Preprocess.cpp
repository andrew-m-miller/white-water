#include "core/flow/Preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

#include "core/warp/Resampler.h"

namespace whitewater {
namespace {

bool validPositiveFinite(double value) {
  return value > 0.0 && std::isfinite(value);
}

int roundedDimension(double value) {
  if (!(value > 0.0) || !std::isfinite(value)) return 0;
  const double rounded = std::floor(value + 0.5);
  if (rounded < 1.0) return 1;
  if (rounded > static_cast<double>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("analysis dimension is too large");
  }
  return static_cast<int>(rounded);
}

long long ceilDivide(long long numerator, int denominator) {
  return (numerator + denominator - 1) / denominator;
}

int mirrorIndex(int index, int size) {
  if (size <= 1) return 0;
  if (index >= 0 && index < size) return index;
  const int period = 2 * (size - 1);
  int folded = index % period;
  if (folded < 0) folded += period;
  return folded < size ? folded : period - folded;
}

void validateCrop(const CropMetadata &crop) {
  if (crop.width <= 0 || crop.height <= 0 || crop.x < 0 || crop.y < 0 ||
      crop.paddedWidth <= 0 || crop.paddedHeight <= 0 || crop.padLeft < 0 ||
      crop.padRight < 0 || crop.padBottom < 0 || crop.padTop < 0 ||
      crop.x != crop.padLeft || crop.y != crop.padBottom ||
      crop.x + crop.width + crop.padRight != crop.paddedWidth ||
      crop.y + crop.height + crop.padTop != crop.paddedHeight) {
    throw std::invalid_argument("inconsistent crop metadata");
  }
}

Image resizeBilinear(const Image &source, const AnalysisGeometry &geometry,
                     bool sourcePremultiplied) {
  if (source.isEmpty() || geometry.analysisWidth <= 0 || geometry.analysisHeight <= 0) {
    return Image();
  }

  Image output(geometry.analysisWidth, geometry.analysisHeight);
  const AlphaMode alpha = sourcePremultiplied ? AlphaMode::kPremultiplied
                                               : AlphaMode::kUnpremultiplied;
  ResampleOptions options(alpha);
  options.filter = ResampleFilter::kBilinear;
  options.edge = EdgeMode::kClamp;

  // Coordinates are pixel-centre coordinates in the source's local image space.  The
  // canonical square-pixel image spans [0, sourceWidth*PAR] horizontally and
  // [0, sourceHeight] vertically, so this one expression handles PAR expansion,
  // megapixel reduction, odd rounded extents, and the half-pixel convention together.
  const ConstImageView sourceView = source.view();
  const ImageView outputView = output.view();
  for (int y = 0; y < output.height(); ++y) {
    float *destination = outputView.row(y);
    const double sourceY = (static_cast<double>(y) + 0.5) / geometry.scaleY;
    for (int x = 0; x < output.width(); ++x) {
      const double sourceX = (static_cast<double>(x) + 0.5) /
                             (geometry.scaleX * geometry.pixelAspectRatio);
      sampleImage(sourceView, Vec2(sourceX, sourceY), options, destination);
      destination += kImageChannels;
    }
  }
  return output;
}

}  // namespace

FieldGeometry AnalysisGeometry::fieldGeometry() const {
  if (analysisWidth <= 0 || analysisHeight <= 0 ||
      !validPositiveFinite(pixelAspectRatio) || !validPositiveFinite(scaleX) ||
      !validPositiveFinite(scaleY)) {
    throw std::invalid_argument("empty or invalid analysis geometry has no field lattice");
  }
  const double spacingX = 1.0 / (scaleX * pixelAspectRatio);
  const double spacingY = 1.0 / scaleY;
  return FieldGeometry(
      Vec2(static_cast<double>(sourceBounds.x1) + 0.5 * spacingX,
           static_cast<double>(sourceBounds.y1) + 0.5 * spacingY),
      spacingX, spacingY);
}

ConstImageView PreparedImage::analysisView() const {
  if (image.isEmpty() || geometry.crop.isEmpty()) return ConstImageView();
  return ConstImageView(image.pixel(geometry.crop.x, geometry.crop.y),
                        geometry.crop.width, geometry.crop.height, image.rowStride());
}

Image PreparedImage::crop() const { return cropImage(image, geometry.crop); }

AnalysisGeometry analysisGeometry(const OwnedFrame &frame, const PreprocessConfig &config) {
  AnalysisGeometry geometry;
  geometry.sourceBounds = frame.bounds;
  geometry.sourceWidth = frame.width();
  geometry.sourceHeight = frame.height();

  if (geometry.sourceWidth <= 0 || geometry.sourceHeight <= 0 || frame.rgba.empty()) {
    return geometry;
  }
  if (!validPositiveFinite(frame.pixelAspectRatio)) {
    throw std::invalid_argument("pixel aspect ratio must be finite and positive");
  }
  if (!std::isfinite(config.megapixelCap) || config.megapixelCap < 0.0) {
    throw std::invalid_argument("megapixel cap must be finite and non-negative");
  }
  if (config.padMultiple <= 0 || config.padLeft < 0 || config.padRight < 0 ||
      config.padBottom < 0 || config.padTop < 0) {
    throw std::invalid_argument("padding multiple and sides must be non-negative");
  }

  geometry.pixelAspectRatio = frame.pixelAspectRatio;
  geometry.canonicalWidth = static_cast<double>(geometry.sourceWidth) *
                            geometry.pixelAspectRatio;
  geometry.canonicalHeight = static_cast<double>(geometry.sourceHeight);
  const double canonicalArea = geometry.canonicalWidth * geometry.canonicalHeight;
  if (!validPositiveFinite(geometry.canonicalWidth) ||
      !validPositiveFinite(canonicalArea)) {
    throw std::invalid_argument("canonical analysis extent is not finite");
  }

  const double capPixelsValue = config.megapixelCap * 1000000.0;
  if (config.megapixelCap > 0.0 && !validPositiveFinite(capPixelsValue)) {
    throw std::invalid_argument("megapixel cap is outside the supported range");
  }
  const bool capApplied = config.megapixelCap > 0.0 && canonicalArea > capPixelsValue;
  if (capApplied &&
      capPixelsValue > static_cast<double>(std::numeric_limits<long long>::max())) {
    throw std::invalid_argument("megapixel cap exceeds integer pixel accounting");
  }

  double scale = 1.0;
  if (capApplied) {
    scale = std::sqrt(capPixelsValue / canonicalArea);
    // A cap is a reduction limit, never an instruction to enlarge a small image.
    scale = std::min(scale, 1.0);
  }
  geometry.analysisWidth = roundedDimension(geometry.canonicalWidth * scale);
  geometry.analysisHeight = roundedDimension(geometry.canonicalHeight * scale);
  if (capApplied) {
    const long long capPixels = std::max(
        1LL, static_cast<long long>(std::floor(capPixelsValue)));
    long long area = static_cast<long long>(geometry.analysisWidth) * geometry.analysisHeight;
    if (area > capPixels) {
      // Rounding each axis independently can put the product a few pixels above the cap.
      // Reduce the axis whose rounded scale is relatively larger, then make one final
      // quotient adjustment.  This is constant time even when a very large source is
      // reduced to a small cap.
      const double roundedScaleX = geometry.analysisWidth / geometry.canonicalWidth;
      const double roundedScaleY = geometry.analysisHeight / geometry.canonicalHeight;
      if (roundedScaleX >= roundedScaleY) {
        geometry.analysisWidth = std::max(
            1, std::min(geometry.analysisWidth,
                        static_cast<int>(capPixels / geometry.analysisHeight)));
      } else {
        geometry.analysisHeight = std::max(
            1, std::min(geometry.analysisHeight,
                        static_cast<int>(capPixels / geometry.analysisWidth)));
      }
      area = static_cast<long long>(geometry.analysisWidth) * geometry.analysisHeight;
      if (area > capPixels) {
        geometry.analysisHeight = std::max(
            1, std::min(geometry.analysisHeight,
                        static_cast<int>(capPixels / geometry.analysisWidth)));
      }
    }
  }
  geometry.scaleX = static_cast<double>(geometry.analysisWidth) / geometry.canonicalWidth;
  geometry.scaleY = static_cast<double>(geometry.analysisHeight) / geometry.canonicalHeight;

  const long long minimumPaddedWidth = static_cast<long long>(geometry.analysisWidth) +
                                       config.padLeft + config.padRight;
  const long long minimumPaddedHeight = static_cast<long long>(geometry.analysisHeight) +
                                        config.padBottom + config.padTop;
  const long long paddedWidth = ceilDivide(minimumPaddedWidth, config.padMultiple) *
                                config.padMultiple;
  const long long paddedHeight = ceilDivide(minimumPaddedHeight, config.padMultiple) *
                                 config.padMultiple;
  if (paddedWidth <= 0 || paddedHeight <= 0 ||
      paddedWidth > std::numeric_limits<int>::max() ||
      paddedHeight > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("padded analysis dimensions are too large");
  }

  geometry.paddedWidth = static_cast<int>(paddedWidth);
  geometry.paddedHeight = static_cast<int>(paddedHeight);
  geometry.crop.x = config.padLeft;
  geometry.crop.y = config.padBottom;
  geometry.crop.width = geometry.analysisWidth;
  geometry.crop.height = geometry.analysisHeight;
  geometry.crop.paddedWidth = geometry.paddedWidth;
  geometry.crop.paddedHeight = geometry.paddedHeight;
  geometry.crop.padLeft = config.padLeft;
  geometry.crop.padBottom = config.padBottom;
  geometry.crop.padRight = geometry.paddedWidth - geometry.analysisWidth - config.padLeft;
  geometry.crop.padTop = geometry.paddedHeight - geometry.analysisHeight - config.padBottom;
  return geometry;
}

Image frameImage(const OwnedFrame &frame, bool premultiplyByMatte) {
  if (frame.width() <= 0 || frame.height() <= 0 || frame.rgba.empty()) return Image();
  if (frame.rowStride < static_cast<std::ptrdiff_t>(frame.width()) * kImageChannels) {
    throw std::invalid_argument("owned frame row stride is smaller than its width");
  }
  const std::size_t required = static_cast<std::size_t>(frame.height() - 1) *
                                   static_cast<std::size_t>(frame.rowStride) +
                               static_cast<std::size_t>(frame.width()) * kImageChannels;
  if (required > frame.rgba.size()) {
    throw std::invalid_argument("owned frame pixel storage is truncated");
  }

  Image image(frame.width(), frame.height());
  const bool alreadyPremultiplied =
      frame.sourceFormat.alpha == CapturedAlphaAssociation::kPremultiplied ||
      frame.sourceFormat.alpha == CapturedAlphaAssociation::kOpaque;
  for (int y = 0; y < frame.height(); ++y) {
    const float *source = frame.pixel(0, y);
    float *destination = image.view().row(y);
    for (int x = 0; x < frame.width(); ++x) {
      destination[0] = source[0];
      destination[1] = source[1];
      destination[2] = source[2];
      destination[3] = source[3];
      if (premultiplyByMatte && !alreadyPremultiplied) {
        destination[0] *= destination[3];
        destination[1] *= destination[3];
        destination[2] *= destination[3];
      }
      source += kImageChannels;
      destination += kImageChannels;
    }
  }
  return image;
}

Image reflectPad(const Image &source, const CropMetadata &crop) {
  if (source.isEmpty()) return Image();
  validateCrop(crop);
  if (source.width() != crop.width || source.height() != crop.height) {
    throw std::invalid_argument("crop metadata does not describe the source image");
  }

  Image padded(crop.paddedWidth, crop.paddedHeight);
  for (int y = 0; y < padded.height(); ++y) {
    const int sourceY = mirrorIndex(y - crop.padBottom, source.height());
    const float *sourceRow = source.view().row(sourceY);
    float *destination = padded.view().row(y);
    for (int x = 0; x < padded.width(); ++x) {
      const int sourceX = mirrorIndex(x - crop.padLeft, source.width());
      const float *pixel = sourceRow + static_cast<std::ptrdiff_t>(sourceX) * kImageChannels;
      for (int c = 0; c < kImageChannels; ++c) destination[c] = pixel[c];
      destination += kImageChannels;
    }
  }
  return padded;
}

Image cropImage(const Image &padded, const CropMetadata &crop) {
  if (padded.isEmpty()) return Image();
  try {
    validateCrop(crop);
  } catch (const std::invalid_argument &) {
    return Image();
  }
  if (padded.width() != crop.paddedWidth || padded.height() != crop.paddedHeight) {
    return Image();
  }

  Image output(crop.width, crop.height);
  for (int y = 0; y < crop.height; ++y) {
    const float *source = padded.view().row(y + crop.y) +
                          static_cast<std::ptrdiff_t>(crop.x) * kImageChannels;
    float *destination = output.view().row(y);
    std::copy(source, source + static_cast<std::ptrdiff_t>(crop.width) * kImageChannels,
              destination);
  }
  return output;
}

PreparedImage preprocess(const OwnedFrame &frame, const PreprocessConfig &config) {
  PreparedImage prepared;
  prepared.transformToken = config.transformToken;
  if (frame.width() <= 0 || frame.height() <= 0 || frame.rgba.empty()) return prepared;

  prepared.geometry = analysisGeometry(frame, config);
  const bool sourcePremultiplied =
      config.premultiplyByMatte ||
      frame.sourceFormat.alpha == CapturedAlphaAssociation::kPremultiplied ||
      frame.sourceFormat.alpha == CapturedAlphaAssociation::kOpaque;
  prepared.premultiplied = sourcePremultiplied;

  const Image source = frameImage(frame, config.premultiplyByMatte);
  const Image analysis = resizeBilinear(source, prepared.geometry, sourcePremultiplied);
  prepared.image = reflectPad(analysis, prepared.geometry.crop);
  return prepared;
}

}  // namespace whitewater
