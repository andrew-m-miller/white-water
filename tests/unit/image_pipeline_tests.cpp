// Dependency-free checks for Phase 2.3's host-free image transforms.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>

#include "core/flow/Composite.h"
#include "core/flow/Preprocess.h"
#include "core/flow/StMap.h"
#include "core/warp/Resampler.h"
#include "core/warp/WarpMap.h"

namespace {

using namespace whitewater;

int failures = 0;

void require(bool condition, const std::string &message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

void requireNear(double actual, double expected, const std::string &message,
                 double epsilon = 1.0e-5) {
  require(std::fabs(actual - expected) <= epsilon, message);
}

OwnedFrame makeFrame(int x1, int y1, int width, int height, double par = 1.0) {
  OwnedFrame frame;
  frame.bounds = {x1, y1, x1 + width, y1 + height};
  frame.origin = Vec2(static_cast<double>(x1), static_cast<double>(y1));
  frame.rowStride = static_cast<std::ptrdiff_t>(width) * kImageChannels;
  frame.sourceFormat.alpha = CapturedAlphaAssociation::kUnpremultiplied;
  frame.pixelAspectRatio = par;
  frame.rgba.resize(static_cast<std::size_t>(width) * height * kImageChannels);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float *pixel = frame.rgba.data() +
                     (static_cast<std::size_t>(y) * width + x) * kImageChannels;
      pixel[0] = static_cast<float>(x + 10 * y);
      pixel[1] = static_cast<float>(100 + x + 10 * y);
      pixel[2] = static_cast<float>(200 + x + 10 * y);
      pixel[3] = ((x + y) & 1) == 0 ? 0.25f : 0.75f;
    }
  }
  return frame;
}

void fillFlow(FlowField *field, Vec2 displacement) {
  for (int y = 0; y < field->rows(); ++y) {
    for (int x = 0; x < field->columns(); ++x) setFlow(field, x, y, displacement);
  }
}

class DecodedStMapWarp final : public WarpMap {
 public:
  DecodedStMapWarp(const Image &stMap, CapturedPixelBounds destinationBounds,
                  CapturedPixelBounds sourceBounds)
      : stMap_(stMap), destinationBounds_(destinationBounds), sourceBounds_(sourceBounds) {}

  Vec2 mapToSource(Vec2 destination) const override {
    const int localX = std::max(
        0, std::min(stMap_.width() - 1,
                    static_cast<int>(std::floor(destination.x - destinationBounds_.x1 - 0.5))));
    const int localY = std::max(
        0, std::min(stMap_.height() - 1,
                    static_cast<int>(std::floor(destination.y - destinationBounds_.y1 - 0.5))));
    const float *encoded = stMap_.pixel(localX, localY);
    return Vec2(static_cast<double>(sourceBounds_.x1) +
                    static_cast<double>(encoded[0]) * sourceBounds_.width(),
                static_cast<double>(sourceBounds_.y1) +
                    static_cast<double>(encoded[1]) * sourceBounds_.height());
  }

 private:
  const Image &stMap_;
  CapturedPixelBounds destinationBounds_;
  CapturedPixelBounds sourceBounds_;
};

void testPremultiply() {
  OwnedFrame frame = makeFrame(-4, 7, 2, 1);
  frame.rgba[0] = 0.8f;
  frame.rgba[1] = 0.4f;
  frame.rgba[2] = 0.2f;
  frame.rgba[3] = 0.25f;

  const Image straight = frameImage(frame, false);
  const Image premultiplied = frameImage(frame, true);
  require(straight.pixel(0, 0)[0] == 0.8f, "straight input keeps RGB");
  require(straight.pixel(0, 0)[1] == 0.4f, "straight input keeps green");
  requireNear(premultiplied.pixel(0, 0)[0], 0.2, "premultiplied red uses matte");
  requireNear(premultiplied.pixel(0, 0)[1], 0.1, "premultiplied green uses matte");
  requireNear(premultiplied.pixel(0, 0)[2], 0.05, "premultiplied blue uses matte");
  require(premultiplied.pixel(0, 0)[3] == 0.25f, "premultiply preserves alpha");
}

void testPreprocessGeometry() {
  OwnedFrame frame = makeFrame(-9, -3, 3, 5, 2.0);
  PreprocessConfig config;
  config.padMultiple = 4;
  const PreparedImage prepared = preprocess(frame, config);

  require(prepared.geometry.analysisWidth == 6, "PAR 2 doubles analysis width");
  require(prepared.geometry.analysisHeight == 5, "odd analysis height is preserved");
  require(prepared.geometry.crop.padLeft == 0, "default reflect pad left is zero");
  require(prepared.geometry.crop.padBottom == 0, "default reflect pad bottom is zero");
  require(prepared.geometry.crop.padRight == 2, "right pad reaches the requested multiple");
  require(prepared.geometry.crop.padTop == 3, "top pad reaches the requested multiple");
  require(prepared.image.width() == 8 && prepared.image.height() == 8,
          "prepared image has padded dimensions");

  const FieldGeometry fieldGeometry = prepared.geometry.fieldGeometry();
  requireNear(fieldGeometry.origin.x, -8.75,
              "PAR-normalized field origin maps the first analysis centre to source pixels");
  requireNear(fieldGeometry.origin.y, -2.5,
              "field origin retains the nonzero source Y bound");
  requireNear(fieldGeometry.spacingX, 0.5,
              "PAR 2 analysis vectors convert back with half-pixel X spacing");
  requireNear(fieldGeometry.spacingY, 1.0,
              "unchanged Y analysis converts back with unit spacing");

  const Image cropped = prepared.crop();
  require(cropped.width() == 6 && cropped.height() == 5, "crop metadata restores analysis size");
  requireNear(cropped.pixel(0, 0)[0], frame.pixel(0, 0)[0],
              "pixel-centre resize preserves the lower-left source pixel");
  requireNear(prepared.image.pixel(6, 0)[0], cropped.pixel(4, 0)[0],
              "right reflect pad samples the inner neighbour");
  requireNear(prepared.image.pixel(7, 0)[0], cropped.pixel(3, 0)[0],
              "second right reflect sample continues inward");

  for (const double par : {0.5, 2.0}) {
    OwnedFrame capped = makeFrame(0, 0, 101, 77, par);
    PreprocessConfig capConfig;
    capConfig.megapixelCap = 0.01;
    const AnalysisGeometry geometry = analysisGeometry(capped, capConfig);
    require(geometry.analysisWidth > 0 && geometry.analysisHeight > 0,
            "cap leaves positive analysis dimensions");
    require(static_cast<double>(geometry.analysisWidth) * geometry.analysisHeight <= 10000.0,
            "megapixel cap bounds rounded analysis area");
    require(geometry.scaleX > 0.0 && geometry.scaleY > 0.0,
            "PAR cap retains independent positive scales");
  }
}

void testExplicitReflectCropRoundTrip() {
  Image source(3, 2);
  for (int y = 0; y < source.height(); ++y) {
    for (int x = 0; x < source.width(); ++x) {
      float *pixel = source.pixel(x, y);
      pixel[0] = static_cast<float>(x + 10 * y) + 0.125f;
      pixel[1] = static_cast<float>(x + 20 * y) + 0.25f;
      pixel[2] = static_cast<float>(x + 30 * y) + 0.5f;
      pixel[3] = static_cast<float>((x + 1) * (y + 2)) / 10.0f;
    }
  }

  CropMetadata crop;
  crop.x = crop.padLeft = 2;
  crop.y = crop.padBottom = 1;
  crop.width = source.width();
  crop.height = source.height();
  crop.padRight = 4;
  crop.padTop = 3;
  crop.paddedWidth = crop.padLeft + crop.width + crop.padRight;
  crop.paddedHeight = crop.padBottom + crop.height + crop.padTop;

  const Image padded = reflectPad(source, crop);
  const Image restored = cropImage(padded, crop);
  require(restored.width() == source.width() && restored.height() == source.height(),
          "explicit asymmetric reflect crop restores source dimensions");
  bool identical = restored.width() == source.width() && restored.height() == source.height();
  if (identical) {
    for (int y = 0; y < source.height(); ++y) {
      identical = std::memcmp(source.view().row(y), restored.view().row(y),
                              static_cast<std::size_t>(source.width()) * kImageChannels *
                                  sizeof(float)) == 0;
      if (!identical) break;
    }
  }
  require(identical, "reflectPad followed by cropImage is bit-exact with four-sided padding");
}

void testStMaps() {
  const CapturedPixelBounds bounds = {-3, 5, 5, 12};
  FlowField field(8, 7, FieldGeometry::forPixels(bounds.x1, bounds.y1));
  fillFlow(&field, Vec2(2.0, -1.5));

  for (const StMapOrigin origin : {StMapOrigin::kBottomLeft, StMapOrigin::kTopLeft}) {
    StMapOptions options;
    options.sourceBounds = bounds;
    options.destinationBounds = bounds;
    options.origin = origin;
    options.mode = StMapMode::kAbsoluteUV;
    const Image absolute = fieldToStMap(field, options);
    const double bottomV = -1.0 / bounds.height();
    const double expectedV = origin == StMapOrigin::kTopLeft ? 1.0 - bottomV : bottomV;
    requireNear(absolute.pixel(0, 0)[0], 2.5 / bounds.width(),
                "absolute ST U includes translation and half-pixel origin");
    requireNear(absolute.pixel(0, 0)[1], expectedV,
                "absolute ST V respects origin convention");
    require(absolute.pixel(0, 0)[2] == 0.0f, "ST B is zero");
    require(absolute.pixel(0, 0)[3] == 1.0f, "ST alpha is one");

    options.mode = StMapMode::kRelativePixels;
    const Image relative = fieldToStMap(field, options);
    requireNear(relative.pixel(0, 0)[0], 2.0, "relative ST keeps real-pixel X");
    requireNear(relative.pixel(0, 0)[1], origin == StMapOrigin::kTopLeft ? 1.5 : -1.5,
                "relative ST flips Y only for top-left origin");
  }

  FlowField identity(8, 7, FieldGeometry::forPixels(bounds.x1, bounds.y1));
  fillFlow(&identity, Vec2());
  StMapOptions identityOptions;
  identityOptions.sourceBounds = bounds;
  identityOptions.destinationBounds = bounds;
  const Image identityMap = fieldToStMap(identity, identityOptions);
  requireNear(identityMap.pixel(0, 0)[0], 0.5 / bounds.width(),
              "identity ST U uses source image dimensions");
  requireNear(identityMap.pixel(0, 0)[1], 0.5 / bounds.height(),
              "identity ST V uses source image dimensions");
}

void testAbsoluteStRoundTripThroughResampler() {
  // Width and height are intentionally different, and the source has a nonzero/negative
  // image origin.  Both dimensions are powers of two so the float ST encoding/decoding is
  // itself exact; this isolates the round-trip assertion to the map and resampler paths.
  const CapturedPixelBounds bounds = {-11, 7, -3, 11};
  FlowField identity(bounds.width(), bounds.height(),
                    FieldGeometry::forPixels(bounds.x1, bounds.y1));
  fillFlow(&identity, Vec2());

  StMapOptions options;
  options.sourceBounds = bounds;
  options.destinationBounds = bounds;
  options.mode = StMapMode::kAbsoluteUV;
  options.origin = StMapOrigin::kBottomLeft;
  const Image stMap = fieldToStMap(identity, options);

  Image source(bounds.width(), bounds.height());
  for (int y = 0; y < source.height(); ++y) {
    for (int x = 0; x < source.width(); ++x) {
      float *pixel = source.pixel(x, y);
      pixel[0] = static_cast<float>(0.07 * x + 0.013 * y + 0.11);
      pixel[1] = static_cast<float>(0.019 * x + 0.11 * y + 0.23);
      pixel[2] = static_cast<float>((x * 7 + y * 5) % 13) / 13.0f;
      pixel[3] = ((x + 2 * y) % 5 == 0) ? 0.0f : static_cast<float>(0.2 + 0.15 * ((x + y) % 4));
    }
  }

  Image restored(source.width(), source.height());
  DecodedStMapWarp warp(stMap, bounds, bounds);
  ResampleOptions sampleOptions(AlphaMode::kUnpremultiplied);
  sampleOptions.filter = ResampleFilter::kBilinear;
  sampleOptions.edge = EdgeMode::kClamp;
  ResampleGeometry geometry;
  geometry.destinationOrigin = Vec2(bounds.x1, bounds.y1);
  geometry.sourceOrigin = Vec2(bounds.x1, bounds.y1);
  resample(source.view(), warp, sampleOptions, geometry, restored.view(), 1);

  bool identical = true;
  for (int y = 0; y < source.height(); ++y) {
    identical = std::memcmp(source.view().row(y), restored.view().row(y),
                            static_cast<std::size_t>(source.width()) * kImageChannels *
                                sizeof(float)) == 0;
    if (!identical) break;
  }
  require(identical, "absolute identity ST decoded through Resampler is bit-exact");
}

void testComposite() {
  Image foreground(2, 1);
  Image background(2, 1);
  foreground.pixel(0, 0)[0] = 1.0f;
  foreground.pixel(0, 0)[3] = 0.0f;
  foreground.pixel(1, 0)[0] = 1.0f;
  foreground.pixel(1, 0)[3] = 0.5f;
  background.pixel(0, 0)[1] = 0.25f;
  background.pixel(0, 0)[3] = 1.0f;
  background.pixel(1, 0)[1] = 0.25f;
  background.pixel(1, 0)[3] = 1.0f;

  const Image straight = over(foreground, background, CompositeAlphaMode::kStraight);
  requireNear(straight.pixel(0, 0)[0], 0.0, "transparent straight foreground contributes no RGB");
  requireNear(straight.pixel(0, 0)[1], 0.25, "straight over retains opaque background");
  requireNear(straight.pixel(0, 0)[3], 1.0, "opaque background keeps alpha one");
  requireNear(straight.pixel(1, 0)[0], 0.5, "straight over premultiplies foreground");
  requireNear(straight.pixel(1, 0)[1], 0.125, "straight over composites background coverage");

  foreground.pixel(0, 0)[0] = 0.0f;
  foreground.pixel(1, 0)[0] = 0.5f;  // premultiplied RGB for alpha 0.5
  const Image premultiplied = over(foreground, background, CompositeAlphaMode::kPremultiplied);
  requireNear(premultiplied.pixel(0, 0)[0], 0.0, "premultiplied transparent edge is zero");
  requireNear(premultiplied.pixel(0, 0)[1], 0.25, "premultiplied background is unchanged");
  requireNear(premultiplied.pixel(1, 0)[0], 0.5, "premultiplied RGB is not divided by alpha");
  requireNear(premultiplied.pixel(1, 0)[1], 0.125, "premultiplied background is covered once");
}

}  // namespace

int main() {
  testPremultiply();
  testPreprocessGeometry();
  testExplicitReflectCropRoundTrip();
  testStMaps();
  testAbsoluteStRoundTripThroughResampler();
  testComposite();
  if (failures == 0) std::cout << "Image pipeline tests passed\n";
  return failures == 0 ? 0 : 1;
}
