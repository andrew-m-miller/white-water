// Regression checks for integer megapixel-cap rounding at extreme aspect ratios.

#include <cstddef>
#include <iostream>
#include <string>

#include "core/flow/Preprocess.h"

namespace {

using namespace whitewater;

int failures = 0;

void require(bool condition, const std::string &message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

OwnedFrame frame(int width, int height) {
  OwnedFrame result;
  result.bounds = {0, 0, width, height};
  result.origin = Vec2();
  result.rowStride = static_cast<std::ptrdiff_t>(width) * kImageChannels;
  result.pixelAspectRatio = 1.0;
  result.rgba.assign(static_cast<std::size_t>(width) * height * kImageChannels, 0.0f);
  return result;
}

void testWideExtremeAspect() {
  // With the old correction, the initial square-root reduction rounds height to 1,
  // selects that already-minimal axis, and leaves width*height above the cap.
  const OwnedFrame source = frame(10001, 1);
  PreprocessConfig config;
  config.megapixelCap = 0.001;  // 1,000 pixels
  const AnalysisGeometry geometry = analysisGeometry(source, config);
  require(geometry.analysisWidth * geometry.analysisHeight <= 1000,
          "wide extreme aspect never exceeds the megapixel cap");
  require(geometry.analysisWidth == 1000 && geometry.analysisHeight == 1,
          "wide extreme aspect retains the best legal aspect under the cap");
}

void testTallExtremeAspect() {
  const OwnedFrame source = frame(1, 10001);
  PreprocessConfig config;
  config.megapixelCap = 0.001;
  const AnalysisGeometry geometry = analysisGeometry(source, config);
  require(geometry.analysisWidth * geometry.analysisHeight <= 1000,
          "tall extreme aspect never exceeds the megapixel cap");
  require(geometry.analysisWidth == 1 && geometry.analysisHeight == 1000,
          "tall extreme aspect retains the best legal aspect under the cap");
}

}  // namespace

int main() {
  testWideExtremeAspect();
  testTallExtremeAspect();
  if (failures == 0) std::cout << "Preprocess cap tests passed\n";
  return failures == 0 ? 0 : 1;
}
