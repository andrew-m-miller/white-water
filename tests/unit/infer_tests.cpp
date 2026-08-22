#include "infer/NullPairwiseEstimator.h"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

bool near(double a, double b, double tolerance = 1.0e-6) {
  return std::fabs(a - b) <= tolerance;
}

whitewater::OwnedFrame frame(int width, int height) {
  whitewater::OwnedFrame result;
  result.bounds = {0, 0, width, height};
  result.origin = whitewater::Vec2();
  result.rowStride = static_cast<std::ptrdiff_t>(width) * whitewater::kImageChannels;
  result.rgba.assign(static_cast<std::size_t>(width) * height * whitewater::kImageChannels,
                     0.0f);
  result.pixelAspectRatio = 1.0;
  return result;
}

}  // namespace

int main() {
  using namespace whitewater;
  const OwnedFrame a = frame(7, 5);
  const OwnedFrame b = frame(7, 5);

  FlowRequest request;
  request.fromTime = 9;
  request.toTime = 8;
  request.columns = 4;
  request.rows = 3;
  request.fromGeometry = FieldGeometry(Vec2(-3.0, 4.0), 2.0, 3.0);
  request.toGeometry = FieldGeometry(Vec2(-2.0, 5.0), 1.5, 2.5);

  NullFlowParameters translation;
  translation.pattern = NullFlowPattern::kTranslation;
  translation.translation = Vec2(3.25, -1.5);
  translation.emitConfidence = true;
  NullPairwiseEstimator estimator(translation);
  FlowResult result = estimator.estimate(a, b, request);
  check(result.succeeded(), "translation estimate succeeds without weights or a GPU");
  check(result.link.fromTime() == 9 && result.link.toTime() == 8,
        "result preserves direction-labelled temporal endpoints");
  check(result.link.fromGeometry() == request.fromGeometry &&
            result.link.toGeometry() == request.toGeometry,
        "result preserves distinct endpoint geometries");
  for (int y = 0; y < request.rows; ++y) {
    for (int x = 0; x < request.columns; ++x) {
      const Vec2 value = flowNode(result.link.field(), x, y);
      check(near(value.x, 3.25) && near(value.y, -1.5),
            "translation is constant at every lattice node");
    }
  }
  check(result.confidence.has_value() &&
            near(sampleScalar(*result.confidence, request.fromGeometry.origin), 1.0),
        "optional deterministic confidence is separate from the link");

  NullFlowParameters affine;
  affine.pattern = NullFlowPattern::kAffine;
  affine.translation = Vec2(1.0, 2.0);
  affine.xx = 0.5;
  affine.xy = -0.25;
  affine.yx = 0.125;
  affine.yy = 0.75;
  FlowResult affineResult = NullPairwiseEstimator(affine).estimate(a, b, request);
  check(affineResult.succeeded(), "affine analytic estimate succeeds");
  const Vec2 position = affineResult.link.field().positionOf(2, 1);
  const Vec2 relative = position - request.fromGeometry.origin;
  const Vec2 affineValue = flowNode(affineResult.link.field(), 2, 1);
  check(near(affineValue.x, 1.0 + 0.5 * relative.x - 0.25 * relative.y) &&
            near(affineValue.y, 2.0 + 0.125 * relative.x + 0.75 * relative.y),
        "affine pattern is evaluated in full-resolution pixel coordinates");

  NullFlowParameters spatial;
  spatial.pattern = NullFlowPattern::kSpatial;
  spatial.translation = Vec2(0.25, -0.5);
  spatial.spatialAmplitude = Vec2(2.0, 3.0);
  spatial.spatialFrequency = Vec2(0.4, 0.2);
  FlowResult spatialResult = NullPairwiseEstimator(spatial).estimate(a, b, request);
  check(spatialResult.succeeded(), "spatially varying analytic estimate succeeds");
  const Vec2 spatialPosition = spatialResult.link.field().positionOf(2, 1);
  const Vec2 spatialRelative = spatialPosition - request.fromGeometry.origin;
  const Vec2 spatialValue = flowNode(spatialResult.link.field(), 2, 1);
  check(near(spatialValue.x, 0.25 + 2.0 * std::sin(0.4 * spatialRelative.x)) &&
            near(spatialValue.y, -0.5 + 3.0 * std::sin(0.2 * spatialRelative.y)),
        "spatial pattern varies non-affinely in full-resolution pixel coordinates");

  int abortChecks = 0;
  FlowRequest aborting = request;
  aborting.abortRequested = [&abortChecks]() { return ++abortChecks >= 2; };
  FlowResult aborted = estimator.estimate(a, b, aborting);
  check(aborted.status == FlowEstimateStatus::kAborted && !aborted.link.isValid(),
        "abort during analytic work returns no publishable link");

  FlowRequest invalid = request;
  invalid.columns = 0;
  FlowResult rejected = estimator.estimate(a, b, invalid);
  check(rejected.status == FlowEstimateStatus::kInvalidRequest,
        "invalid lattice dimensions are reported without throwing");
  FlowResult emptyRejected = estimator.estimate(OwnedFrame(), b, request);
  check(emptyRejected.status == FlowEstimateStatus::kInvalidRequest,
        "empty input frames are rejected without throwing");

  if (failures == 0) std::cout << "Inference contract tests passed\n";
  return failures == 0 ? 0 : 1;
}
