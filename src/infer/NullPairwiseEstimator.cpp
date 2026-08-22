#include "infer/NullPairwiseEstimator.h"

#include <cmath>
#include <exception>
#include <utility>

namespace whitewater {
namespace {

FlowResult failure(FlowEstimateStatus status, std::string message) {
  FlowResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

bool abortRequested(const FlowRequest &request) {
  return request.abortRequested && request.abortRequested();
}

}  // namespace

FlowResult NullPairwiseEstimator::estimate(const OwnedFrame &a, const OwnedFrame &b,
                                           const FlowRequest &request) {
  if (a.isEmpty() || b.isEmpty())
    return failure(FlowEstimateStatus::kInvalidRequest,
                   "NullPairwiseEstimator requires two non-empty frames");
  if (request.columns <= 0 || request.rows <= 0)
    return failure(FlowEstimateStatus::kInvalidRequest,
                   "flow lattice dimensions must be positive");
  if (parameters_.fingerprint.empty())
    return failure(FlowEstimateStatus::kInvalidRequest,
                   "the synthetic estimator fingerprint must not be empty");
  if (abortRequested(request))
    return failure(FlowEstimateStatus::kAborted, "flow estimation aborted before work");

  try {
    request.fromGeometry.validate();
    request.toGeometry.validate();
    FlowField field(request.columns, request.rows, request.fromGeometry);
    const Vec2 origin = request.fromGeometry.origin;

    for (int row = 0; row < field.rows(); ++row) {
      if (abortRequested(request))
        return failure(FlowEstimateStatus::kAborted, "flow estimation aborted");
      for (int column = 0; column < field.columns(); ++column) {
        const Vec2 q = field.positionOf(column, row);
        const Vec2 relative = q - origin;
        Vec2 displacement;
        switch (parameters_.pattern) {
          case NullFlowPattern::kIdentity:
            break;
          case NullFlowPattern::kTranslation:
            displacement = parameters_.translation;
            break;
          case NullFlowPattern::kAffine:
            displacement = Vec2(
                parameters_.translation.x + parameters_.xx * relative.x +
                    parameters_.xy * relative.y,
                parameters_.translation.y + parameters_.yx * relative.x +
                    parameters_.yy * relative.y);
            break;
          case NullFlowPattern::kSpatial:
            displacement = Vec2(
                parameters_.translation.x +
                    parameters_.spatialAmplitude.x *
                        std::sin(parameters_.spatialFrequency.x * relative.x),
                parameters_.translation.y +
                    parameters_.spatialAmplitude.y *
                        std::sin(parameters_.spatialFrequency.y * relative.y));
            break;
        }
        setFlow(&field, column, row, displacement);
      }
    }

    FlowResult result;
    result.status = FlowEstimateStatus::kSuccess;
    result.link = FlowLink(request.fromTime, request.toTime, request.fromGeometry,
                           request.toGeometry, std::move(field), parameters_.fingerprint);
    if (parameters_.emitConfidence) {
      ScalarField confidence(request.columns, request.rows, request.fromGeometry);
      for (std::size_t index = 0; index < confidence.size(); ++index)
        confidence.data()[index] = 1.0f;
      result.confidence = std::move(confidence);
    }
    return result;
  } catch (const std::exception &error) {
    return failure(FlowEstimateStatus::kInvalidRequest, error.what());
  } catch (...) {
    return failure(FlowEstimateStatus::kFailed,
                   "unknown failure in NullPairwiseEstimator");
  }
}

}  // namespace whitewater
