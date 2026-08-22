#include "core/flow/FlowCompose.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace whitewater {

namespace {

void requireField(const FlowField &field, const char *name) {
  if (field.isEmpty()) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  field.geometry().validate();
}

void requireScalar(const ScalarField &field, const char *name) {
  if (field.isEmpty()) {
    throw std::invalid_argument(std::string(name) + " must be non-empty");
  }
  field.geometry().validate();
}

void requireSameShape(const FlowField &field, const ScalarField &confidence,
                      const char *name) {
  if (field.columns() != confidence.columns() || field.rows() != confidence.rows() ||
      field.geometry() != confidence.geometry()) {
    throw std::invalid_argument(std::string(name) + " shape/geometry does not match its field");
  }
}

void requireComposable(const FlowLink &a, const FlowLink &b) {
  if (!a.isValid() || !b.isValid()) {
    throw std::invalid_argument("cannot compose an invalid FlowLink");
  }
  if (a.to() != b.from()) {
    throw std::invalid_argument("FlowLink endpoints/ intermediate geometry do not match");
  }
  if (a.modelFingerprint() != b.modelFingerprint()) {
    throw std::invalid_argument("FlowLink model fingerprints do not match");
  }
}

void requireRoundTrip(const FlowLink &forward, const FlowLink &backward) {
  if (!forward.isValid() || !backward.isValid()) {
    throw std::invalid_argument("forward/backward check requires valid FlowLinks");
  }
  if (forward.from() != backward.to() || forward.to() != backward.from()) {
    throw std::invalid_argument("forward/backward endpoints or geometry do not reverse exactly");
  }
  if (forward.modelFingerprint() != backward.modelFingerprint()) {
    throw std::invalid_argument("forward/backward model fingerprints do not match");
  }
}

float confidenceValue(float value) {
  if (!std::isfinite(value)) {
    return 0.0f;
  }
  return std::max(0.0f, std::min(1.0f, value));
}

bool isZeroField(const FlowField &field) {
  for (std::size_t index = 0; index < field.size(); ++index) {
    if (field.data()[index] != 0.0f) {
      return false;
    }
  }
  return true;
}

template <int kChannels>
Field<kChannels> smoothGaussianImpl(const Field<kChannels> &field, double sigma) {
  if (field.isEmpty()) {
    return field;
  }
  if (!std::isfinite(sigma) || sigma < 0.0) {
    throw std::invalid_argument("Gaussian sigma must be finite and non-negative");
  }
  if (sigma == 0.0) {
    return field;
  }

  const int columns = field.columns();
  const int rows = field.rows();

  auto makeKernel = [sigma](double spacing, int extent) {
    // The support is three standard deviations.  Clamp the radius before converting to an
    // int so a very large but finite sigma cannot overflow a loop bound.
    const double radiusInNodes = 3.0 * sigma / spacing;
    const int maxRadius = std::max(0, extent - 1);
    int radius = maxRadius;
    if (std::isfinite(radiusInNodes) && radiusInNodes < static_cast<double>(maxRadius)) {
      radius = static_cast<int>(std::ceil(radiusInNodes));
    }

    std::vector<double> weights(static_cast<std::size_t>(radius) * 2 + 1, 0.0);
    for (int offset = -radius; offset <= radius; ++offset) {
      // Divide before squaring so a perfectly valid subnormal sigma cannot underflow the
      // denominator to zero.  An overlarge ratio simply produces exp(-infinity) = 0,
      // which is the correct negligible tail.
      const double scaled = (offset * spacing) / sigma;
      weights[static_cast<std::size_t>(offset + radius)] =
          std::exp(-0.5 * scaled * scaled);
    }
    return std::make_pair(radius, std::move(weights));
  };

  const auto horizontalKernel = makeKernel(field.geometry().spacingX, columns);
  const auto verticalKernel = makeKernel(field.geometry().spacingY, rows);
  const int horizontalRadius = horizontalKernel.first;
  const int verticalRadius = verticalKernel.first;
  const std::vector<double> &horizontalWeights = horizontalKernel.second;
  const std::vector<double> &verticalWeights = verticalKernel.second;

  // The intermediate has the same interleaved layout as the field.  All accumulation is in
  // double so the result is independent of whether a particular axis has an even or odd
  // support, then is rounded once at the field boundary.
  std::vector<float> horizontal(field.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float *out = horizontal.data() +
                   (static_cast<std::size_t>(row) * columns + column) * kChannels;
      for (int channel = 0; channel < kChannels; ++channel) {
        double sum = 0.0;
        double weightSum = 0.0;
        for (int offset = -horizontalRadius; offset <= horizontalRadius; ++offset) {
          const int sampleColumn =
              std::max(0, std::min(columns - 1, column + offset));
          const double weight = horizontalWeights[static_cast<std::size_t>(offset +
                                                                            horizontalRadius)];
          const float value = field.node(sampleColumn, row)[channel];
          sum += weight * static_cast<double>(value);
          weightSum += weight;
        }
        out[channel] = static_cast<float>(sum / weightSum);
      }
    }
  }

  Field<kChannels> smoothed(columns, rows, field.geometry());
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      float *out = smoothed.node(column, row);
      for (int channel = 0; channel < kChannels; ++channel) {
        double sum = 0.0;
        double weightSum = 0.0;
        for (int offset = -verticalRadius; offset <= verticalRadius; ++offset) {
          const int sampleRow = std::max(0, std::min(rows - 1, row + offset));
          const double weight = verticalWeights[static_cast<std::size_t>(offset +
                                                                          verticalRadius)];
          const float *value = horizontal.data() +
                               (static_cast<std::size_t>(sampleRow) * columns + column) *
                                   kChannels;
          sum += weight * static_cast<double>(value[channel]);
          weightSum += weight;
        }
        out[channel] = static_cast<float>(sum / weightSum);
      }
    }
  }
  return smoothed;
}

ScalarField propagateRoundTripConfidence(const FlowLink &forward, const FlowLink &backward,
                                          const ScalarField &forwardConfidence,
                                          const ScalarField &backwardConfidence,
                                          const ScalarField &base) {
  requireScalar(forwardConfidence, "forward confidence");
  requireScalar(backwardConfidence, "backward confidence");
  requireSameShape(forward.field(), forwardConfidence, "forward confidence");
  requireSameShape(backward.field(), backwardConfidence, "backward confidence");
  if (base.geometry() != forward.fromGeometry() || base.columns() != forward.field().columns() ||
      base.rows() != forward.field().rows()) {
    throw std::invalid_argument("round-trip confidence geometry does not match residual");
  }

  ScalarField result(base.columns(), base.rows(), base.geometry());
  for (int row = 0; row < result.rows(); ++row) {
    for (int column = 0; column < result.columns(); ++column) {
      const Vec2 q = result.positionOf(column, row);
      const Vec2 displacement = flowNode(forward.field(), column, row);
      const float first = confidenceValue(forwardConfidence.node(column, row)[0]);
      const float second = confidenceValue(
          sampleScalar(backwardConfidence, q + displacement));
      const float baseValue = confidenceValue(base.node(column, row)[0]);
      result.node(column, row)[0] = first * second * baseValue;
    }
  }
  return result;
}

}  // namespace

FlowField composeFields(const FlowField &a, const FlowField &b) {
  requireField(a, "first field");
  requireField(b, "second field");

  // A raw field does not carry endpoint labels, so it cannot decide whether two distinct
  // geometries are an invalid mismatch or the expected source/intermediate pair.  Sampling
  // b in its own lattice is mathematically well-defined; the typed FlowLink overload is the
  // boundary that rejects an actually inconsistent intermediate geometry.
  if (isZeroField(b)) {
    return a;
  }
  FlowField result(a.columns(), a.rows(), a.geometry());
  for (int row = 0; row < a.rows(); ++row) {
    for (int column = 0; column < a.columns(); ++column) {
      const Vec2 q = a.positionOf(column, row);
      const Vec2 first = flowNode(a, column, row);
      const Vec2 second = sampleFlow(b, q + first);
      setFlow(&result, column, row, first + second);
    }
  }
  return result;
}

FlowLink compose(const FlowLink &a, const FlowLink &b) {
  requireComposable(a, b);
  FlowField result = composeFields(a.field(), b.field());
  return FlowLink(a.from(), b.to(), std::move(result), a.modelFingerprint());
}

ScalarField composeConfidence(const ScalarField &confidenceA,
                              const ScalarField &confidenceB, const FlowField &pathA) {
  requireField(pathA, "first path");
  requireScalar(confidenceA, "first confidence");
  requireScalar(confidenceB, "second confidence");
  requireSameShape(pathA, confidenceA, "first confidence");

  ScalarField result(pathA.columns(), pathA.rows(), pathA.geometry());
  for (int row = 0; row < pathA.rows(); ++row) {
    for (int column = 0; column < pathA.columns(); ++column) {
      const Vec2 q = pathA.positionOf(column, row);
      const Vec2 displacement = flowNode(pathA, column, row);
      const float first = confidenceValue(confidenceA.node(column, row)[0]);
      const float second = confidenceValue(sampleScalar(confidenceB, q + displacement));
      result.node(column, row)[0] = first * second;
    }
  }
  return result;
}

ScalarField composeConfidence(const FlowLink &a, const ScalarField &confidenceA,
                              const FlowLink &b, const ScalarField &confidenceB) {
  requireComposable(a, b);
  if (confidenceA.geometry() != a.fromGeometry() || confidenceB.geometry() != b.fromGeometry()) {
    throw std::invalid_argument("confidence geometry does not match its FlowLink");
  }
  if (confidenceA.columns() != a.field().columns() || confidenceA.rows() != a.field().rows() ||
      confidenceB.columns() != b.field().columns() || confidenceB.rows() != b.field().rows()) {
    throw std::invalid_argument("confidence shape does not match its FlowLink");
  }
  return composeConfidence(confidenceA, confidenceB, a.field());
}

ScalarField composeConfidence(const FlowLink &a, const FlowLink &b,
                              const ScalarField &confidenceA,
                              const ScalarField &confidenceB) {
  return composeConfidence(a, confidenceA, b, confidenceB);
}

FlowField forwardBackwardResidual(const FlowLink &forward, const FlowLink &backward) {
  requireRoundTrip(forward, backward);
  FlowField residual(forward.field().columns(), forward.field().rows(), forward.fromGeometry());
  for (int row = 0; row < residual.rows(); ++row) {
    for (int column = 0; column < residual.columns(); ++column) {
      const Vec2 q = residual.positionOf(column, row);
      const Vec2 first = flowNode(forward.field(), column, row);
      const Vec2 second = sampleFlow(backward.field(), q + first);
      setFlow(&residual, column, row, first + second);
    }
  }
  return residual;
}

ScalarField confidenceFromResidual(const FlowField &residual, double tolerance) {
  requireField(residual, "residual field");
  if (!std::isfinite(tolerance) || tolerance < 0.0) {
    throw std::invalid_argument("forward/backward tolerance must be finite and non-negative");
  }

  ScalarField result(residual.columns(), residual.rows(), residual.geometry());
  for (int row = 0; row < residual.rows(); ++row) {
    for (int column = 0; column < residual.columns(); ++column) {
      const Vec2 value = flowNode(residual, column, row);
      const double magnitude = length(value);
      float confidence = 0.0f;
      if (tolerance == 0.0) {
        confidence = magnitude == 0.0 ? 1.0f : 0.0f;
      } else if (std::isfinite(magnitude)) {
        confidence = static_cast<float>(std::max(0.0, std::min(1.0, 1.0 - magnitude / tolerance)));
      }
      result.node(column, row)[0] = confidence;
    }
  }
  return result;
}

ScalarField forwardBackwardConfidence(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance) {
  return confidenceFromResidual(forwardBackwardResidual(forward, backward), tolerance);
}

ScalarField forwardBackwardConfidence(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance, const ScalarField &forwardConfidence,
                                      const ScalarField &backwardConfidence) {
  requireRoundTrip(forward, backward);
  const FlowField residual = forwardBackwardResidual(forward, backward);
  const ScalarField base = confidenceFromResidual(residual, tolerance);
  return propagateRoundTripConfidence(forward, backward, forwardConfidence, backwardConfidence,
                                      base);
}

ForwardBackwardResult forwardBackward(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance) {
  ForwardBackwardResult result;
  result.residual = forwardBackwardResidual(forward, backward);
  result.confidence = confidenceFromResidual(result.residual, tolerance);
  return result;
}

FlowField smoothGaussian(const FlowField &field, double sigma) {
  return smoothGaussianImpl(field, sigma);
}

ScalarField smoothGaussian(const ScalarField &field, double sigma) {
  return smoothGaussianImpl(field, sigma);
}

}  // namespace whitewater
