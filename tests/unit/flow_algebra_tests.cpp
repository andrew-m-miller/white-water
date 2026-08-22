// Dependency-free checks for the host-free flow algebra.
//
// This file intentionally has a tiny failure-counter runner rather than pulling a test
// framework into src/core's dependency boundary.  Checks stay active in Release builds;
// compiling it directly is also useful while the Phase 2 targets are being assembled.

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/flow/FlowChain.h"
#include "core/flow/FlowCompose.h"
#include "core/flow/FlowWarpMap.h"

namespace {

using namespace whitewater;

constexpr double kEpsilon = 1.0e-5;
int failures = 0;

void require(bool condition, const std::string &message) {
  if (condition) return;
  ++failures;
  std::cerr << "FAIL: " << message << '\n';
}

void expectNear(double actual, double expected, double epsilon = kEpsilon,
                const char *message = "value is within tolerance") {
  require(std::abs(actual - expected) <= epsilon, message);
}

void expectThrows(const std::function<void()> &operation,
                  const char *message = "operation rejects invalid input") {
  bool threw = false;
  try {
    operation();
  } catch (const std::invalid_argument &) {
    threw = true;
  } catch (...) {
    // A different exception is still a failed contract check, but keep the runner alive so
    // the remaining algebra tests report their own results.
  }
  require(threw, message);
}

FlowField constantField(int columns, int rows, const FieldGeometry &geometry, Vec2 displacement) {
  FlowField result(columns, rows, geometry);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      setFlow(&result, column, row, displacement);
    }
  }
  return result;
}

ScalarField scalarField(int columns, int rows, const FieldGeometry &geometry, float value) {
  ScalarField result(columns, rows, geometry);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      result.node(column, row)[0] = value;
    }
  }
  return result;
}

void testGeometryAndSampling() {
  expectThrows([] { FieldGeometry(Vec2(), 0.0, 1.0); });
  expectThrows([] { FieldGeometry(Vec2(), -1.0, 1.0); });
  expectThrows([] { FieldGeometry(Vec2(), std::numeric_limits<double>::quiet_NaN(), 1.0); });
  expectThrows([] { FieldGeometry(Vec2(), 1.0, std::numeric_limits<double>::infinity()); });

  const FieldGeometry geometry = FieldGeometry::forPixels(-7, 11, 2.0, 0.5);
  const Vec2 image = geometry.latticeToImage(Vec2(3.0, 4.0));
  expectNear(image.x, 0.0);
  expectNear(image.y, 13.25);
  const Vec2 lattice = geometry.imageToLattice(image);
  expectNear(lattice.x, 3.0);
  expectNear(lattice.y, 4.0);

  FlowField field(3, 2, geometry);
  setFlow(&field, 0, 0, Vec2(2.0, -3.0));
  float sampled[2] = {0.0f, 0.0f};
  field.sample(Vec2(-100.0, -100.0), sampled);
  expectNear(sampled[0], 2.0);
  expectNear(sampled[1], -3.0);
}

void testTypedCompositionAndDirections() {
  const FieldGeometry aGeometry = FieldGeometry::forPixels(-3, 5, 2.0, 0.5);
  const FieldGeometry bGeometry = FieldGeometry::forPixels(9, -7, 0.75, 1.5);
  const FieldGeometry cGeometry = FieldGeometry::forPixels(2, 4, 1.25, 0.75);
  const int columns = 5;
  const int rows = 3;

  const FlowLink a(FlowEndpoint(12, aGeometry), FlowEndpoint(11, bGeometry),
                   constantField(columns, rows, aGeometry, Vec2(2.0, -1.0)), "model-a");
  const FlowLink b(FlowEndpoint(11, bGeometry), FlowEndpoint(10, cGeometry),
                   constantField(columns, rows, bGeometry, Vec2(-3.0, 4.0)), "model-a");
  const FlowLink composed = compose(a, b);
  require(composed.fromTime() == 12, "composition preserves the first from-time");
  require(composed.toTime() == 10, "composition preserves the final to-time");
  require(composed.fromGeometry() == aGeometry,
          "composition preserves the first from-geometry");
  require(composed.toGeometry() == cGeometry,
          "composition preserves the final to-geometry");
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const Vec2 value = flowNode(composed.field(), column, row);
      expectNear(value.x, -1.0);
      expectNear(value.y, 3.0);
    }
  }

  const FlowLink wrongIntermediate(FlowEndpoint(11, aGeometry), FlowEndpoint(10, cGeometry),
                                   constantField(columns, rows, aGeometry, Vec2()), "model-a");
  expectThrows([&] { compose(a, wrongIntermediate); });
  const FlowLink wrongFingerprint(FlowEndpoint(11, bGeometry), FlowEndpoint(10, cGeometry),
                                  constantField(columns, rows, bGeometry, Vec2()), "model-b");
  expectThrows([&] { compose(a, wrongFingerprint); });

  // Both temporal directions are valid; only a reversed request is invalid.
  const FlowLink reverse(FlowEndpoint(11, bGeometry), FlowEndpoint(12, aGeometry),
                         constantField(columns, rows, bGeometry, Vec2(-2.0, 1.0)), "model-a");
  const FlowField roundTrip = forwardBackwardResidual(a, reverse);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const Vec2 value = flowNode(roundTrip, column, row);
      expectNear(value.x, 0.0);
      expectNear(value.y, 0.0);
    }
  }
}

void testRawAffineComposition() {
  const FieldGeometry geometry = FieldGeometry::forPixels(-2, -1, 0.5, 2.0);
  FlowField first(7, 5, geometry);
  FlowField second(7, 5, geometry);
  for (int row = 0; row < first.rows(); ++row) {
    for (int column = 0; column < first.columns(); ++column) {
      const Vec2 q = first.positionOf(column, row);
      setFlow(&first, column, row, Vec2(0.25 * q.x + 1.0, -0.1 * q.y + 0.5));
      setFlow(&second, column, row, Vec2(0.4 * q.x - 2.0, 0.2 * q.y + 3.0));
    }
  }
  const FlowField result = composeFields(first, second);
  // Check an interior point where bilinear sampling of this affine field is exact.
  const int column = 3;
  const int row = 2;
  const Vec2 q = result.positionOf(column, row);
  const Vec2 a = flowNode(first, column, row);
  const Vec2 expected = a + Vec2(0.4 * (q.x + a.x) - 2.0,
                                 0.2 * (q.y + a.y) + 3.0);
  const Vec2 actual = flowNode(result, column, row);
  expectNear(actual.x, expected.x);
  expectNear(actual.y, expected.y);
}

void testSpatiallyVaryingComposition() {
  const FieldGeometry geometry = FieldGeometry::forPixels(0, 0);
  FlowField first = constantField(6, 4, geometry, Vec2(1.0, 0.0));
  FlowField second(6, 4, geometry);
  for (int row = 0; row < second.rows(); ++row) {
    for (int column = 0; column < second.columns(); ++column) {
      setFlow(&second, column, row,
              Vec2(static_cast<double>(column * column),
                   static_cast<double>(row * row + column)));
    }
  }
  const FlowField result = composeFields(first, second);
  const Vec2 value = flowNode(result, 2, 2);
  // The first link lands exactly on second node (3, 2), whose non-affine displacement is
  // (9, 7), then adds the first link's (1, 0).
  expectNear(value.x, 10.0, kEpsilon, "spatial composition samples the warped X node");
  expectNear(value.y, 7.0, kEpsilon, "spatial composition samples the warped Y node");
}

void testConfidenceAndSmoothing() {
  const FieldGeometry aGeometry = FieldGeometry::forPixels(-5, -3, 2.0, 0.5);
  const FieldGeometry bGeometry = FieldGeometry::forPixels(2, 4, 0.75, 1.25);
  const int columns = 5;
  const int rows = 3;
  const FlowLink a(FlowEndpoint(2, aGeometry), FlowEndpoint(1, bGeometry),
                   constantField(columns, rows, aGeometry, Vec2(100.0, 100.0)), "model");
  const FlowLink b = FlowLink::withSharedGeometry(
      1, 0, constantField(columns, rows, bGeometry, Vec2()), "model");
  ScalarField confidenceA = scalarField(columns, rows, aGeometry, 0.5f);
  ScalarField confidenceB = scalarField(columns, rows, bGeometry, 0.8f);
  // The 100-pixel path is clamped to the edge of confidenceB, rather than extrapolated.
  const ScalarField composed = composeConfidence(a, confidenceA, b, confidenceB);
  expectNear(composed.node(0, 0)[0], 0.4);

  FlowField impulse(5, 5, FieldGeometry::forPixels(0, 0));
  impulse.node(2, 2)[0] = 1.0f;
  const FlowField smoothed = smoothGaussian(impulse, 1.0);
  require(smoothed.node(2, 2)[0] < 1.0f, "Gaussian smoothing lowers an impulse peak");
  require(smoothed.node(1, 2)[0] > 0.0f,
          "Gaussian smoothing spreads an impulse to neighboring nodes");
  const FlowField exact = smoothGaussian(impulse, 0.0);
  for (std::size_t index = 0; index < impulse.size(); ++index) {
    require(exact.data()[index] == impulse.data()[index],
            "zero-width Gaussian smoothing is exact identity");
  }
}

void testChainPlanningAndIdentity() {
  const FlowChainPlan forward = planFlowChain(2, 6);
  require((forward.links == std::vector<FlowLinkRequest>{{2, 3}, {3, 4}, {4, 5}, {5, 6}}),
          "forward chain requests advance one frame toward the reference");
  const FlowChainPlan backward = planFlowChain(6, 2);
  require((backward.links == std::vector<FlowLinkRequest>{{6, 5}, {5, 4}, {4, 3}, {3, 2}}),
          "backward chain requests retreat one frame toward the reference");
  const FlowChainPlan identity = planFlowChain(7, 7);
  require(identity.isIdentity(), "equal frame and reference times produce identity plan");
  require(identity.links.empty(), "identity plan requests no pairwise links");

  const FieldGeometry geometry = FieldGeometry::forPixels(3, -4, 0.5, 2.0);
  const int columns = 5;
  const int rows = 3;
  FlowChain chain(300, 0);
  std::vector<FlowLink> links;
  links.reserve(chain.requests().size());
  for (const FlowLinkRequest &request : chain.requests()) {
    links.push_back(FlowLink::withSharedGeometry(
        request.fromTime, request.toTime,
        constantField(columns, rows, geometry, Vec2()), "identity"));
  }
  const FlowLink accumulated = chain.accumulate(links);
  require(accumulated.isIdentity(), "hundreds of identity links remain identity");
  for (std::size_t index = 0; index < accumulated.field().size(); ++index) {
    require(accumulated.field().data()[index] == 0.0f,
            "identity chain has exact zero displacement storage");
  }

  const FlowChain exactChain(4, 4);
  const FlowField exactIdentity = exactChain.accumulateField({}, columns, rows, geometry);
  for (std::size_t index = 0; index < exactIdentity.size(); ++index) {
    require(exactIdentity.data()[index] == 0.0f,
            "equal-time chain returns exact zero identity field");
  }

  std::vector<FlowLink> reversed = links;
  reversed[1] = FlowLink::withSharedGeometry(
      298, 299, constantField(columns, rows, geometry, Vec2()), "identity");
  expectThrows([&] { chain.accumulate(reversed); });
}

void testWarpAdapter() {
  const FieldGeometry geometry = FieldGeometry::forPixels(0, 0, 2.0, 0.5);
  FlowField field = constantField(3, 2, geometry, Vec2(4.0, -2.0));
  FlowWarpMap map(std::move(field));
  const Vec2 source = map.mapToSource(Vec2(0.5, 0.25));
  expectNear(source.x, 4.5);
  expectNear(source.y, -1.75);
}

}  // namespace

int main() {
  try {
    testGeometryAndSampling();
    testTypedCompositionAndDirections();
  testRawAffineComposition();
  testSpatiallyVaryingComposition();
    testConfidenceAndSmoothing();
    testChainPlanningAndIdentity();
    testWarpAdapter();
  } catch (const std::exception &error) {
    require(false, std::string("unexpected exception: ") + error.what());
  } catch (...) {
    require(false, "unexpected non-standard exception");
  }

  if (failures == 0) {
    std::cout << "flow algebra tests passed\n";
    return EXIT_SUCCESS;
  }
  std::cerr << failures << " flow algebra test(s) failed\n";
  return EXIT_FAILURE;
}
