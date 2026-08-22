// Host-free operations on direction-labelled backward flow links.

#ifndef WHITEWATER_CORE_FLOW_FLOWCOMPOSE_H
#define WHITEWATER_CORE_FLOW_FLOWCOMPOSE_H

#include "core/flow/FlowLink.h"

namespace whitewater {

// Compose two raw fields using each field's own lattice.  This overload is useful for tests
// and for callers that have already established the endpoint relationship; production chain
// code should prefer compose(FlowLink, FlowLink), which also checks endpoint times, endpoint
// geometries and model fingerprints.
FlowField composeFields(const FlowField &a, const FlowField &b);
inline FlowField composeField(const FlowField &a, const FlowField &b) {
  return composeFields(a, b);
}
inline FlowField compose(const FlowField &a, const FlowField &b) {
  return composeFields(a, b);
}

// If a maps A -> B and b maps B -> C, return the A -> C map:
//
//     composed(q) = a(q) + b(q + a(q))
//
// Both links carry backward displacements, so the order here is temporal/path order, not
// lexical endpoint order.  The intermediate endpoint and model fingerprint must match
// exactly.
FlowLink compose(const FlowLink &a, const FlowLink &b);

// Confidence is deliberately a parallel field, not an optional member of FlowLink.  It is
// sampled along exactly the same warped path as the second displacement and multiplied by
// the first confidence.  Inputs are expected to be in [0, 1]; the result is clamped to that
// interval so a malformed estimator value cannot grow confidence through composition.
ScalarField composeConfidence(const ScalarField &confidenceA,
                              const ScalarField &confidenceB, const FlowField &pathA);

ScalarField composeConfidence(const FlowLink &a, const ScalarField &confidenceA,
                              const FlowLink &b, const ScalarField &confidenceB);

// Same operation with the link arguments first, for callers that naturally mirror compose.
ScalarField composeConfidence(const FlowLink &a, const FlowLink &b,
                              const ScalarField &confidenceA,
                              const ScalarField &confidenceB);

// Forward/backward consistency.  For f: A -> B and b: B -> A, the residual at q in A is
// f(q) + b(q + f(q)).  It is a displacement field, so zero means an exact round trip.
FlowField forwardBackwardResidual(const FlowLink &forward, const FlowLink &backward);
// Storage-only form for synthetic callers that have already established the reverse
// relationship outside this type.  The FlowLink form above is the checked production path.
inline FlowField forwardBackwardResidual(const FlowField &forward, const FlowField &backward) {
  return composeFields(forward, backward);
}
inline FlowField fbResidual(const FlowLink &forward, const FlowLink &backward) {
  return forwardBackwardResidual(forward, backward);
}

// Turn a residual into a confidence in [0, 1].  A residual of zero is confidence 1;
// residuals at or beyond tolerance are confidence 0, with a linear falloff in between.
// A zero tolerance is useful for exact synthetic tests and becomes an exact-zero test.
ScalarField confidenceFromResidual(const FlowField &residual, double tolerance);

ScalarField forwardBackwardConfidence(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance = 1.0);
inline ScalarField forwardBackwardConfidence(const FlowField &forward, const FlowField &backward,
                                            double tolerance = 1.0) {
  return confidenceFromResidual(forwardBackwardResidual(forward, backward), tolerance);
}

// If per-direction confidence fields are available, they are multiplied into the
// residual-derived confidence along the same path used by the check.
ScalarField forwardBackwardConfidence(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance, const ScalarField &forwardConfidence,
                                      const ScalarField &backwardConfidence);

struct ForwardBackwardResult {
  FlowField residual;
  ScalarField confidence;
};

ForwardBackwardResult forwardBackward(const FlowLink &forward, const FlowLink &backward,
                                      double tolerance = 1.0);

// Spatial-noise primitive.  sigma is in full-resolution image pixels, so anisotropic field
// lattices use independent radii and kernels according to spacingX/spacingY.  Boundaries
// are clamped to the nearest lattice node.  Smoothing is deliberately local; it makes no
// claim about temporal chain drift.
FlowField smoothGaussian(const FlowField &field, double sigma);
ScalarField smoothGaussian(const ScalarField &field, double sigma);
inline FlowField gaussianSmooth(const FlowField &field, double sigma) {
  return smoothGaussian(field, sigma);
}
inline ScalarField gaussianSmooth(const ScalarField &field, double sigma) {
  return smoothGaussian(field, sigma);
}

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FLOWCOMPOSE_H
