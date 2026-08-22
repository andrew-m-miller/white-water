#ifndef WHITEWATER_INFER_NULLPAIRWISEESTIMATOR_H
#define WHITEWATER_INFER_NULLPAIRWISEESTIMATOR_H

#include <string>
#include <utility>

#include "infer/PairwiseFlowEstimator.h"

namespace whitewater {

enum class NullFlowPattern {
  kIdentity,
  kTranslation,
  kAffine,
  kSpatial,
};

struct NullFlowParameters {
  NullFlowPattern pattern = NullFlowPattern::kIdentity;
  Vec2 translation;

  // Affine displacement around the first lattice node:
  //   d(q) = translation + matrix * (q - origin)
  double xx = 0.0;
  double xy = 0.0;
  double yx = 0.0;
  double yy = 0.0;

  // Spatial pattern adds separable sine waves in full-resolution pixel space. It is a
  // deterministic non-affine field for composition/sampling tests, not a model surrogate.
  Vec2 spatialAmplitude;
  Vec2 spatialFrequency;

  bool emitConfidence = false;
  std::string fingerprint = "null-pairwise/v1";
};

class NullPairwiseEstimator final : public PairwiseFlowEstimator {
 public:
  explicit NullPairwiseEstimator(NullFlowParameters parameters = {})
      : parameters_(std::move(parameters)) {}

  FlowResult estimate(const OwnedFrame &a, const OwnedFrame &b,
                      const FlowRequest &request) override;

 private:
  NullFlowParameters parameters_;
};

}  // namespace whitewater

#endif  // WHITEWATER_INFER_NULLPAIRWISEESTIMATOR_H
