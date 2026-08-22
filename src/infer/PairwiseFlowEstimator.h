#ifndef WHITEWATER_INFER_PAIRWISEFLOWESTIMATOR_H
#define WHITEWATER_INFER_PAIRWISEFLOWESTIMATOR_H

#include <functional>
#include <optional>
#include <string>

#include "core/flow/FlowLink.h"
#include "core/image/OwnedFrame.h"

namespace whitewater {

struct FlowRequest {
  int fromTime = 0;
  int toTime = 0;
  int columns = 0;
  int rows = 0;
  FieldGeometry fromGeometry;
  FieldGeometry toGeometry;

  // Called between bounded pieces of estimator work. The OFX adapter is added in Phase 4;
  // keeping this callback host-free lets the same contract serve the CLI and unit tests.
  std::function<bool()> abortRequested;
};

enum class FlowEstimateStatus {
  kSuccess,
  kAborted,
  kInvalidRequest,
  kFailed,
};

struct FlowResult {
  FlowEstimateStatus status = FlowEstimateStatus::kFailed;
  FlowLink link;
  std::optional<ScalarField> confidence;
  std::string message;

  bool succeeded() const { return status == FlowEstimateStatus::kSuccess && link.isValid(); }
};

class PairwiseFlowEstimator {
 public:
  virtual ~PairwiseFlowEstimator() = default;

  // Produces a backward link from frame a/request.fromTime to b/request.toTime.
  // Implementations report failure in FlowResult; runtime exceptions must be contained
  // before the OFX boundary eventually calls this interface.
  virtual FlowResult estimate(const OwnedFrame &a, const OwnedFrame &b,
                              const FlowRequest &request) = 0;
};

}  // namespace whitewater

#endif  // WHITEWATER_INFER_PAIRWISEFLOWESTIMATOR_H
