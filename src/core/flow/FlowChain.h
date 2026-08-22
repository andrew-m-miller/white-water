// Pure reference-frame chain planning and accumulation.

#ifndef WHITEWATER_CORE_FLOW_FLOWCHAIN_H
#define WHITEWATER_CORE_FLOW_FLOWCHAIN_H

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/flow/FlowCompose.h"

namespace whitewater {

// One inference request.  Requests are ordered from the rendered frame toward the
// reference, so the first request's output is the next request's input.
struct FlowLinkRequest {
  int fromTime = 0;
  int toTime = 0;

  FlowLinkRequest() = default;
  FlowLinkRequest(int from, int to) : fromTime(from), toTime(to) {}
};

inline bool operator==(const FlowLinkRequest &a, const FlowLinkRequest &b) {
  return a.fromTime == b.fromTime && a.toTime == b.toTime;
}

inline bool operator!=(const FlowLinkRequest &a, const FlowLinkRequest &b) {
  return !(a == b);
}

struct FlowChainPlan {
  int frameTime = 0;
  int referenceTime = 0;
  std::vector<FlowLinkRequest> links;

  bool isIdentity() const { return frameTime == referenceTime && links.empty(); }
  const std::vector<FlowLinkRequest> &requests() const { return links; }
  std::size_t size() const { return links.size(); }
  bool empty() const { return links.empty(); }
  const FlowLinkRequest &operator[](std::size_t index) const { return links[index]; }
  std::vector<FlowLinkRequest>::const_iterator begin() const { return links.begin(); }
  std::vector<FlowLinkRequest>::const_iterator end() const { return links.end(); }
};

// A result that can represent the exact N == R identity without inventing a synthetic
// model link or a model fingerprint.  For a non-identity chain, link is engaged and contains
// the composed backward map.  The convenience accumulate() method below is provided for
// callers that know their plan cannot be identity.
struct FlowChainResult {
  bool identity = false;
  std::optional<FlowLink> link;

  bool isIdentity() const { return identity && !link.has_value(); }
  const FlowLink &value() const { return link.value(); }
};

class FlowChain {
 public:
  FlowChain(int frameTime, int referenceTime)
      : plan_(makePlan(frameTime, referenceTime)) {}

  static FlowChainPlan makePlan(int frameTime, int referenceTime);
  static FlowChainPlan plan(int frameTime, int referenceTime) {
    return makePlan(frameTime, referenceTime);
  }

  const FlowChainPlan &plan() const { return plan_; }
  const std::vector<FlowLinkRequest> &requests() const { return plan_.links; }
  bool isIdentity() const { return plan_.isIdentity(); }

  // Validate and compose links in exactly the order returned by requests().  Endpoint and
  // fingerprint validation is delegated to FlowLink composition after this request-level
  // check, so a reversed link cannot silently form a plausible chain.
  FlowChainResult accumulateResult(const std::vector<FlowLink> &links) const;

  // Convenience for non-identity chains.  Calling this for N == R is an error because there
  // is no link to return; use accumulateResult() or identityField() for that case.
  FlowLink accumulate(const std::vector<FlowLink> &links) const;

  // Return a displacement field for either case.  The empty identity plan returns an exact
  // zero field in the supplied lattice; non-identity plans validate that the accumulated
  // field starts in that lattice.  The model fingerprint is intentionally not needed for a
  // field-only result.
  FlowField accumulateField(const std::vector<FlowLink> &links, int columns, int rows,
                            const FieldGeometry &geometry) const;

  static FlowField identityField(int columns, int rows, const FieldGeometry &geometry);
  static FlowLink identityLink(int time, int columns, int rows, const FieldGeometry &geometry,
                               std::string modelFingerprint) {
    return FlowLink::identity(time, columns, rows, geometry, std::move(modelFingerprint));
  }

 private:
  FlowChainPlan plan_;
};

// Free functions make the pure planner usable without retaining a FlowChain object.
FlowChainPlan planFlowChain(int frameTime, int referenceTime);
inline std::vector<FlowLinkRequest> requestedFlowLinks(int frameTime, int referenceTime) {
  return planFlowChain(frameTime, referenceTime).links;
}

FlowChainResult accumulateFlowChain(const FlowChainPlan &plan,
                                    const std::vector<FlowLink> &links);

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FLOWCHAIN_H
