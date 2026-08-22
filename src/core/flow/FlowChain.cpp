#include "core/flow/FlowChain.h"

#include <stdexcept>

namespace whitewater {

FlowChainPlan FlowChain::makePlan(int frameTime, int referenceTime) {
  FlowChainPlan result;
  result.frameTime = frameTime;
  result.referenceTime = referenceTime;

  if (frameTime > referenceTime) {
    for (int current = frameTime; current > referenceTime; --current) {
      result.links.emplace_back(current, current - 1);
    }
  } else if (frameTime < referenceTime) {
    for (int current = frameTime; current < referenceTime; ++current) {
      result.links.emplace_back(current, current + 1);
    }
  }
  return result;
}

FlowChainResult FlowChain::accumulateResult(const std::vector<FlowLink> &links) const {
  if (links.size() != plan_.links.size()) {
    throw std::invalid_argument("FlowChain received the wrong number of links");
  }
  if (plan_.isIdentity()) {
    return FlowChainResult{true, std::nullopt};
  }

  for (std::size_t index = 0; index < links.size(); ++index) {
    const FlowLinkRequest &request = plan_.links[index];
    if (!links[index].isValid() || links[index].fromTime() != request.fromTime ||
        links[index].toTime() != request.toTime) {
      throw std::invalid_argument("FlowChain link direction does not match its request");
    }
  }

  FlowLink accumulated = links.front();
  for (std::size_t index = 1; index < links.size(); ++index) {
    accumulated = compose(accumulated, links[index]);
  }
  return FlowChainResult{false, std::move(accumulated)};
}

FlowLink FlowChain::accumulate(const std::vector<FlowLink> &links) const {
  FlowChainResult result = accumulateResult(links);
  if (result.isIdentity()) {
    throw std::invalid_argument("an identity FlowChain has no accumulated FlowLink");
  }
  return std::move(result.link.value());
}

FlowField FlowChain::accumulateField(const std::vector<FlowLink> &links, int columns, int rows,
                                     const FieldGeometry &geometry) const {
  if (plan_.isIdentity()) {
    if (!links.empty()) {
      throw std::invalid_argument("identity FlowChain cannot accept links");
    }
    return identityField(columns, rows, geometry);
  }

  const FlowLink result = accumulate(links);
  if (result.field().columns() != columns || result.field().rows() != rows ||
      result.field().geometry() != geometry) {
    throw std::invalid_argument("accumulated FlowChain field does not match requested lattice");
  }
  return result.field();
}

FlowField FlowChain::identityField(int columns, int rows, const FieldGeometry &geometry) {
  return FlowField(columns, rows, geometry);
}

FlowChainPlan planFlowChain(int frameTime, int referenceTime) {
  return FlowChain::makePlan(frameTime, referenceTime);
}

FlowChainResult accumulateFlowChain(const FlowChainPlan &plan,
                                    const std::vector<FlowLink> &links) {
  const FlowChain canonical(plan.frameTime, plan.referenceTime);
  if (canonical.plan().links != plan.links) {
    throw std::invalid_argument("FlowChainPlan requests are not canonical for its endpoints");
  }
  return canonical.accumulateResult(links);
}

}  // namespace whitewater
