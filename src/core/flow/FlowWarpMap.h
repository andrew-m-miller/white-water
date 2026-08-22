// Adapt a completed backward flow to the core resampler's WarpMap interface.

#ifndef WHITEWATER_CORE_FLOW_FLOWWARPMAP_H
#define WHITEWATER_CORE_FLOW_FLOWWARPMAP_H

#include <utility>

#include "core/flow/FlowLink.h"
#include "core/warp/WarpMap.h"

namespace whitewater {

// The adapter owns an immutable copy of the field.  That keeps its lifetime independent of
// a temporary chain result and makes the render-time map safe to call concurrently from
// every row worker, matching WarpMap's contract.
class FlowWarpMap final : public WarpMap {
 public:
  explicit FlowWarpMap(const FlowField &field) : field_(field) {}
  explicit FlowWarpMap(FlowField &&field) : field_(std::move(field)) {}
  explicit FlowWarpMap(const FlowLink &link) : field_(link.field()) {}

  Vec2 mapToSource(Vec2 destination) const override {
    return destination + sampleFlow(field_, destination);
  }

  const FlowField &field() const { return field_; }

 private:
  FlowField field_;
};

// A short name is useful in code that already calls all maps "flow maps"; retain the
// explicit FlowWarpMap name for callers constructing the adapter at the resampler boundary.
using FlowToWarpMap = FlowWarpMap;

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FLOWWARPMAP_H
