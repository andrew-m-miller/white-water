#include "core/flow/FlowLink.h"

#include <stdexcept>

namespace whitewater {

FlowLink::FlowLink(FlowEndpoint from, FlowEndpoint to, FlowField backwardDisplacement,
                   std::string modelFingerprint)
    : from_(std::move(from)),
      to_(std::move(to)),
      backwardDisplacement_(std::move(backwardDisplacement)),
      modelFingerprint_(std::move(modelFingerprint)) {
  from_.geometry.validate();
  to_.geometry.validate();
  if (backwardDisplacement_.isEmpty()) {
    throw std::invalid_argument("FlowLink requires a non-empty displacement field");
  }
  if (backwardDisplacement_.geometry() != from_.geometry) {
    throw std::invalid_argument("FlowLink field geometry does not match its from endpoint");
  }
  valid_ = true;
}

FlowLink FlowLink::identity(int time, int columns, int rows, const FieldGeometry &geometry,
                            std::string modelFingerprint) {
  return FlowLink(FlowEndpoint(time, geometry), FlowEndpoint(time, geometry),
                  FlowField(columns, rows, geometry), std::move(modelFingerprint));
}

bool FlowLink::isIdentity() const {
  if (!valid_) {
    return false;
  }
  for (std::size_t i = 0; i < backwardDisplacement_.size(); ++i) {
    if (backwardDisplacement_.data()[i] != 0.0f) {
      return false;
    }
  }
  return true;
}

}  // namespace whitewater
