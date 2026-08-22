// A direction-labelled pairwise backward flow and the coordinate spaces it connects.
//
// A Field is deliberately only storage.  It does not say which frame it came from, which
// frame it points toward, or which model produced it.  FlowLink is the value that makes
// those facts part of the algebra, so composing a link in the wrong direction fails at a
// checked boundary instead of producing a plausible but wrong warp.

#ifndef WHITEWATER_CORE_FLOW_FLOWLINK_H
#define WHITEWATER_CORE_FLOW_FLOWLINK_H

#include <string>
#include <utility>

#include "core/flow/Field.h"

namespace whitewater {

// A temporal endpoint and the image coordinate space used at that time.  The geometry is
// not inferred from the other endpoint: anamorphic and odd-sized analysis lattices must be
// represented independently in both directions.
struct FlowEndpoint {
  int time = 0;
  FieldGeometry geometry;

  FlowEndpoint() = default;
  FlowEndpoint(int timeValue, FieldGeometry geometryValue)
      : time(timeValue), geometry(std::move(geometryValue)) {}
};

inline bool operator==(const FlowEndpoint &a, const FlowEndpoint &b) {
  return a.time == b.time && a.geometry == b.geometry;
}

inline bool operator!=(const FlowEndpoint &a, const FlowEndpoint &b) { return !(a == b); }

// A pairwise backward displacement.  For a link from A to B, field(q) is the displacement
// from a destination point q in A's space to the source point in B's space.  Consequently
// the field lattice has `from.geometry`; the destination geometry is carried explicitly so
// composition can validate the intermediate coordinate space.
class FlowLink {
 public:
  FlowLink() = default;

  FlowLink(FlowEndpoint from, FlowEndpoint to, FlowField backwardDisplacement,
           std::string modelFingerprint);

  // Common case: both endpoints share the field's lattice geometry.  Links with distinct
  // endpoint geometries use the explicit FlowEndpoint constructor above.
  static FlowLink withSharedGeometry(int fromTime, int toTime,
                                     FlowField backwardDisplacement,
                                     std::string modelFingerprint);

  // A zero displacement link is useful when a caller needs a concrete value for an exact
  // identity chain.  FlowChain also represents identity as an empty request list, so this
  // helper is optional rather than a hidden special case in composition.
  static FlowLink identity(int time, int columns, int rows, const FieldGeometry &geometry,
                           std::string modelFingerprint);

  bool isValid() const { return valid_; }
  bool isIdentity() const;

  const FlowEndpoint &from() const { return from_; }
  const FlowEndpoint &to() const { return to_; }
  int fromTime() const { return from_.time; }
  int toTime() const { return to_.time; }
  const FieldGeometry &fromGeometry() const { return from_.geometry; }
  const FieldGeometry &toGeometry() const { return to_.geometry; }
  const std::string &modelFingerprint() const { return modelFingerprint_; }

  const FlowField &field() const { return backwardDisplacement_; }

 private:
  FlowEndpoint from_;
  FlowEndpoint to_;
  FlowField backwardDisplacement_;
  std::string modelFingerprint_;
  bool valid_ = false;
};

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FLOWLINK_H
