// Vendored from warp-drive 887a123, src/core/warp/WarpMap.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// The one interface every warp implementation presents.
//
// A warp map answers a single question: for this point in destination space, where in
// source space does it come from? Because the whole pipeline is posed destination ->
// source, that answer is the backward map the renderer needs directly, and no solver ever
// has to be numerically inverted.
//
// Implementations must be immutable and safe to call from many threads at once. That is
// not a comment on style -- the render path calls this from every worker thread with no
// synchronization, so a lazily built cache inside a warp map would be a data race. Build
// everything in the constructor.

#ifndef WHITEWATER_CORE_WARP_WARPMAP_H
#define WHITEWATER_CORE_WARP_WARPMAP_H

#include "core/geom/Vec2.h"

namespace whitewater {

class WarpMap {
 public:
  virtual ~WarpMap() = default;

  // The source-space point that the given destination-space point samples from.
  virtual Vec2 mapToSource(Vec2 destination) const = 0;

  // The backward displacement. Zero everywhere means no warp, which makes an identity
  // map trivially detectable and is why the displacement field stores this rather than
  // the absolute source position.
  Vec2 displacementAt(Vec2 destination) const { return mapToSource(destination) - destination; }
};

}  // namespace whitewater

#endif  // WHITEWATER_CORE_WARP_WARPMAP_H
