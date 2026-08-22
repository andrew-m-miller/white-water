// Adapted from warp-drive 887a123, src/ofx/Plugin.cpp. MTI Film internal.
// The bundle's sole module translation unit.  All implementation code lives in the static
// host-bound library so it can be exercised without loading an OFX module in unit tests.

#include "ofx/OpticalFlowPlugin.h"
#include "ofxsImageEffect.h"

// The vendored support library wires the legacy per-plugin `setHost` callback but does not
// expose the optional OFX 1.5 module-level entry point.  Export the entry point as a thin
// forwarding shim so the bundle has the exact three-symbol surface promised by cmake/ofx.map
// and cmake/ofx.exports.  The declaration is intentionally local to this module TU; the
// support library keeps the actual host pointer in its existing private implementation.
namespace OFX {
namespace Private {
void setHost(OfxHost *host);
}  // namespace Private
}  // namespace OFX

#if defined(__GNUC__) || defined(__clang__)
#define WHITEWATER_OFX_EXPORT __attribute__((visibility("default")))
#else
#define WHITEWATER_OFX_EXPORT
#endif

extern "C" WHITEWATER_OFX_EXPORT OfxStatus OfxSetHost(const OfxHost *host) {
  OFX::Private::setHost(const_cast<OfxHost *>(host));
  return kOfxStatOK;
}

namespace OFX {
namespace Plugin {

void getPluginIDs(OFX::PluginFactoryArray &factories) {
  static whitewater::ofx::OpticalFlowPluginFactory trackFactory(
      whitewater::ofx::kOpticalFlowPluginIdentifier, 1, 0);
  static whitewater::ofx::StMapPluginFactory stMapFactory(
      whitewater::ofx::kStMapPluginIdentifier, 1, 0);
  factories.push_back(&trackFactory);
  factories.push_back(&stMapFactory);
}

}  // namespace Plugin
}  // namespace OFX

#undef WHITEWATER_OFX_EXPORT
