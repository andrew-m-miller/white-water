// Vendored from warp-drive 887a123, src/ofx/HostQuirks.cpp. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "ofx/HostQuirks.h"

#include <cstdio>

#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

namespace {

HostQuirks gQuirks;
bool gResolved = false;

// Flame, Flare, Flame Assist and Smoke are one application with four licences and one OFX
// implementation, so they share one quirk set. Matching on a prefix rather than an exact
// identifier also covers whatever Autodesk renames the family to next.
bool looksLikeFlame(const std::string &hostName) {
  static const char *const kIdentifiers[] = {
      "com.autodesk.flame", "com.autodesk.flare", "com.autodesk.flameassist",
      "com.autodesk.smoke", "com.autodesk.lustre"};
  for (const char *identifier : kIdentifiers) {
    if (hostName.compare(0, std::string(identifier).size(), identifier) == 0) {
      return true;
    }
  }
  return false;
}

const char *yesNo(bool value) { return value ? "yes" : "no"; }

}  // namespace

const HostQuirks &hostQuirks() { return gQuirks; }

void resolveHostQuirks() {
  if (gResolved) {
    return;
  }
  gResolved = true;

  const OFX::ImageEffectHostDescription *host = OFX::getImageEffectHostDescription();
  if (host == nullptr) {
    return;
  }

  gQuirks.name = host->hostName;
  gQuirks.label = host->hostLabel;
  gQuirks.versionMajor = host->versionMajor;
  gQuirks.versionMinor = host->versionMinor;

  gQuirks.supportsMultiResolution = host->supportsMultiResolution;
  gQuirks.supportsTiles = host->supportsTiles;
  gQuirks.supportsOverlays = host->supportsOverlays;
  gQuirks.supportsCustomInteracts = host->supportsCustomInteract;
  gQuirks.supportsParametricParams = host->supportsParametricParameter;
  gQuirks.supportsMessageSuiteV2 = host->supportsMessageSuiteV2;

  gQuirks.isFlame = looksLikeFlame(host->hostName);
  if (!gQuirks.isFlame) {
    return;
  }

  // Everything below is Flame 2026.2, measured. It is applied to every Flame version
  // because we have measured no other; if an older release turns out to differ, this is
  // the branch that grows a version test.
  gQuirks.gpuSupportPropertiesAreStrings = true;
  gQuirks.nativeOriginIsReadable = false;
  gQuirks.parameterPanelIsFlat = true;
  gQuirks.usefulLabelLength = 12;
  gQuirks.interactsReceiveKeyEvents = false;
  gQuirks.interactPixelScaleIsTrustworthy = false;
  gQuirks.rectanglePrimitiveDrawsFilled = true;
}

void reportHostQuirks() {
  const HostQuirks &q = gQuirks;
  std::fprintf(stderr,
               "White Water: host \"%s\" (%s) version %d.%d\n"
               "  recognized as Flame family : %s\n"
               "  multi-resolution / tiles   : %s / %s\n"
               "  overlays / custom interacts: %s / %s\n"
               "  parametric parameters      : %s\n"
               "  persistent message suite V2: %s\n"
               "  GPU support props are text : %s\n"
               "  native origin readable     : %s\n"
               "  parameter panel is flat    : %s (label budget %zu chars)\n"
               "  interacts get key events   : %s\n"
               "  interact pixel scale usable: %s\n"
               "  rectangle primitive fills  : %s\n",
               q.name.c_str(), q.label.c_str(), q.versionMajor, q.versionMinor,
               yesNo(q.isFlame), yesNo(q.supportsMultiResolution), yesNo(q.supportsTiles),
               yesNo(q.supportsOverlays), yesNo(q.supportsCustomInteracts),
               yesNo(q.supportsParametricParams), yesNo(q.supportsMessageSuiteV2),
               yesNo(q.gpuSupportPropertiesAreStrings),
               yesNo(q.nativeOriginIsReadable), yesNo(q.parameterPanelIsFlat),
               q.usefulLabelLength, yesNo(q.interactsReceiveKeyEvents),
               yesNo(q.interactPixelScaleIsTrustworthy),
               yesNo(q.rectanglePrimitiveDrawsFilled));
}

}  // namespace ofx
}  // namespace whitewater
