// Vendored from warp-drive 887a123, src/ofx/HostQuirks.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// What a host actually does, keyed on which host it is.
//
// Everything in here was measured by the probe plugin and written down in
// docs/host-notes.md; nothing in it comes from vendor documentation, because for the
// capabilities this project depends on the vendor documentation is wrong in both
// directions -- it omits overlay interacts, which work, and it is silent about the GPU
// property types, which do not behave as the specification says.
//
// The point of collecting it here rather than sprinkling `if (host == flame)` through the
// plugin is that a host difference then has exactly one place to be declared, one place to
// be corrected when a newer version fixes it, and one place to be read by somebody trying
// to work out why the plugin behaves differently in two applications. The struct is
// therefore mostly *data*: several fields have no consumer in the render path at all and
// exist because the overlay and editor phases will need them and because they are dumped
// to stderr at load, where Flame's shell log picks them up.
//
// Versioning. Everything measured is Flame 2026.2. The resolved quirks carry the host's
// reported version so that a future divergence -- an older Flame that does not deliver pen
// events, say -- has a natural place to be expressed, rather than needing this file to be
// restructured first.

#ifndef WHITEWATER_OFX_HOSTQUIRKS_H
#define WHITEWATER_OFX_HOSTQUIRKS_H

#include <cstddef>
#include <string>

namespace whitewater {
namespace ofx {

struct HostQuirks {
  // --- identity, as reported ---
  std::string name;   // kOfxPropName, e.g. "com.autodesk.flame"
  std::string label;  // kOfxPropLabel
  int versionMajor = 0;
  int versionMinor = 0;

  // Flame and its siblings share one OFX implementation, so they share one quirk set.
  bool isFlame = false;

  // --- capabilities the host reports, mirrored here so the whole picture is in one
  // struct rather than split between this and OFX::getImageEffectHostDescription() ---
  bool supportsMultiResolution = true;
  bool supportsTiles = false;
  bool supportsOverlays = false;
  bool supportsCustomInteracts = false;
  bool supportsParametricParams = false;

  // The message suite V2 carries setPersistentMessage/clearPersistentMessage -- the only
  // OFX channel for a message that stays attached to the effect (a render-time document
  // parse failure, a clip-format mismatch) rather than a transient dialog. The support
  // library fetches it optionally, so its absence is not fatal and its presence is a real
  // per-host fact worth recording. The transient message suite V1 (sendMessage, used for
  // Edit-button failures) is mandatory: a host lacking it cannot load this plugin at all, so
  // there is nothing to record for V1 -- if the plugin is running, V1 is present.
  bool supportsMessageSuiteV2 = false;

  // --- measured behaviour the host does not report ---

  // kOfxImageEffectPropOpenGLRenderSupported and friends read as "true"/"false" strings in
  // Flame; reading the same property as an int returns kOfxStatErrUnknown. The C++ support
  // library already reads them as strings, so nothing in the render path has to care --
  // but any code that reaches for these properties directly must, and a future OpenGL
  // render path is exactly such code.
  bool gpuSupportPropertiesAreStrings = false;

  // kOfxImageEffectHostPropNativeOrigin exists in Flame with dimension 0, so the origin
  // cannot be learned from the host. The spec's pre-1.4 default -- bottom left -- is what
  // we assume, and a viewer screenshot confirmed it empirically.
  bool nativeOriginIsReadable = true;

  // Flame lays parameters out in flat columns and does not visibly nest a Group's
  // children. Panel structure is therefore not a thing to depend on: parameter *order* is
  // the only layout tool that survives.
  bool parameterPanelIsFlat = false;

  // Flame truncates parameter labels in the panel: every "probe OfxParamTypeX" label
  // rendered as "probe OfxParamT". Labels have to be short enough to be read.
  std::size_t usefulLabelLength = 64;

  // An overlay in Flame receives pen down, motion and up with correct image-space
  // coordinates, but no key events at all across four probe sessions. This is the finding
  // that moved editing out of the viewer and into a separate application.
  bool interactsReceiveKeyEvents = true;

  // kOfxInteractPropPixelScale reads [1,1] in Flame regardless of viewer zoom, so a
  // fixed screen-space grab radius cannot be converted into image space.
  bool interactPixelScaleIsTrustworthy = true;

  // kOfxDrawPrimitiveRectangle renders filled in Flame. Outlines must be line loops.
  bool rectanglePrimitiveDrawsFilled = false;
};

// Resolved once, from the host description the support library fetched at load. Calling
// this before the host has been set returns a default-constructed set, which is the
// conservative answer rather than a crash.
const HostQuirks &hostQuirks();

// Resolves the quirks from the current host description. Called from the plugin factory's
// load action, which is the first point at which the host description exists.
void resolveHostQuirks();

// One line per field, to stderr. Flame captures plugin stderr in its shell log under
// /opt/Autodesk/log, which makes this the cheapest possible answer to "what did the plugin
// think it was talking to?" on a machine we cannot attach a debugger to.
void reportHostQuirks();

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_HOSTQUIRKS_H
