// White Water host capability probe.
//
// Vendored from warp-drive 887a123, tools/hostprobe/hostprobe.cpp. MTI Film internal.
// Currently unmodified apart from names: it still asks warp-drive's questions, and the
// answers it produces are already recorded as "inherited" in docs/host-notes.md.
//
// This is a diagnostic OFX plugin, not part of the shipping product. It loads in any
// OFX host, passes frames through untouched, and writes a report describing exactly what
// the host supports: suites, host properties, parameter types, overlay interacts, GPU
// render suites, out-of-render frame access, and whether a large string parameter
// survives a setup save/reload.
//
// Report location: $WHITEWATER_PROBE_LOG, else $TMPDIR/whitewater-hostprobe.txt, else
// /tmp/whitewater-hostprobe.txt. Everything is also mirrored to stderr, which Flame
// captures in its shell log.
//
// Deliberately dependency-free: OFX C headers only, no OpenGL, no support library.
// Anything between the probe and the host is something that could itself be the reason a
// measurement came out the way it did -- and the C++ support library in particular fetches
// suites unconditionally at load, so a probe built on it could not report on a host that
// lacks one.
//
// ===========================================================================
// PHASE 0 -- what this probe still has to be taught to ask
// ===========================================================================
//
// White Water depends on things warp-drive never needed, so this file is not finished.
// The five questions are stated in full, with their consequences, in the Open section of
// docs/host-notes.md; this is the map from each one to the code that has to change.
//
//   1. Two clips in the General context.
//      onDescribeInContext() near line 741 defines one source and one output. Add an
//      optional second input clip (kOfxImageClipPropOptional), then report from
//      probeInstanceProperties() what the host says about each: connectedness, components,
//      and what clipGetImage returns on the disconnected one.
//
//   2. clipGetImage at arbitrary times DURING render.  <-- the load-bearing one
//      probeOutOfRenderFrameAccess() near line 519 already does exactly the right thing,
//      but it is called from the instance-changed action (line 878) -- a host that is idle
//      and waiting on us. Call it again from onRender(), on a host thread, while the host
//      holds whatever locks a render holds. That is a different question and the whole
//      on-demand chain design rests on the answer.
//
//   3. ONNX Runtime inside Flame's process.
//      Not this file. It needs a SECOND bundle that links the runtime, so that a runtime
//      which refuses to initialise cannot take the capability probe down with it. See
//      tools/hostprobe/CMakeLists.txt.
//
//   4. Does setSupportsTiles(false) actually yield whole-frame render windows?
//      Line 720 and lines 749/754 already declare 0. onRender() near line 923 needs to
//      report the render window against the region of definition so we can see whether
//      Flame honoured it.
//
//   5. getFramesNeeded honesty.
//      There is no kOfxImageEffectActionGetFramesNeeded handler at all. Add one that
//      declares only {N}, and confirm the pulls in item 2 still succeed. Then add a build
//      or parameter switch that declares a long range instead, and record what Flame does
//      -- the fear is that it materialises hundreds of upstream frames for one output.
//
// Items 2 and 3 are the project's real risk. Either one failing changes the architecture,
// not the schedule.
// ===========================================================================

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "ofxCore.h"
#include "ofxDrawSuite.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxInteract.h"
#include "ofxKeySyms.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParam.h"
#include "ofxParametricParam.h"
#include "ofxProgress.h"
#include "ofxProperty.h"
#include "ofxTimeLine.h"

namespace {

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

OfxHost *gHost = nullptr;

const OfxImageEffectSuiteV1 *gEffect = nullptr;
const OfxPropertySuiteV1 *gProp = nullptr;
const OfxParameterSuiteV1 *gParam = nullptr;
const OfxMessageSuiteV1 *gMessage = nullptr;
const OfxMultiThreadSuiteV1 *gThread = nullptr;
const OfxDrawSuiteV1 *gDraw = nullptr;
const OfxInteractSuiteV1 *gInteract = nullptr;

bool gHostSupportsOverlays = false;
bool gOverlayRegistered = false;
bool gRenderLogged = false;
bool gHostCalledOfxSetHost = false;

// Parameter names.
const char *kParamRunProbe = "runProbe";
const char *kParamWriteBigString = "writeBigString";
const char *kParamVerifyBigString = "verifyBigString";
const char *kParamBigString = "bigString";

// Size of the payload used for the string-parameter persistence test. A spline document
// with animated shapes is realistically in this range once serialized.
const size_t kBigStringBytes = 1024 * 1024;

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

std::string reportPath() {
  if (const char *explicitPath = std::getenv("WHITEWATER_PROBE_LOG"))
    return explicitPath;
  const char *tmp = std::getenv("TMPDIR");
  std::string dir = tmp && tmp[0] ? tmp : "/tmp";
  if (!dir.empty() && dir.back() == '/')
    dir.pop_back();
  return dir + "/whitewater-hostprobe.txt";
}

void emit(const std::string &line) {
  std::fprintf(stderr, "[whitewater-probe] %s\n", line.c_str());
  std::fflush(stderr);

  static const std::string path = reportPath();
  if (FILE *f = std::fopen(path.c_str(), "a")) {
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
}

void emitf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void emitf(const char *fmt, ...) {
  char buf[4096];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);
  emit(buf);
}

void section(const char *title) {
  emit("");
  emitf("== %s ==", title);
}

const char *statusName(OfxStatus s) {
  switch (s) {
  case kOfxStatOK: return "kOfxStatOK";
  case kOfxStatFailed: return "kOfxStatFailed";
  case kOfxStatErrFatal: return "kOfxStatErrFatal";
  case kOfxStatErrUnknown: return "kOfxStatErrUnknown";
  case kOfxStatErrMissingHostFeature: return "kOfxStatErrMissingHostFeature";
  case kOfxStatErrUnsupported: return "kOfxStatErrUnsupported";
  case kOfxStatErrExists: return "kOfxStatErrExists";
  case kOfxStatErrFormat: return "kOfxStatErrFormat";
  case kOfxStatErrMemory: return "kOfxStatErrMemory";
  case kOfxStatErrBadHandle: return "kOfxStatErrBadHandle";
  case kOfxStatErrBadIndex: return "kOfxStatErrBadIndex";
  case kOfxStatErrValue: return "kOfxStatErrValue";
  case kOfxStatReplyYes: return "kOfxStatReplyYes";
  case kOfxStatReplyNo: return "kOfxStatReplyNo";
  case kOfxStatReplyDefault: return "kOfxStatReplyDefault";
  default: return "<unrecognised status>";
  }
}

// ---------------------------------------------------------------------------
// Property inspection
// ---------------------------------------------------------------------------

enum PropType { kStr, kInt, kDbl, kPtr };

struct PropSpec {
  const char *name;
  PropType type;
};

// Reads a property of a known type across all its dimensions and renders it as one line.
// A host is entitled to not implement any given property, so a failure here is data, not
// an error -- it is reported verbatim.
std::string readProp(OfxPropertySetHandle props, const char *name, PropType type) {
  if (!gProp)
    return "<no property suite>";

  int dim = 0;
  OfxStatus st = gProp->propGetDimension(props, name, &dim);
  if (st != kOfxStatOK)
    return std::string("<absent: ") + statusName(st) + ">";
  if (dim == 0)
    return "<present, dimension 0>";

  std::string out;
  for (int i = 0; i < dim; ++i) {
    if (i)
      out += ", ";
    char buf[256];
    switch (type) {
    case kStr: {
      char *value = nullptr;
      st = gProp->propGetString(props, name, i, &value);
      out += (st == kOfxStatOK && value) ? std::string("\"") + value + "\""
                                         : std::string("<") + statusName(st) + ">";
      break;
    }
    case kInt: {
      int value = 0;
      st = gProp->propGetInt(props, name, i, &value);
      std::snprintf(buf, sizeof(buf), "%d", value);
      out += (st == kOfxStatOK) ? buf : std::string("<") + statusName(st) + ">";
      break;
    }
    case kDbl: {
      double value = 0.0;
      st = gProp->propGetDouble(props, name, i, &value);
      std::snprintf(buf, sizeof(buf), "%g", value);
      out += (st == kOfxStatOK) ? buf : std::string("<") + statusName(st) + ">";
      break;
    }
    case kPtr: {
      void *value = nullptr;
      st = gProp->propGetPointer(props, name, i, &value);
      std::snprintf(buf, sizeof(buf), "%p", value);
      out += (st == kOfxStatOK) ? buf : std::string("<") + statusName(st) + ">";
      break;
    }
    }
  }
  return out;
}

void dumpProps(OfxPropertySetHandle props, const PropSpec *specs, size_t count) {
  for (size_t i = 0; i < count; ++i)
    emitf("  %-52s = %s", specs[i].name, readProp(props, specs[i].name, specs[i].type).c_str());
}

int propInt(OfxPropertySetHandle props, const char *name, int fallback) {
  int value = fallback;
  if (!gProp || gProp->propGetInt(props, name, 0, &value) != kOfxStatOK)
    return fallback;
  return value;
}

// ---------------------------------------------------------------------------
// Suite probing
// ---------------------------------------------------------------------------

const void *fetchAndReport(const char *suiteName, int version) {
  const void *suite = gHost->fetchSuite(gHost->host, suiteName, version);
  emitf("  %-34s v%-2d %s", suiteName, version, suite ? "available" : "MISSING");
  return suite;
}

void probeSuites() {
  section("Suites");

  gEffect = (const OfxImageEffectSuiteV1 *)fetchAndReport(kOfxImageEffectSuite, 1);
  gProp = (const OfxPropertySuiteV1 *)fetchAndReport(kOfxPropertySuite, 1);
  gParam = (const OfxParameterSuiteV1 *)fetchAndReport(kOfxParameterSuite, 1);
  fetchAndReport(kOfxMemorySuite, 1);
  gThread = (const OfxMultiThreadSuiteV1 *)fetchAndReport(kOfxMultiThreadSuite, 1);
  gMessage = (const OfxMessageSuiteV1 *)fetchAndReport(kOfxMessageSuite, 1);
  fetchAndReport(kOfxMessageSuite, 2);
  fetchAndReport(kOfxProgressSuite, 1);
  fetchAndReport(kOfxProgressSuite, 2);
  fetchAndReport(kOfxTimeLineSuite, 1);
  fetchAndReport(kOfxParametricParameterSuite, 1);
  gInteract = (const OfxInteractSuiteV1 *)fetchAndReport(kOfxInteractSuite, 1);
  gDraw = (const OfxDrawSuiteV1 *)fetchAndReport(kOfxDrawSuite, 1);
  fetchAndReport(kOfxOpenGLRenderSuite, 1);
  fetchAndReport(kOfxOpenCLProgramSuite, 1);

  if (gThread) {
    unsigned int cpus = 0;
    if (gThread->multiThreadNumCPUs(&cpus) == kOfxStatOK)
      emitf("  multiThreadNumCPUs                      = %u", cpus);
  }
}

void probeHostProperties() {
  section("Host properties");

  static const PropSpec specs[] = {
      {kOfxPropName, kStr},
      {kOfxPropLabel, kStr},
      {kOfxPropVersion, kInt},
      {kOfxPropVersionLabel, kStr},
      {kOfxPropAPIVersion, kInt},
      {kOfxImageEffectHostPropIsBackground, kInt},
      {kOfxImageEffectHostPropNativeOrigin, kStr},
      {kOfxImageEffectPropSupportsOverlays, kInt},
      {kOfxImageEffectPropSupportsMultiResolution, kInt},
      {kOfxImageEffectPropSupportsTiles, kInt},
      {kOfxImageEffectPropTemporalClipAccess, kInt},
      {kOfxImageEffectPropSupportsMultipleClipDepths, kInt},
      {kOfxImageEffectPropSupportsMultipleClipPARs, kInt},
      {kOfxImageEffectPropSetableFrameRate, kInt},
      {kOfxImageEffectPropSetableFielding, kInt},
      {kOfxImageEffectInstancePropSequentialRender, kInt},
      {kOfxImageEffectPropSupportedContexts, kStr},
      {kOfxImageEffectPropSupportedPixelDepths, kStr},
      {kOfxImageEffectPropSupportedComponents, kStr},
      {kOfxImageEffectPropRenderQualityDraft, kInt},
      {kOfxParamHostPropSupportsCustomInteract, kInt},
      {kOfxParamHostPropSupportsStringAnimation, kInt},
      {kOfxParamHostPropSupportsChoiceAnimation, kInt},
      {kOfxParamHostPropSupportsStrChoice, kInt},
      {kOfxParamHostPropSupportsStrChoiceAnimation, kInt},
      {kOfxParamHostPropSupportsBooleanAnimation, kInt},
      {kOfxParamHostPropSupportsCustomAnimation, kInt},
      {kOfxParamHostPropMaxParameters, kInt},
      {kOfxParamHostPropMaxPages, kInt},
      {kOfxParamHostPropPageRowColumnCount, kInt},
      {kOfxImageEffectPropOpenGLRenderSupported, kStr},
      {kOfxImageEffectPropCudaRenderSupported, kStr},
      {kOfxImageEffectPropCudaStreamSupported, kStr},
      {kOfxImageEffectPropMetalRenderSupported, kStr},
      {kOfxImageEffectPropOpenCLRenderSupported, kStr},
      {kOfxImageEffectPropOpenCLSupported, kStr},
  };

  OfxPropertySetHandle props = gHost->host;
  dumpProps(props, specs, sizeof(specs) / sizeof(specs[0]));

  gHostSupportsOverlays = propInt(props, kOfxImageEffectPropSupportsOverlays, 0) != 0;

  // Several GPU properties are documented as int in some host implementations and string
  // in others; re-read the interesting ones as int so a mismatch is visible in the report
  // rather than silently reported as absent.
  emit("  -- GPU render support, re-read as int --");
  static const PropSpec gpuAsInt[] = {
      {kOfxImageEffectPropOpenGLRenderSupported, kInt},
      {kOfxImageEffectPropCudaRenderSupported, kInt},
      {kOfxImageEffectPropMetalRenderSupported, kInt},
      {kOfxImageEffectPropOpenCLRenderSupported, kInt},
  };
  dumpProps(props, gpuAsInt, sizeof(gpuAsInt) / sizeof(gpuAsInt[0]));
}

// ---------------------------------------------------------------------------
// Overlay interact
// ---------------------------------------------------------------------------

// Per-action tallies, so the report can state how many of each interact action actually
// arrived rather than only that one did.
struct InteractStats {
  std::string action;
  long count = 0;
  int samples = 0;
};

std::vector<InteractStats> gInteractStats;

InteractStats &interactStats(const char *action) {
  for (InteractStats &s : gInteractStats)
    if (s.action == action)
      return s;
  gInteractStats.push_back(InteractStats{action, 0, 0});
  return gInteractStats.back();
}

// Logs the first occurrence of each interact action, plus the arguments that decide
// whether a real editing UI can be built on top: pen coordinates, pixel scale, and
// keyboard events.
//
// An overlay that only draws is not enough for a spline editor. It needs pen down/up as
// well as motion, keyboard events for tool switching and delete, and canonical pen
// coordinates that stay correct as the viewer is zoomed and panned. Anything missing here
// is a reason to fall back to the external editor design in docs/design/external-editor.md.
OfxStatus overlayMain(const char *action, const void *handle, OfxPropertySetHandle inArgs,
                      OfxPropertySetHandle /*outArgs*/) {
  // Counting every action, and sampling each kind separately, is what makes "the host
  // dropped my events" distinguishable from "the probe stopped printing". An earlier
  // version shared one sample budget across all pen actions, so a burst of motion events
  // silently consumed the budget before the pen-down that followed.
  InteractStats &stats = interactStats(action);
  ++stats.count;
  const bool firstTime = stats.count == 1;
  if (firstTime)
    emitf("  INTERACT ACTION: %s  (handle %p)", action, handle);

  const bool isPen = std::strncmp(action, "OfxInteractActionPen", 20) == 0;
  const bool isKey = std::strncmp(action, "OfxInteractActionKey", 20) == 0;

  // Pen motion fires continuously; a per-action budget keeps the report readable while
  // still showing enough samples to see how coordinates behave.
  if (isPen && inArgs && stats.samples < 8) {
    ++stats.samples;
    emitf("    pen: canonical [%s]  viewport [%s]  pressure %s",
          readProp(inArgs, kOfxInteractPropPenPosition, kDbl).c_str(),
          readProp(inArgs, kOfxInteractPropPenViewportPosition, kInt).c_str(),
          readProp(inArgs, kOfxInteractPropPenPressure, kDbl).c_str());
    emitf("    pixel scale [%s]  render scale [%s]  time %s",
          readProp(inArgs, kOfxInteractPropPixelScale, kDbl).c_str(),
          readProp(inArgs, kOfxImageEffectPropRenderScale, kDbl).c_str(),
          readProp(inArgs, kOfxPropTime, kDbl).c_str());
  }

  // Pixel scale is how an overlay converts a screen-space grab radius into canonical
  // units. Flame reports [1,1] on pen events regardless of viewer zoom, so check whether
  // the draw action reports something more useful.
  if (std::strcmp(action, kOfxInteractActionDraw) == 0 && inArgs && stats.samples < 3) {
    ++stats.samples;
    emitf("    draw: pixel scale [%s]  render scale [%s]  background [%s]",
          readProp(inArgs, kOfxInteractPropPixelScale, kDbl).c_str(),
          readProp(inArgs, kOfxImageEffectPropRenderScale, kDbl).c_str(),
          readProp(inArgs, kOfxInteractPropBackgroundColour, kDbl).c_str());
  }

  // Every distinct key is worth recording: if keyboard events never arrive, tool shortcuts
  // and delete-vertex have to be done some other way.
  if (isKey && inArgs) {
    emitf("    key: %s sym %s string %s", action,
          readProp(inArgs, kOfxPropKeySym, kInt).c_str(),
          readProp(inArgs, kOfxPropKeyString, kStr).c_str());
  }

  if (std::strcmp(action, kOfxInteractActionDraw) == 0 && gDraw) {
    // Draws a deliberately asymmetric figure anchored at canonical (0,0), so the viewer
    // itself answers where the origin is and which way the axes run -- Flame reports
    // kOfxImageEffectHostPropNativeOrigin as present but empty, so it cannot be queried.
    // A long arm runs along +X, a short one along +Y, and a rectangle sits in the first
    // quadrant. DrawSuite avoids any OpenGL link dependency.
    void *ptr = nullptr;
    if (gProp && gProp->propGetPointer(inArgs, kOfxInteractPropDrawContext, 0, &ptr) == kOfxStatOK) {
      OfxDrawContextHandle ctx = (OfxDrawContextHandle)ptr;
      OfxRGBAColourF red = {1.0f, 0.2f, 0.2f, 1.0f};
      gDraw->setColour(ctx, &red);
      gDraw->setLineWidth(ctx, 2.0f);

      const OfxPointD axisX[2] = {{0.0, 0.0}, {300.0, 0.0}};
      const OfxPointD axisY[2] = {{0.0, 0.0}, {0.0, 100.0}};
      gDraw->draw(ctx, kOfxDrawPrimitiveLines, axisX, 2);
      gDraw->draw(ctx, kOfxDrawPrimitiveLines, axisY, 2);

      // Flame draws kOfxDrawPrimitiveRectangle filled rather than as an outline, which
      // matters because a spline editor wants outlines almost everywhere. Draw both, so
      // the difference stays visible in any host the probe is run in: a filled rectangle
      // on the left, and the same shape as a line loop on the right.
      OfxRGBAColourF green = {0.2f, 1.0f, 0.4f, 1.0f};
      gDraw->setColour(ctx, &green);
      const OfxPointD filled[2] = {{120.0, 40.0}, {260.0, 140.0}};
      gDraw->draw(ctx, kOfxDrawPrimitiveRectangle, filled, 2);

      OfxRGBAColourF cyan = {0.3f, 0.9f, 1.0f, 1.0f};
      gDraw->setColour(ctx, &cyan);
      const OfxPointD outline[4] = {
          {300.0, 40.0}, {440.0, 40.0}, {440.0, 140.0}, {300.0, 140.0}};
      gDraw->draw(ctx, kOfxDrawPrimitiveLineLoop, outline, 4);

      const OfxPointD originLabel = {12.0, 12.0};
      const OfxPointD xLabel = {310.0, 0.0};
      gDraw->drawText(ctx, "0,0", &originLabel, kOfxDrawTextAlignmentLeft);
      gDraw->drawText(ctx, "+X", &xLabel, kOfxDrawTextAlignmentLeft);
    }
  }
  return kOfxStatReplyDefault;
}

// ---------------------------------------------------------------------------
// Parameter type probing
// ---------------------------------------------------------------------------

std::string makeBigString() {
  // Deterministic, non-repeating enough that a host truncating or mangling the value is
  // detectable, and cheap to verify without storing a copy.
  std::string s;
  s.reserve(kBigStringBytes);
  const char *alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  for (size_t i = 0; i < kBigStringBytes; ++i)
    s.push_back(alphabet[(i * 31 + (i / 64)) % 64]);
  return s;
}

unsigned long checksum(const std::string &s) {
  unsigned long h = 5381;
  for (unsigned char c : s)
    h = ((h << 5) + h) ^ c;
  return h;
}

void probeParamTypes(OfxParamSetHandle paramSet) {
  section("Parameter types accepted by paramDefine");

  static const char *types[] = {
      kOfxParamTypeInteger,   kOfxParamTypeInteger2D, kOfxParamTypeInteger3D,
      kOfxParamTypeDouble,    kOfxParamTypeDouble2D,  kOfxParamTypeDouble3D,
      kOfxParamTypeRGB,       kOfxParamTypeRGBA,      kOfxParamTypeBoolean,
      kOfxParamTypeChoice,    kOfxParamTypeStrChoice, kOfxParamTypeString,
      kOfxParamTypeCustom,    kOfxParamTypeGroup,     kOfxParamTypePage,
      kOfxParamTypePushButton, kOfxParamTypeParametric,
  };

  for (const char *type : types) {
    std::string name = std::string("probe_") + type;
    OfxPropertySetHandle props = nullptr;
    OfxStatus st = gParam->paramDefine(paramSet, type, name.c_str(), &props);
    emitf("  %-24s %s", type, st == kOfxStatOK ? "accepted" : statusName(st));
    if (st == kOfxStatOK && props) {
      gProp->propSetString(props, kOfxPropLabel, 0, (std::string("probe ") + type).c_str());
      // A host that accepts the definition may still refuse individual properties; the
      // 2D/3D and choice cases are where hosts differ most, so exercise them.
      if (std::strcmp(type, kOfxParamTypeDouble2D) == 0)
        gProp->propSetString(props, kOfxParamPropDoubleType, 0, kOfxParamDoubleTypeXYAbsolute);
      if (std::strcmp(type, kOfxParamTypeChoice) == 0) {
        gProp->propSetString(props, kOfxParamPropChoiceOption, 0, "first");
        gProp->propSetString(props, kOfxParamPropChoiceOption, 1, "second");
      }
    }
  }
}

void defineProbeControls(OfxParamSetHandle paramSet) {
  struct Button {
    const char *name;
    const char *label;
    const char *hint;
  };
  static const Button buttons[] = {
      {kParamRunProbe, "Run Probe",
       "Write host capabilities, out-of-render frame access results and interact status to the report file."},
      {kParamWriteBigString, "Write 1 MB String",
       "Fill the hidden string parameter with a 1 MB payload. Save the setup, reload it, then press Verify."},
      {kParamVerifyBigString, "Verify 1 MB String",
       "Check the hidden string parameter still holds the exact 1 MB payload."},
  };

  for (const Button &b : buttons) {
    OfxPropertySetHandle props = nullptr;
    if (gParam->paramDefine(paramSet, kOfxParamTypePushButton, b.name, &props) == kOfxStatOK) {
      gProp->propSetString(props, kOfxPropLabel, 0, b.label);
      gProp->propSetString(props, kOfxParamPropHint, 0, b.hint);
    }
  }

  OfxPropertySetHandle props = nullptr;
  if (gParam->paramDefine(paramSet, kOfxParamTypeString, kParamBigString, &props) == kOfxStatOK) {
    gProp->propSetString(props, kOfxPropLabel, 0, "Payload");
    gProp->propSetString(props, kOfxParamPropStringMode, 0, kOfxParamStringIsMultiLine);
    gProp->propSetInt(props, kOfxParamPropSecret, 0, 1);
    gProp->propSetInt(props, kOfxParamPropAnimates, 0, 0);
    gProp->propSetString(props, kOfxParamPropDefault, 0, "");
  }
}

// ---------------------------------------------------------------------------
// Runtime probes, driven from the buttons
// ---------------------------------------------------------------------------

void probeOutOfRenderFrameAccess(OfxImageEffectHandle instance, double time) {
  section("Out-of-render frame access (clipGetImage outside kOfxImageEffectActionRender)");
  emit("  A working fast path here means the editor can pull frames on demand. If every");
  emit("  offset fails, the frame cache must be filled by a pass-through render instead.");

  OfxImageClipHandle clip = nullptr;
  OfxStatus st = gEffect->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &clip, nullptr);
  if (st != kOfxStatOK || !clip) {
    emitf("  clipGetHandle failed: %s", statusName(st));
    return;
  }

  const double offsets[] = {0.0, -1.0, 1.0, -5.0, 5.0, -24.0, 24.0};
  for (double dt : offsets) {
    OfxPropertySetHandle image = nullptr;
    st = gEffect->clipGetImage(clip, time + dt, nullptr, &image);
    if (st == kOfxStatOK && image) {
      std::string bounds = readProp(image, kOfxImagePropBounds, kInt);
      emitf("  t%+.0f: OK  bounds [%s]", dt, bounds.c_str());
      gEffect->clipReleaseImage(image);
    } else {
      emitf("  t%+.0f: %s", dt, statusName(st));
    }
  }
}

void probeInstanceProperties(OfxImageEffectHandle instance) {
  section("Effect instance properties");

  OfxPropertySetHandle props = nullptr;
  if (gEffect->getPropertySet(instance, &props) != kOfxStatOK || !props) {
    emit("  getPropertySet failed");
    return;
  }

  static const PropSpec specs[] = {
      {kOfxImageEffectPropContext, kStr},
      {kOfxImageEffectPropProjectSize, kDbl},
      {kOfxImageEffectPropProjectExtent, kDbl},
      {kOfxImageEffectPropProjectOffset, kDbl},
      {kOfxImageEffectPropProjectPixelAspectRatio, kDbl},
      {kOfxImageEffectInstancePropEffectDuration, kDbl},
      {kOfxImageEffectPropFrameRate, kDbl},
      {kOfxPropIsInteractive, kInt},
  };
  dumpProps(props, specs, sizeof(specs) / sizeof(specs[0]));

  OfxImageClipHandle clip = nullptr;
  if (gEffect->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &clip, nullptr) == kOfxStatOK) {
    OfxPropertySetHandle clipProps = nullptr;
    if (gEffect->clipGetPropertySet(clip, &clipProps) == kOfxStatOK && clipProps) {
      emit("  -- Source clip --");
      static const PropSpec clipSpecs[] = {
          {kOfxImageEffectPropPixelDepth, kStr},
          {kOfxImageEffectPropComponents, kStr},
          {kOfxImageEffectPropPreMultiplication, kStr},
          {kOfxImagePropPixelAspectRatio, kDbl},
          {kOfxImageEffectPropFrameRate, kDbl},
          {kOfxImageEffectPropFrameRange, kDbl},
          {kOfxImageClipPropConnected, kInt},
          {kOfxImageClipPropContinuousSamples, kInt},
      };
      dumpProps(clipProps, clipSpecs, sizeof(clipSpecs) / sizeof(clipSpecs[0]));
    }
  }
}

void reportOverlayStatus() {
  section("Overlay interact");
  emitf("  Host reports %s = %d", kOfxImageEffectPropSupportsOverlays, gHostSupportsOverlays ? 1 : 0);
  emitf("  Overlay entry point registered at describe time: %s", gOverlayRegistered ? "yes" : "no");

  if (gInteractStats.empty()) {
    emit("  No interact actions received at all -- the host never created the overlay, so");
    emit("  on-canvas editing is not viable here.");
    return;
  }

  emit("  Actions received (count is every event, not just the sampled ones):");
  for (const InteractStats &s : gInteractStats)
    emitf("    %-40s %ld", s.action.c_str(), s.count);

  // Absence of these is what decides whether an overlay can carry a real editor, so call
  // them out by name instead of leaving it to the reader to notice what is missing.
  static const char *required[] = {kOfxInteractActionDraw,    kOfxInteractActionPenDown,
                                   kOfxInteractActionPenUp,   kOfxInteractActionPenMotion,
                                   kOfxInteractActionKeyDown, kOfxInteractActionKeyUp};
  emit("  Editing capability check:");
  for (const char *action : required) {
    long count = 0;
    for (const InteractStats &s : gInteractStats)
      if (s.action == action)
        count = s.count;
    emitf("    %-40s %s", action, count > 0 ? "received" : "NEVER RECEIVED");
  }
  emit("  If key actions were never received, exercise the viewer with the pen and the");
  emit("  keyboard before trusting that result -- then treat it as final.");
}

void writeBigString(OfxParamSetHandle paramSet) {
  section("String parameter persistence: write");

  OfxParamHandle param = nullptr;
  OfxStatus st = gParam->paramGetHandle(paramSet, kParamBigString, &param, nullptr);
  if (st != kOfxStatOK || !param) {
    emitf("  paramGetHandle failed: %s", statusName(st));
    return;
  }

  const std::string payload = makeBigString();
  st = gParam->paramSetValue(param, payload.c_str());
  emitf("  paramSetValue of %zu bytes: %s", payload.size(), statusName(st));
  emitf("  expected checksum: %lu", checksum(payload));

  char *readBack = nullptr;
  if (gParam->paramGetValue(param, &readBack) == kOfxStatOK && readBack) {
    const std::string got(readBack);
    emitf("  immediate read back: %zu bytes, checksum %lu, %s", got.size(), checksum(got),
          got == payload ? "IDENTICAL" : "DIFFERENT");
  } else {
    emit("  immediate read back failed");
  }
  emit("  Now save the setup, reload it, and press Verify.");
}

void verifyBigString(OfxParamSetHandle paramSet) {
  section("String parameter persistence: verify");

  OfxParamHandle param = nullptr;
  OfxStatus st = gParam->paramGetHandle(paramSet, kParamBigString, &param, nullptr);
  if (st != kOfxStatOK || !param) {
    emitf("  paramGetHandle failed: %s", statusName(st));
    return;
  }

  char *readBack = nullptr;
  st = gParam->paramGetValue(param, &readBack);
  if (st != kOfxStatOK || !readBack) {
    emitf("  paramGetValue failed: %s", statusName(st));
    return;
  }

  const std::string got(readBack);
  const std::string expected = makeBigString();
  emitf("  got %zu bytes (expected %zu), checksum %lu (expected %lu)", got.size(), expected.size(),
        checksum(got), checksum(expected));
  if (got == expected)
    emit("  RESULT: payload survived intact -- the spline document can live in a string param.");
  else if (got.empty())
    emit("  RESULT: payload is empty -- the host did not persist it at all.");
  else
    emit("  RESULT: payload was altered or truncated -- document must be stored in a side file.");
}

void showMessage(OfxImageEffectHandle instance, const char *text) {
  if (gMessage)
    gMessage->message(instance, kOfxMessageMessage, nullptr, "%s", text);
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

OfxStatus onLoad() {
  emit("");
  emit("=========================================================================");
  {
    const std::time_t now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    emitf("White Water host probe -- session started %s (pid %d)", stamp, (int)getpid());
  }
  emitf("Report file: %s", reportPath().c_str());
  emitf("Host used OfxSetHost entry point: %s", gHostCalledOfxSetHost ? "yes" : "no");
  emit("=========================================================================");

  if (!gHost)
    return kOfxStatErrBadHandle;

  probeSuites();
  if (!gEffect || !gProp || !gParam) {
    emit("FATAL: a mandatory suite is missing, the probe cannot continue.");
    return kOfxStatErrMissingHostFeature;
  }
  probeHostProperties();
  return kOfxStatOK;
}

OfxStatus onDescribe(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffect->getPropertySet(effect, &props);

  gProp->propSetString(props, kOfxPropLabel, 0, "White Water Host Probe");
  gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "White Water");
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1, kOfxImageEffectContextGeneral);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthShort);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 2, kOfxBitDepthByte);
  gProp->propSetInt(props, kOfxImageEffectPluginPropSingleInstance, 0, 0);
  gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0, kOfxImageEffectRenderFullySafe);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsMultiResolution, 0, 1);
  gProp->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 1);

  // Registering an overlay on a host that claims not to support them is the interesting
  // experiment, but it is also the one that could destabilise a host, so it is opt-in.
  const bool force = std::getenv("WHITEWATER_PROBE_FORCE_OVERLAY") != nullptr;
  if (gHostSupportsOverlays || force) {
    const char *entry = gDraw ? kOfxImageEffectPluginPropOverlayInteractV2
                              : kOfxImageEffectPluginPropOverlayInteractV1;
    OfxStatus st = gProp->propSetPointer(props, entry, 0, (void *)(size_t)overlayMain);
    gOverlayRegistered = (st == kOfxStatOK);
    emitf("Registered overlay entry point via %s: %s%s", entry, statusName(st),
          force && !gHostSupportsOverlays ? " (forced by WHITEWATER_PROBE_FORCE_OVERLAY)" : "");
  } else {
    emit("Host does not advertise overlay support; overlay not registered.");
    emit("Set WHITEWATER_PROBE_FORCE_OVERLAY=1 to register it anyway and see what happens.");
  }
  return kOfxStatOK;
}

OfxStatus onDescribeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  const std::string context = readProp(inArgs, kOfxImageEffectPropContext, kStr);
  section(("Describe in context " + context).c_str());

  OfxPropertySetHandle clipProps = nullptr;
  gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &clipProps);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
  gProp->propSetInt(clipProps, kOfxImageEffectPropSupportsTiles, 0, 0);

  gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &clipProps);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
  gProp->propSetInt(clipProps, kOfxImageEffectPropSupportsTiles, 0, 0);

  OfxParamSetHandle paramSet = nullptr;
  gEffect->getParamSet(effect, &paramSet);
  defineProbeControls(paramSet);
  probeParamTypes(paramSet);
  return kOfxStatOK;
}

// Copies the source to the output untouched, so the probe is harmless in a real comp, and
// logs the render arguments once per session.
OfxStatus onRender(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs) {
  double time = 0.0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);

  if (!gRenderLogged) {
    gRenderLogged = true;
    section("First render");
    static const PropSpec specs[] = {
        {kOfxPropTime, kDbl},
        {kOfxImageEffectPropFieldToRender, kStr},
        {kOfxImageEffectPropRenderWindow, kInt},
        {kOfxImageEffectPropRenderScale, kDbl},
        {kOfxImageEffectPropSequentialRenderStatus, kInt},
        {kOfxImageEffectPropInteractiveRenderStatus, kInt},
        {kOfxImageEffectPropRenderQualityDraft, kInt},
        {kOfxImageEffectPropOpenGLEnabled, kInt},
        {kOfxImageEffectPropCudaEnabled, kInt},
        {kOfxImageEffectPropMetalEnabled, kInt},
        {kOfxImageEffectPropOpenCLEnabled, kInt},
    };
    dumpProps(inArgs, specs, sizeof(specs) / sizeof(specs[0]));
    probeInstanceProperties(instance);
  }

  OfxImageClipHandle sourceClip = nullptr, outputClip = nullptr;
  gEffect->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &sourceClip, nullptr);
  gEffect->clipGetHandle(instance, kOfxImageEffectOutputClipName, &outputClip, nullptr);

  OfxPropertySetHandle sourceImage = nullptr, outputImage = nullptr;
  if (sourceClip)
    gEffect->clipGetImage(sourceClip, time, nullptr, &sourceImage);
  if (outputClip)
    gEffect->clipGetImage(outputClip, time, nullptr, &outputImage);

  OfxStatus result = kOfxStatOK;
  if (!outputImage) {
    result = kOfxStatFailed;
  } else {
    void *outData = nullptr;
    int outRowBytes = 0, outBounds[4] = {0, 0, 0, 0};
    gProp->propGetPointer(outputImage, kOfxImagePropData, 0, &outData);
    gProp->propGetInt(outputImage, kOfxImagePropRowBytes, 0, &outRowBytes);
    for (int i = 0; i < 4; ++i)
      gProp->propGetInt(outputImage, kOfxImagePropBounds, i, &outBounds[i]);

    char *depth = nullptr, *components = nullptr;
    gProp->propGetString(outputImage, kOfxImageEffectPropPixelDepth, 0, &depth);
    gProp->propGetString(outputImage, kOfxImageEffectPropComponents, 0, &components);

    int bytesPerComponent = 4;
    if (depth && std::strcmp(depth, kOfxBitDepthByte) == 0)
      bytesPerComponent = 1;
    else if (depth && std::strcmp(depth, kOfxBitDepthShort) == 0)
      bytesPerComponent = 2;

    int componentCount = 4;
    if (components && std::strcmp(components, kOfxImageComponentAlpha) == 0)
      componentCount = 1;
    else if (components && std::strcmp(components, kOfxImageComponentRGB) == 0)
      componentCount = 3;

    const int pixelBytes = bytesPerComponent * componentCount;
    const int outWidth = outBounds[2] - outBounds[0];

    void *srcData = nullptr;
    int srcRowBytes = 0, srcBounds[4] = {0, 0, 0, 0};
    if (sourceImage) {
      gProp->propGetPointer(sourceImage, kOfxImagePropData, 0, &srcData);
      gProp->propGetInt(sourceImage, kOfxImagePropRowBytes, 0, &srcRowBytes);
      for (int i = 0; i < 4; ++i)
        gProp->propGetInt(sourceImage, kOfxImagePropBounds, i, &srcBounds[i]);
    }

    for (int y = outBounds[1]; y < outBounds[3]; ++y) {
      char *dstRow = (char *)outData + (size_t)(y - outBounds[1]) * outRowBytes;
      const bool srcHasRow = srcData && y >= srcBounds[1] && y < srcBounds[3];
      if (!srcHasRow) {
        std::memset(dstRow, 0, (size_t)outWidth * pixelBytes);
        continue;
      }
      const char *srcRow = (const char *)srcData + (size_t)(y - srcBounds[1]) * srcRowBytes;
      for (int x = outBounds[0]; x < outBounds[2]; ++x) {
        char *dstPixel = dstRow + (size_t)(x - outBounds[0]) * pixelBytes;
        if (x >= srcBounds[0] && x < srcBounds[2])
          std::memcpy(dstPixel, srcRow + (size_t)(x - srcBounds[0]) * pixelBytes, pixelBytes);
        else
          std::memset(dstPixel, 0, pixelBytes);
      }
    }
  }

  if (sourceImage)
    gEffect->clipReleaseImage(sourceImage);
  if (outputImage)
    gEffect->clipReleaseImage(outputImage);
  return result;
}

OfxStatus onInstanceChanged(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs) {
  char *name = nullptr;
  gProp->propGetString(inArgs, kOfxPropName, 0, &name);
  if (!name)
    return kOfxStatReplyDefault;

  double time = 0.0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);

  OfxParamSetHandle paramSet = nullptr;
  gEffect->getParamSet(instance, &paramSet);

  if (std::strcmp(name, kParamRunProbe) == 0) {
    probeHostProperties();
    probeInstanceProperties(instance);
    probeOutOfRenderFrameAccess(instance, time);
    reportOverlayStatus();
    emit("");
    emit("Probe complete.");
    showMessage(instance, ("White Water probe written to:\n" + reportPath()).c_str());
    return kOfxStatOK;
  }
  if (std::strcmp(name, kParamWriteBigString) == 0) {
    writeBigString(paramSet);
    showMessage(instance, "1 MB payload written. Save the setup, reload it, then press Verify.");
    return kOfxStatOK;
  }
  if (std::strcmp(name, kParamVerifyBigString) == 0) {
    verifyBigString(paramSet);
    showMessage(instance, ("Verification result written to:\n" + reportPath()).c_str());
    return kOfxStatOK;
  }
  return kOfxStatReplyDefault;
}

// ---------------------------------------------------------------------------
// Plugin entry points
// ---------------------------------------------------------------------------

void setHostFunc(OfxHost *host) { gHost = host; }

OfxStatus pluginMain(const char *action, const void *handle, OfxPropertySetHandle inArgs,
                     OfxPropertySetHandle /*outArgs*/) {
  try {
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;

    if (std::strcmp(action, kOfxActionLoad) == 0)
      return onLoad();
    if (std::strcmp(action, kOfxActionDescribe) == 0)
      return onDescribe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
      return onDescribeInContext(effect, inArgs);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0) {
      emit("Action: create instance");
      return kOfxStatOK;
    }
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0) {
      emit("Action: destroy instance");
      return kOfxStatOK;
    }
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
      return onRender(effect, inArgs);
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0)
      return onInstanceChanged(effect, inArgs);
    if (std::strcmp(action, kOfxActionUnload) == 0) {
      emit("Action: unload");
      return kOfxStatOK;
    }
  } catch (const std::exception &e) {
    emitf("EXCEPTION in action %s: %s", action, e.what());
    return kOfxStatFailed;
  } catch (...) {
    emitf("UNKNOWN EXCEPTION in action %s", action);
    return kOfxStatFailed;
  }
  return kOfxStatReplyDefault;
}

OfxPlugin gPlugin = {
    kOfxImageEffectPluginApi,
    1,
    "com.mtifilm.whitewater.hostprobe",
    1,
    0,
    setHostFunc,
    pluginMain,
};

} // namespace

// ofxCore.h defines OfxExport as a bare `extern` on Unix, which is not enough when the
// plugin is compiled with -fvisibility=hidden: the entry points end up absent from the
// dynamic symbol table and the host silently sees no plugin at all. Force default
// visibility on exactly these three symbols.
#define WHITEWATER_OFX_EXPORT extern "C" __attribute__((visibility("default")))

WHITEWATER_OFX_EXPORT int OfxGetNumberOfPlugins(void) { return 1; }

WHITEWATER_OFX_EXPORT OfxPlugin *OfxGetPlugin(int nth) { return nth == 0 ? &gPlugin : nullptr; }

// Optional since OFX 1.4. Hosts that use it hand over the host struct before
// OfxGetNumberOfPlugins; hosts that do not will call setHostFunc instead.
WHITEWATER_OFX_EXPORT OfxStatus OfxSetHost(const OfxHost *host) {
  gHost = const_cast<OfxHost *>(host);
  gHostCalledOfxSetHost = true;
  return kOfxStatOK;
}
