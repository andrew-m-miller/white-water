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

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

// dlfcn is libc, and the link-map walk is libc too (glibc's link.h on Linux, dyld's own
// API on macOS). Neither adds a dependency, which is the property this file exists to
// preserve: nothing between the probe and the host.
#include <dlfcn.h>
#ifdef __linux__
#include <link.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

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

// --- Phase 0 state -----------------------------------------------------------------
//
// The name of the optional second input. Arbitrary in the General context, and short
// because Flame truncates labels; it appears in the node's socket list, so it should read
// as what it is.
const char *const kProbeInsertClipName = "Insert";

bool gSecondClipDefined = false;

// Item 2: whether clipGetImage at a distant time works from *inside* a render, on a host
// thread, while the host holds whatever a render holds. Done once -- repeating it every
// tile would flood the log and slow a real comp to a crawl.
bool gInRenderPullDone = false;
long gInRenderPullsOK = 0;
long gInRenderPullsFailed = 0;

// Item 4: does declaring SupportsTiles=0 actually get us whole frames? Answered by
// comparing each render window against the output clip's region of definition. Tracked
// across every render rather than sampled once, because one full-frame render proves
// nothing if the next twenty are strips.
long gRenderCalls = 0;
long gPartialWindowRenders = 0;

// Item 5: is getFramesNeeded even called, and does declaring only {N} hold?
long gFramesNeededCalls = 0;

// --- 0C item 2: instance and process lifetime -------------------------------------
//
// Does a Batch node's FINAL/BACKGROUND render happen in the SAME process and SAME OFX
// instance as the foreground interactive session? That decides whether a RAM-only
// "Precache" has any production value at all. Two OFX instances may render concurrently
// (the plugin declares kOfxImageEffectRenderFullySafe), so this state is guarded by a
// mutex rather than assumed single-threaded across create/destroy/render actions.
std::mutex gInstanceLifetimeMutex;
std::atomic<long> gNextInstanceToken{1};
std::map<OfxImageEffectHandle, long> gInstanceTokens;
long gLiveInstanceCount = 0;
long gTotalInstancesSeen = 0;

// 0C item 3: every observation so far has renderScale == [1,1]; confirm whether Flame ever
// hands anything else (draft mode, proxy playback), at any PAR. Tracked across every render
// rather than sampled once, for the same reason item 4 is.
bool gRenderScaleSeen = false;
bool gRenderScaleEverNonUnit = false;
double gRenderScaleMinX = 1.0, gRenderScaleMaxX = 1.0;
double gRenderScaleMinY = 1.0, gRenderScaleMaxY = 1.0;

// 0C item 4: does Flame ever hand a negative kOfxImagePropRowBytes, or an image whose
// Bounds are a strict sub-region of the render window (a window into a larger allocation)?
// The vendored pass-through copy assumes neither; nothing has confirmed either happens here.
bool gNegativeRowBytesSeen = false;
bool gSubWindowImageSeen = false;

// 0C item 5: anamorphic tile re-check. Reuses gPartialWindowRenders/gRenderCalls above --
// this just records whether any PAR != 1 render occurred, and whether such renders were
// among the partial ones, to close out the coordinate-system scare noted in the tile check.
bool gAnamorphicRenderSeen = false;
double gAnamorphicParSeen = 0.0;
long gAnamorphicRenderCalls = 0;
long gAnamorphicPartialRenders = 0;

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
// Host address space: what is already loaded, and what we would collide with
// ---------------------------------------------------------------------------
//
// Phase 0 item 3. Flame 2026.2.1 was measured carrying libonnxruntime.so.1.22.0, CUDA 12.8
// and TensorRT 10.8 in its own process. That is good news -- ONNX Runtime demonstrably
// initialises inside Flame -- and a hazard, because a plugin that bundles its own ORT then
// has two copies in one address space.
//
// The hazard is only real if the host's copy is *globally visible* to us. A library the
// host dlopened with RTLD_LOCAL is not in our resolution scope and cannot capture our
// references; one loaded RTLD_GLOBAL, or pulled in as a DT_NEEDED of the executable, is.
// dlsym(RTLD_DEFAULT, ...) asks exactly that question from exactly the right place: inside
// a plugin the host has already loaded.
//
// This costs nothing and settles whether the expensive experiment -- bundling a real ORT
// and testing RTLD_DEEPBIND -- is needed at all.

// Symbols whose presence in the global scope would capture our references. OrtGetApiBase is
// the entire ONNX Runtime C entry point, so if it resolves here, ours would never be called.
struct GlobalSymbolProbe {
  const char *symbol;
  const char *belongsTo;
};

void probeGlobalSymbolScope() {
  section("Global symbol scope (would our bundled libraries be captured?)");
  emit("  dlsym(RTLD_DEFAULT, ...) from inside the loaded plugin. A hit means the host's");
  emit("  copy is visible to us and our references would bind to it, first-loaded-wins.");
  emit("  A miss means the host loaded it privately (RTLD_LOCAL) and there is no collision.");

  static const GlobalSymbolProbe symbols[] = {
      {"OrtGetApiBase", "ONNX Runtime"},
      {"OrtSessionOptionsAppendExecutionProvider_CUDA", "ONNX Runtime CUDA EP"},
      {"cudaRuntimeGetVersion", "CUDA runtime"},
      {"cudaMalloc", "CUDA runtime"},
      {"cudnnGetVersion", "cuDNN"},
      {"cublasCreate_v2", "cuBLAS"},
      {"createInferBuilder_INTERNAL", "TensorRT"},
      {"protobuf_shutdown", "protobuf (old ABI)"},
      {"_ZN6google8protobuf8internal14ShutdownEveryN", "protobuf (C++ ABI)"},
      {"TBB_runtime_interface_version", "TBB"},
  };

  bool anyCaptured = false;
  for (const GlobalSymbolProbe &s : symbols) {
    dlerror();
    void *addr = dlsym(RTLD_DEFAULT, s.symbol);
    if (addr) {
      anyCaptured = true;
      // Which object it came from is the actionable half -- a path under /opt/Autodesk
      // names the host as the owner, which is a different problem from a system library.
      const char *from = "<origin unknown>";
      Dl_info info;
      if (dladdr(addr, &info) && info.dli_fname)
        from = info.dli_fname;
      emitf("  VISIBLE  %-46s (%s)", s.symbol, s.belongsTo);
      emitf("           from %s", from);
    } else {
      emitf("  absent   %-46s (%s)", s.symbol, s.belongsTo);
    }
  }

  emit("");
  if (anyCaptured) {
    emit("  At least one symbol is globally visible. Bundling our own copy of that library");
    emit("  risks our calls binding to the host's while our state came from ours -- a");
    emit("  version-skew crash with no useful backtrace. Mitigations, in order: run");
    emit("  inference out of process; or dlopen ours RTLD_LOCAL|RTLD_DEEPBIND; or link it");
    emit("  statically with hidden visibility.");
  } else {
    emit("  Nothing captured. The host keeps these private, so a bundled copy resolves to");
    emit("  itself and in-process inference has no symbol problem to solve.");
  }
}

// Case-insensitive substring match against the libraries we might end up duplicating.
// Hand-rolled rather than reaching for <algorithm> or strcasestr: the first adds a header
// for one call, the second is not portable.
bool objectIsInteresting(const char *name) {
  if (!name || !name[0])
    return false;
  static const char *kNeedles[] = {"onnxruntime", "cuda",     "cublas", "cudnn",
                                   "nvinfer",     "nvonnx",   "tbb",    "protobuf",
                                   "torch",       "openvino"};
  for (const char *needle : kNeedles) {
    for (const char *p = name; *p; ++p) {
      const char *a = p;
      const char *b = needle;
      while (*a && *b && (*a | 0x20) == (*b | 0x20)) {
        ++a;
        ++b;
      }
      if (!*b)
        return true;
    }
  }
  return false;
}

// What is actually mapped into this process, filtered to things we might duplicate. This is
// the same information /proc/<pid>/maps gives, but obtainable without sudo because we are
// running inside the process rather than looking at it from outside.
void reportLoadedObjects() {
  section("Loaded objects of interest (from inside the host process)");

  int shown = 0;

#ifdef __linux__
  dl_iterate_phdr(
      [](struct dl_phdr_info *info, size_t, void *data) -> int {
        if (info->dlpi_name && objectIsInteresting(info->dlpi_name)) {
          emitf("  %s", info->dlpi_name);
          ++*static_cast<int *>(data);
        }
        return 0;
      },
      &shown);
#elif defined(__APPLE__)
  const uint32_t count = _dyld_image_count();
  for (uint32_t i = 0; i < count; ++i) {
    const char *name = _dyld_get_image_name(i);
    if (objectIsInteresting(name)) {
      emitf("  %s", name);
      ++shown;
    }
  }
#else
  emit("  <no link-map API on this platform>");
#endif

  if (shown == 0) {
    emit("  None. The host has no ML or CUDA runtime mapped at this moment -- note that a");
    emit("  library the host loads lazily would not appear until it has been used once.");
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

// Phase 0 item 2 -- the load-bearing one.
//
// warp-drive measured clipGetImage at arbitrary offsets from an instance-changed action,
// where the host is idle and waiting on us. Doing it from inside a render is a different
// question: we are on a host thread, mid-render, while the host holds whatever locks a
// render holds. White Water's whole on-demand flow chain assumes this works. If it does
// not, the design becomes a mandatory Analyze pass writing a disk cache.
//
// Deliberately conservative: one call per offset, image released immediately, no attempt to
// read pixels. If this is going to deadlock a host it should do so having done the least
// possible damage.
void probeInRenderFrameAccess(OfxImageEffectHandle instance, double time) {
  section("In-render frame access (clipGetImage DURING kOfxImageEffectActionRender)");
  emit("  This is the assumption the on-demand flow chain rests on. warp-drive only ever");
  emit("  measured the equivalent from an instance-changed action, which is a host at rest.");

  OfxImageClipHandle clip = nullptr;
  OfxStatus st =
      gEffect->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &clip, nullptr);
  if (st != kOfxStatOK || !clip) {
    emitf("  clipGetHandle failed: %s", statusName(st));
    return;
  }

  // Includes offsets past the likely ends of a clip on purpose: an out-of-range pull has to
  // fail cleanly rather than take the host with it, and the chain will ask for exactly that
  // whenever the reference frame sits near a boundary.
  const double offsets[] = {-1.0, 1.0, -10.0, 10.0, -100.0, 100.0};
  for (double dt : offsets) {
    OfxPropertySetHandle image = nullptr;
    emitf("  requesting t%+.0f ...", dt);
    st = gEffect->clipGetImage(clip, time + dt, nullptr, &image);
    if (st == kOfxStatOK && image) {
      ++gInRenderPullsOK;
      std::string bounds = readProp(image, kOfxImagePropBounds, kInt);
      emitf("  t%+.0f: OK  bounds [%s]", dt, bounds.c_str());
      gEffect->clipReleaseImage(image);
    } else {
      ++gInRenderPullsFailed;
      emitf("  t%+.0f: %s", dt, statusName(st));
    }
  }

  // An offset past the end of the clip returning an image rather than an error is a real
  // result and easy to misread as success. Say so where it is observed, not only in a
  // document nobody has open at the time.
  emit("  NOTE: check these against the clip's frame range. A host that clamps rather than");
  emit("  failing will return a valid image for a time outside the clip, so the chain must");
  emit("  consult kOfxImageEffectPropFrameRange itself and never infer range from failure.");
  emit("  Survived every pull without deadlocking. If this line is missing from the report,");
  emit("  the host hung or died on one of the requests above -- which is itself the answer.");
}

// Phase 0 item 1, the runtime half: what the host says about a second, optional clip and
// what it hands over when that clip is not connected.
void probeSecondClip(OfxImageEffectHandle instance, double time) {
  section("Optional second input clip");
  if (!gSecondClipDefined) {
    emit("  Not defined in this context -- open the probe in a General-context node.");
    return;
  }

  OfxImageClipHandle clip = nullptr;
  OfxStatus st = gEffect->clipGetHandle(instance, kProbeInsertClipName, &clip, nullptr);
  emitf("  clipGetHandle('%s'): %s", kProbeInsertClipName, statusName(st));
  if (st != kOfxStatOK || !clip)
    return;

  OfxPropertySetHandle clipProps = nullptr;
  if (gEffect->clipGetPropertySet(clip, &clipProps) == kOfxStatOK && clipProps) {
    static const PropSpec specs[] = {
        {kOfxImageClipPropConnected, kInt},
        {kOfxImageClipPropOptional, kInt},
        {kOfxImageEffectPropPixelDepth, kStr},
        {kOfxImageEffectPropComponents, kStr},
        {kOfxImageEffectPropPreMultiplication, kStr},
        {kOfxImagePropPixelAspectRatio, kDbl},
        {kOfxImageEffectPropFrameRange, kDbl},
    };
    dumpProps(clipProps, specs, sizeof(specs) / sizeof(specs[0]));
  }

  // The question that decides how the render path branches: what comes back when nobody
  // has plugged anything in. A clean failure is easy to handle; a black image that claims
  // success is a trap, because the comp would silently be over black.
  OfxPropertySetHandle image = nullptr;
  st = gEffect->clipGetImage(clip, time, nullptr, &image);
  if (st == kOfxStatOK && image) {
    emitf("  clipGetImage returned an image: bounds [%s], components %s",
          readProp(image, kOfxImagePropBounds, kInt).c_str(),
          readProp(image, kOfxImageEffectPropComponents, kStr).c_str());
    emit("  If this clip is NOT connected in the node, note that carefully: the host is");
    emit("  handing over an image for an unconnected optional input, so connectedness must");
    emit("  be read from the property, never inferred from clipGetImage succeeding.");
    gEffect->clipReleaseImage(image);
  } else {
    emitf("  clipGetImage: %s", statusName(st));
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

// The five Phase 0 questions, answered in one block at the end of the report.
//
// The detail above is what gets pasted into docs/host-notes.md, but a reader on the box
// needs to know whether to keep going or stop, and scrolling a few hundred lines to work
// that out is how a measurement session turns into an afternoon. Every line here restates
// something already logged; nothing is computed only here.
void reportPhaseZeroSummary() {
  section("PHASE 0 SUMMARY");

  emitf("  1. Second optional clip     : %s",
        gSecondClipDefined ? "defined -- see 'Optional second input clip' above"
                           : "NOT DEFINED (open the probe in a General-context node)");

  if (!gInRenderPullDone) {
    emit("  2. clipGetImage in render   : NOT ATTEMPTED (the node never rendered; view it"
         " in the viewer)");
  } else if (gInRenderPullsFailed == 0) {
    emitf("  2. clipGetImage in render   : WORKS -- %ld of %ld offsets returned an image,"
          " no deadlock. The on-demand flow chain is viable.",
          gInRenderPullsOK, gInRenderPullsOK + gInRenderPullsFailed);
  } else {
    emitf("  2. clipGetImage in render   : PARTIAL -- %ld ok, %ld failed. Read the"
          " per-offset results; which offsets failed decides whether this is a range"
          " limit or a real refusal.",
          gInRenderPullsOK, gInRenderPullsFailed);
  }

  emit("  3. ORT/CUDA symbol capture  : see 'Global symbol scope' above.");
  emit("       VISIBLE lines mean a bundled copy would be captured by the host's, so");
  emit("       inference goes out of process or through RTLD_DEEPBIND. All absent means");
  emit("       in-process bundling is clean.");

  if (gRenderCalls == 0) {
    emit("  4. Tiles (SupportsTiles=0)  : NO RENDERS YET");
  } else if (gPartialWindowRenders == 0) {
    emitf("  4. Tiles (SupportsTiles=0)  : honoured -- %ld render(s) so far, all full"
          " frame. Renders after this point are logged individually below.",
          gRenderCalls);
  } else {
    emitf("  4. Tiles (SupportsTiles=0)  : NOT HONOURED -- %ld of %ld renders were partial."
          " The render path needs tile assembly.",
          gPartialWindowRenders, gRenderCalls);
  }

  if (gFramesNeededCalls == 0) {
    emit("  5. getFramesNeeded          : NEVER CALLED. The host does not ask, so what we");
    emit("       declare cannot cause over-fetching -- and pulls in render are the only");
    emit("       way to get other frames regardless.");
  } else {
    emitf("  5. getFramesNeeded          : called %ld times; we declared only {N} each time."
          " Cross-check against whether item 2's pulls still succeeded.",
          gFramesNeededCalls);
  }

  emit("");
  emit("  Paste this report into docs/host-notes.md under a Measured heading, with the");
  emit("  host build and the date. Items 2 and 3 decide the architecture.");
}

// 0C items 3, 4, 5: render-time observations, gathered incidentally by every onRender()
// call above rather than by a dedicated probe step. Reported together in one section
// because none of the three has its own "run this" action -- they only need a render to
// have happened at least once, from the viewer or a real comp.
void reportRenderObservations() {
  section("0C render observations (items 3, 4, 5)");

  if (!gRenderScaleSeen) {
    emit("  3. Render scale             : NO RENDERS YET");
  } else if (!gRenderScaleEverNonUnit) {
    emitf("  3. Render scale             : always [1 1] across %ld render(s). No draft or"
          " proxy scale observed.",
          gRenderCalls);
  } else {
    emitf("  3. Render scale             : NON-UNIT seen -- x in [%.4f %.4f], y in [%.4f"
          " %.4f] across %ld render(s). See the per-render log above for which ones.",
          gRenderScaleMinX, gRenderScaleMaxX, gRenderScaleMinY, gRenderScaleMaxY,
          gRenderCalls);
  }

  emitf("  4. Negative rowBytes seen   : %s",
        gNegativeRowBytesSeen ? "YES -- see render log above" : "no");
  emitf("     Sub-window images seen   : %s",
        gSubWindowImageSeen ? "YES -- see render log above" : "no");

  if (!gAnamorphicRenderSeen) {
    emit("  5. Anamorphic tile re-check : no PAR != 1 render observed on this clip; the PAR");
    emit("       fix in the tile check above is unexercised here.");
  } else if (gAnamorphicPartialRenders == 0) {
    emitf("  5. Anamorphic tile re-check : PAR %.3f seen across %ld render(s), all full"
          " frame -- confirms the earlier tiled-looking result was the coordinate-system"
          " mismatch, not a host tiling behavior.",
          gAnamorphicParSeen, gAnamorphicRenderCalls);
  } else {
    emitf("  5. Anamorphic tile re-check : PAR %.3f seen across %ld render(s), %ld of them"
          " PARTIAL -- host tiled an anamorphic clip despite SupportsTiles=0.",
          gAnamorphicParSeen, gAnamorphicRenderCalls, gAnamorphicPartialRenders);
  }

  emit("");
  emit("  Paste this section into docs/host-notes.md alongside the Phase 0 summary above.");
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

// 0C item 2: instance and process lifetime.
//
// Assigns a monotonic token to every OFX instance so the report can tell "the same
// instance rendered again" from "a different instance (or process) appeared" -- exactly
// the distinction a duplicated Batch node, or a background render running in a second
// process, raises. Kept deliberately trivial: map operations here do not throw, but the
// lock is still scoped tightly and nothing here can itself fail an action.
OfxStatus onCreateInstance(OfxImageEffectHandle instance) {
  long token = 0;
  long live = 0;
  {
    std::lock_guard<std::mutex> lock(gInstanceLifetimeMutex);
    token = gNextInstanceToken.fetch_add(1);
    gInstanceTokens[instance] = token;
    ++gLiveInstanceCount;
    ++gTotalInstancesSeen;
    live = gLiveInstanceCount;
  }
  emitf("0C item 2: Action: create instance  token=%ld pid=%d live=%ld", token, (int)getpid(),
        live);
  return kOfxStatOK;
}

OfxStatus onDestroyInstance(OfxImageEffectHandle instance) {
  long token = -1;
  long live = 0;
  {
    std::lock_guard<std::mutex> lock(gInstanceLifetimeMutex);
    auto it = gInstanceTokens.find(instance);
    if (it != gInstanceTokens.end()) {
      token = it->second;
      gInstanceTokens.erase(it);
    }
    if (gLiveInstanceCount > 0)
      --gLiveInstanceCount;
    live = gLiveInstanceCount;
  }
  emitf("0C item 2: Action: destroy instance  token=%ld pid=%d live=%ld", token, (int)getpid(),
        live);
  return kOfxStatOK;
}

// 0C item 2, the key line. A Batch node's FOREGROUND (interactive) render and its
// FINAL/BACKGROUND render both reach this handler if the latter happens at all for this
// node type -- what decides the architecture is whether the pid and instance token agree
// between the two. kOfxImageEffectPropInteractiveRenderStatus=1 marks the foreground
// session; =0 together with kOfxImageEffectPropSequentialRenderStatus=1 is the shape of a
// final/background render. If that combination shows up under a different pid, that is
// the decisive finding: a RAM-only Precache in the foreground process would not be visible
// to it.
OfxStatus onBeginSequenceRender(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs) {
  long token = -1;
  {
    std::lock_guard<std::mutex> lock(gInstanceLifetimeMutex);
    auto it = gInstanceTokens.find(instance);
    if (it != gInstanceTokens.end())
      token = it->second;
  }

  double range[2] = {0.0, 0.0};
  gProp->propGetDouble(inArgs, kOfxImageEffectPropFrameRange, 0, &range[0]);
  gProp->propGetDouble(inArgs, kOfxImageEffectPropFrameRange, 1, &range[1]);
  const int interactive = propInt(inArgs, kOfxImageEffectPropInteractiveRenderStatus, -1);
  const int sequential = propInt(inArgs, kOfxImageEffectPropSequentialRenderStatus, -1);

  emitf("0C item 2: Action: begin sequence render  token=%ld pid=%d frameRange=[%.3f %.3f]"
        " interactive=%d sequential=%d",
        token, (int)getpid(), range[0], range[1], interactive, sequential);
  return kOfxStatOK;
}

OfxStatus onEndSequenceRender(OfxImageEffectHandle instance) {
  long token = -1;
  {
    std::lock_guard<std::mutex> lock(gInstanceLifetimeMutex);
    auto it = gInstanceTokens.find(instance);
    if (it != gInstanceTokens.end())
      token = it->second;
  }
  emitf("0C item 2: Action: end sequence render  token=%ld pid=%d", token, (int)getpid());
  return kOfxStatOK;
}

// 0C item 2 report, called from the Run Probe button. States what has been observed so
// far in THIS process and gives the reader the exact procedure to run in Flame, plus how
// to read the resulting log, to answer whether a RAM-only Precache has production value.
void reportInstanceLifetime() {
  section("0C item 2: instance and process lifetime");
  emitf("  Current pid: %d", (int)getpid());

  long totalSeen = 0, live = 0;
  {
    std::lock_guard<std::mutex> lock(gInstanceLifetimeMutex);
    totalSeen = gTotalInstancesSeen;
    live = gLiveInstanceCount;
  }
  emitf("  Instances seen in this process so far: %ld total, %ld currently live", totalSeen,
        live);

  emit("");
  emit("  Procedure to run in Flame (this is 0C item 2 -- ST convention is item 1, tracked");
  emit("  separately):");
  emit("    1. Put this probe on a Batch node. Save the setup, then reload it.");
  emit("    2. Switch the Batch schematic away from the node and back to it.");
  emit("    3. DUPLICATE the node.");
  emit("    4. Render the node in the FOREGROUND / interactive path (e.g. step the viewer).");
  emit("    5. Render the node with a BACKGROUND or FINAL render.");
  emit("    6. Quit Flame, reopen it, and repeat steps 4-5.");
  emit("");
  emit("  How to read it: every 'create instance' / 'destroy instance' line carries a");
  emit("  token and this process's pid. Every 'begin sequence render' line carries the");
  emit("  same token, this pid, the frame range, and interactive/sequential status.");
  emit("");
  emit("  SAME pid on the foreground render and on the final/background render: a");
  emit("  RAM-only Precache survives into the final render, so it is production-viable.");
  emit("");
  emit("  DIFFERENT pid on the final/background render: it runs in a separate process,");
  emit("  so a RAM-only Precache built in the foreground process is invisible to it and");
  emit("  has no production value.");
  emit("");
  emit("  NO second 'begin sequence render' observed at all for a background render:");
  emit("  this has the same practical effect as 'different process' -- no precache");
  emit("  benefit -- but a DIFFERENT meaning, and the two must not be conflated in");
  emit("  docs/host-notes.md. Read it as EITHER 'background/final render does not apply");
  emit("  to this node type or context' OR 'it was never actually exercised in this");
  emit("  session' -- and say in the report which of the two actually happened (i.e.");
  emit("  whether a background render was attempted at all).");
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
  // Before anything of ours is in the process, record what already is. Doing this at load
  // rather than on the button means the answer is not contaminated by our own dependencies.
  probeGlobalSymbolScope();
  reportLoadedObjects();
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
  // Two reads of the same property on purpose. readProp renders a value for a human and
  // quotes strings to do it, so its result is a display string and must never be compared
  // against anything -- doing exactly that is what made the first run report "context is
  // not General" inside a section headed 'Describe in context "OfxImageEffectContextGeneral"'
  // and silently skip the second clip.
  const std::string contextForDisplay = readProp(inArgs, kOfxImageEffectPropContext, kStr);
  section(("Describe in context " + contextForDisplay).c_str());

  char *contextValue = nullptr;
  gProp->propGetString(inArgs, kOfxImageEffectPropContext, 0, &contextValue);
  const std::string context = contextValue ? contextValue : "";

  OfxPropertySetHandle clipProps = nullptr;
  gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &clipProps);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
  gProp->propSetInt(clipProps, kOfxImageEffectPropSupportsTiles, 0, 0);

  // Phase 0 item 1: a second, optional input. White Water needs one for the insert layer,
  // and nothing has ever confirmed that Flame shows it, how the sockets present, or what a
  // disconnected optional clip returns. Only the General context may have extra inputs --
  // Filter is defined as exactly one source -- so defining it there would be an error, and
  // an error out of describeInContext means Flame shows no plugin at all.
  gSecondClipDefined = (context == kOfxImageEffectContextGeneral);
  if (gSecondClipDefined) {
    OfxPropertySetHandle insertProps = nullptr;
    OfxStatus st = gEffect->clipDefine(effect, kProbeInsertClipName, &insertProps);
    if (st == kOfxStatOK && insertProps) {
      gProp->propSetString(insertProps, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);
      gProp->propSetString(insertProps, kOfxImageEffectPropSupportedComponents, 1, kOfxImageComponentAlpha);
      gProp->propSetInt(insertProps, kOfxImageEffectPropSupportsTiles, 0, 0);
      // The property that makes it legal to leave it unconnected. Set non-fatally: a host
      // that refuses it must not take the whole plugin down with it.
      OfxStatus opt = gProp->propSetInt(insertProps, kOfxImageClipPropOptional, 0, 1);
      emitf("  Defined optional second clip '%s': clipDefine %s, %s %s", kProbeInsertClipName,
            statusName(st), kOfxImageClipPropOptional, statusName(opt));
    } else {
      gSecondClipDefined = false;
      emitf("  clipDefine('%s') failed: %s", kProbeInsertClipName, statusName(st));
    }
  } else {
    emit("  Context is not General; second input clip not defined (Filter takes one source).");
  }

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

// Phase 0 item 5.
//
// White Water plans to declare only {N} here while pulling {R..N} with clipGetImage during
// render, which is out of contract. The alternative -- declaring the true range -- risks
// Flame materialising hundreds of upstream frames to produce one output frame, which would
// be far worse. This handler declares the honest minimum so that item 2's pulls are being
// tested under exactly the conditions the real plugin would create.
//
// It also answers a prior question: whether Flame calls this action at all. A host that
// never calls it cannot be over-fetching because of what we declare here.
OfxStatus onGetFramesNeeded(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs,
                            OfxPropertySetHandle outArgs) {
  double time = 0.0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);

  ++gFramesNeededCalls;
  if (gFramesNeededCalls <= 3)
    emitf("Action: getFramesNeeded at t=%.3f (call %ld) -- declaring only {N}", time,
          gFramesNeededCalls);

  // The property is "OfxImageClipPropFrameRange_" post-pended with the clip's name, one per
  // input clip, as a pair of doubles per requested range.
  const double range[2] = {time, time};
  std::string prop = std::string("OfxImageClipPropFrameRange_") + kOfxImageEffectSimpleSourceClipName;
  OfxStatus st = gProp->propSetDoubleN(outArgs, prop.c_str(), 2, range);
  if (gFramesNeededCalls <= 3)
    emitf("  %s = [%.0f %.0f]: %s", prop.c_str(), range[0], range[1], statusName(st));

  if (gSecondClipDefined) {
    std::string insertProp = std::string("OfxImageClipPropFrameRange_") + kProbeInsertClipName;
    OfxStatus ist =
        gProp->propSetDoubleN(outArgs, insertProp.c_str(), 2, range);
    if (gFramesNeededCalls <= 3)
      emitf("  %s = [%.0f %.0f]: %s", insertProp.c_str(), range[0], range[1], statusName(ist));
  }

  (void)effect;
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
    probeSecondClip(instance, time);
  }

  // Item 4: tiles. We declared SupportsTiles=0; the host is then obliged to give us whole
  // frames. Compare every render window against the output region of definition rather than
  // trusting the first one -- a single full-frame render proves nothing if the next twenty
  // are strips, which is exactly what a progressive viewer refresh looks like.
  ++gRenderCalls;
  // Hoisted out of the block below (rather than left local to it, as it was before the 0C
  // additions) so 0C item 4's sub-window check can compare against it further down in this
  // function, after the block that fills it has closed.
  int window[4] = {0, 0, 0, 0};
  {
    for (int i = 0; i < 4; ++i)
      gProp->propGetInt(inArgs, kOfxImageEffectPropRenderWindow, i, &window[i]);

    double renderScale[2] = {1.0, 1.0};
    gProp->propGetDouble(inArgs, kOfxImageEffectPropRenderScale, 0, &renderScale[0]);
    gProp->propGetDouble(inArgs, kOfxImageEffectPropRenderScale, 1, &renderScale[1]);

    // 0C item 3: render scale. Track min/max per axis and whether it was ever non-unit,
    // across every render -- one [1,1] render proves nothing about draft mode or proxy
    // playback if the next one is scaled.
    {
      const double kEps = 1e-9;
      if (!gRenderScaleSeen) {
        gRenderScaleSeen = true;
        gRenderScaleMinX = gRenderScaleMaxX = renderScale[0];
        gRenderScaleMinY = gRenderScaleMaxY = renderScale[1];
      } else {
        if (renderScale[0] < gRenderScaleMinX) gRenderScaleMinX = renderScale[0];
        if (renderScale[0] > gRenderScaleMaxX) gRenderScaleMaxX = renderScale[0];
        if (renderScale[1] < gRenderScaleMinY) gRenderScaleMinY = renderScale[1];
        if (renderScale[1] > gRenderScaleMaxY) gRenderScaleMaxY = renderScale[1];
      }
      const double dx = renderScale[0] < 1.0 ? 1.0 - renderScale[0] : renderScale[0] - 1.0;
      const double dy = renderScale[1] < 1.0 ? 1.0 - renderScale[1] : renderScale[1] - 1.0;
      if (dx > kEps || dy > kEps)
        gRenderScaleEverNonUnit = true;
    }

    OfxImageClipHandle out = nullptr;
    OfxRectD rod = {0, 0, 0, 0};
    if (gEffect->clipGetHandle(instance, kOfxImageEffectOutputClipName, &out, nullptr) ==
            kOfxStatOK &&
        out && gEffect->clipGetRegionOfDefinition(out, time, &rod) == kOfxStatOK) {
      // These two rectangles are NOT in the same coordinate system, and comparing them
      // directly is wrong on any anamorphic clip. The render window is in PIXELS; the
      // region of definition comes back in CANONICAL (square-pixel) coordinates. They
      // coincide only when the pixel aspect ratio is 1, which is exactly why a PAR-1
      // measurement made this look correct and a PAR-2 one reported every render as
      // tiled. Measured 2026-08-20: a 4608x3164 PAR-2 source has a 9216x3164 RoD --
      // the same region, twice as wide in canonical units.
      //
      //   x_pixel = x_canonical * renderScale.x / par
      double par = 1.0;
      OfxPropertySetHandle outProps = nullptr;
      if (gEffect->clipGetPropertySet(out, &outProps) == kOfxStatOK && outProps)
        gProp->propGetDouble(outProps, kOfxImagePropPixelAspectRatio, 0, &par);
      if (!(par > 0.0))
        par = 1.0;

      const double rodPixelX1 = rod.x1 * renderScale[0] / par;
      const double rodPixelX2 = rod.x2 * renderScale[0] / par;
      const double rodPixelY1 = rod.y1 * renderScale[1];
      const double rodPixelY2 = rod.y2 * renderScale[1];

      // One pixel of slack: the conversion is a rounding boundary, not an exact integer
      // relationship, and a half-pixel disagreement is not a tile.
      const double kSlack = 1.0;
      const bool partial = window[0] > rodPixelX1 + kSlack || window[1] > rodPixelY1 + kSlack ||
                           window[2] < rodPixelX2 - kSlack || window[3] < rodPixelY2 - kSlack;
      if (partial)
        ++gPartialWindowRenders;

      // 0C item 5: anamorphic tile re-check. The PAR fix above already makes the comparison
      // correct; this just tallies, for reportRenderObservations(), whether any PAR != 1
      // render happened at all, and whether such renders were among the partial ones -- the
      // thing that closes out the old "PAR-2 looked tiled" scare as a coordinate-system bug,
      // not a host behavior.
      {
        const double dPar = par < 1.0 ? 1.0 - par : par - 1.0;
        if (dPar > 1e-6) {
          gAnamorphicRenderSeen = true;
          gAnamorphicParSeen = par;
          ++gAnamorphicRenderCalls;
          if (partial)
            ++gAnamorphicPartialRenders;
        }
      }

      if (gRenderCalls <= 8 || partial) {
        emitf("  render %ld: window [%d %d %d %d] rod(canonical) [%.0f %.0f %.0f %.0f]"
              " par %.3f -> rod(pixels) [%.0f %.0f %.0f %.0f] %s",
              gRenderCalls, window[0], window[1], window[2], window[3], rod.x1, rod.y1,
              rod.x2, rod.y2, par, rodPixelX1, rodPixelY1, rodPixelX2, rodPixelY2,
              partial ? "PARTIAL -- host tiled us despite SupportsTiles=0" : "full frame");
      }
    }
  }

  // Item 2, done once. Placed after the pass-through work below would be safer for the
  // host, but it is placed here on purpose: if this deadlocks we want to know it deadlocked
  // in the pull and not in the copy.
  if (!gInRenderPullDone) {
    gInRenderPullDone = true;
    probeInRenderFrameAccess(instance, time);
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

    // 0C item 4: negative rowBytes and sub-window images. Detection and reporting only --
    // the copy loop below is otherwise untouched. It assumes a non-negative stride and was
    // never written to handle anything else, so a negative stride is logged and the loop is
    // skipped for this render rather than let it walk off the allocation.
    const bool negativeRowBytes = outRowBytes < 0 || srcRowBytes < 0;
    if (negativeRowBytes && !gNegativeRowBytesSeen) {
      emitf("  0C item 4: NEGATIVE ROWBYTES -- out %d, src %d. Skipping the pass-through"
            " copy for this render; the loop was never written to handle a negative stride.",
            outRowBytes, srcRowBytes);
    }
    if (negativeRowBytes)
      gNegativeRowBytesSeen = true;

    if (!gSubWindowImageSeen) {
      // A sub-window image is one whose Bounds are a strict subset of the render window we
      // were asked to fill -- i.e. a pointer into a larger allocation, where Bounds/RowBytes
      // (not window) is what actually describes the accessible region.
      const bool outIsSubWindow = outBounds[0] > window[0] || outBounds[1] > window[1] ||
                                  outBounds[2] < window[2] || outBounds[3] < window[3];
      const bool srcIsSubWindow =
          sourceImage && (srcBounds[0] > window[0] || srcBounds[1] > window[1] ||
                          srcBounds[2] < window[2] || srcBounds[3] < window[3]);
      if (outIsSubWindow || srcIsSubWindow) {
        gSubWindowImageSeen = true;
        emitf("  0C item 4: SUB-WINDOW IMAGE -- out bounds [%d %d %d %d], src bounds"
              " [%d %d %d %d], render window [%d %d %d %d].",
              outBounds[0], outBounds[1], outBounds[2], outBounds[3], srcBounds[0],
              srcBounds[1], srcBounds[2], srcBounds[3], window[0], window[1], window[2],
              window[3]);
      }
    }

    if (!negativeRowBytes) {
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
    probeSecondClip(instance, time);
    probeOutOfRenderFrameAccess(instance, time);
    probeGlobalSymbolScope();
    reportLoadedObjects();
    reportOverlayStatus();
    reportPhaseZeroSummary();
    reportRenderObservations();
    reportInstanceLifetime();
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
                     OfxPropertySetHandle outArgs) {
  try {
    OfxImageEffectHandle effect = (OfxImageEffectHandle)handle;

    if (std::strcmp(action, kOfxActionLoad) == 0)
      return onLoad();
    if (std::strcmp(action, kOfxActionDescribe) == 0)
      return onDescribe(effect);
    if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0)
      return onDescribeInContext(effect, inArgs);
    if (std::strcmp(action, kOfxActionCreateInstance) == 0)
      return onCreateInstance(effect);
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0)
      return onDestroyInstance(effect);
    if (std::strcmp(action, kOfxImageEffectActionBeginSequenceRender) == 0)
      return onBeginSequenceRender(effect, inArgs);
    if (std::strcmp(action, kOfxImageEffectActionEndSequenceRender) == 0)
      return onEndSequenceRender(effect);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
      return onRender(effect, inArgs);
    if (std::strcmp(action, kOfxImageEffectActionGetFramesNeeded) == 0)
      return onGetFramesNeeded(effect, inArgs, outArgs);
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
