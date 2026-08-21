// White Water ONNX Runtime isolation probe.
//
// Answers the last open Phase 0 question, and only that one: can a plugin bundle its own
// ONNX Runtime and use it inside Flame, given that Flame already has ONNX Runtime 1.22.0
// loaded in the global symbol scope?
//
// Measured 2026-08-20 (docs/host-notes.md): OrtGetApiBase, the CUDA runtime, cuDNN, cuBLAS,
// TensorRT and TBB are all globally visible to a loaded plugin, from /opt/Autodesk/lib64.
// A bundled copy therefore risks having our calls bind to Flame's while our state came from
// ours -- a version-skew crash with no useful backtrace, on a machine with no debugger.
//
// ---------------------------------------------------------------------------
// Why this is a separate bundle
// ---------------------------------------------------------------------------
//
// A runtime that refuses to initialise, or that takes the host down when it tries, must not
// be able to destroy the capability probe's ability to report anything. The two are shipped
// as separate .ofx.bundles for that reason and share no code.
//
// ---------------------------------------------------------------------------
// Why dlopen and never link
// ---------------------------------------------------------------------------
//
// Linking ONNX Runtime would record it as a DT_NEEDED, which the loader resolves through the
// global scope -- binding us to Flame's copy before any of our code runs, and defeating the
// entire measurement. So this file includes only the C API header for its struct layouts,
// links nothing, and reaches every function through dlsym on a handle we opened ourselves.
//
// ---------------------------------------------------------------------------
// What is actually being tested, in three levels
// ---------------------------------------------------------------------------
//
// 1. RESOLUTION. Does dlsym on our handle return a different OrtGetApiBase than
//    dlsym(RTLD_DEFAULT, ...)? Cheap, and necessary but NOT sufficient.
//
// 2. INTERNAL CONSISTENCY. This is what level 1 misses and why RTLD_LOCAL alone is not
//    enough. RTLD_LOCAL governs whether *our* symbols enter the global scope for later
//    lookups; it does nothing about our library's own relocations, which still search the
//    global scope first and can bind internally to Flame's ORT. dlsym would hand back our
//    OrtGetApiBase while the library underneath was half-bound to theirs. RTLD_DEEPBIND is
//    what reverses that search order.
//
//    The version string is the discriminator, which is why the bundled runtime should
//    deliberately NOT be 1.22.0: if our handle reports 1.22.0, we got Flame's, whatever the
//    pointers said.
//
// 3. SURVIVAL. RTLD_DEEPBIND's real hazards -- malloc interposition, and exception
//    unwinding across the boundary where std::type_info identity breaks so `catch` and
//    dynamic_cast can fail -- do not appear until real work happens. So each mode creates
//    an environment, builds a session from an embedded model, runs one inference, checks
//    the arithmetic, and tears everything down.
//
// A mode that passes all three is usable. Anything less and inference goes out of process,
// which is already known to work on this hardware (Mocha Pro does exactly that).
//
// ---------------------------------------------------------------------------
// Platforms
// ---------------------------------------------------------------------------
//
// RTLD_DEEPBIND is glibc-only. macOS does not need it: the two-level namespace records which
// library each symbol came from, so this collision largely cannot arise there. The macOS
// path therefore runs the plain-dlopen mode only, and says so rather than reporting a
// missing mode as a failure.

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <unistd.h>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

#include "onnxruntime_c_api.h"

#include "probe_model.inc"

namespace {

// ---------------------------------------------------------------------------
// Host plumbing (deliberately minimal -- this plugin measures one thing)
// ---------------------------------------------------------------------------

OfxHost *gHost = nullptr;
const OfxImageEffectSuiteV1 *gEffect = nullptr;
const OfxPropertySuiteV1 *gProp = nullptr;
const OfxParameterSuiteV1 *gParam = nullptr;
const OfxMessageSuiteV1 *gMessage = nullptr;

const char *const kParamRunProbe = "runOrtProbe";

std::string reportPath() {
  if (const char *explicitPath = std::getenv("WHITEWATER_ORT_PROBE_LOG"))
    return explicitPath;
  const char *tmp = std::getenv("TMPDIR");
  std::string dir = tmp && tmp[0] ? tmp : "/tmp";
  if (!dir.empty() && dir.back() == '/')
    dir.pop_back();
  return dir + "/whitewater-ortprobe.txt";
}

void emit(const std::string &line) {
  std::fprintf(stderr, "[whitewater-ortprobe] %s\n", line.c_str());
  std::fflush(stderr);
  static const std::string path = reportPath();
  if (FILE *f = std::fopen(path.c_str(), "a")) {
    std::fprintf(f, "%s\n", line.c_str());
    std::fclose(f);
  }
}

void emitf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void emitf(const char *fmt, ...) {
  char buf[2048];
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

// ---------------------------------------------------------------------------
// Locating our bundled runtime
// ---------------------------------------------------------------------------
//
// By explicit path, never by soname. dlopen("libonnxruntime.so.1") would consult the search
// path and could hand back the host's already-loaded copy without touching disk -- which
// would silently answer a different question than the one being asked.

#ifdef __APPLE__
const char *const kRuntimeSuffixA = "libonnxruntime.dylib";
// No second copy on macOS: only one mode runs there, so a second would be 40 MB of bundle
// nobody opens.
#else
const char *const kRuntimeSuffixA = "libonnxruntime.so";
const char *const kRuntimeSuffixB = "libonnxruntime-b.so";
#endif

// Where this plugin's own binary lives, so the runtime can be found beside it rather than
// through an environment variable an artist would have to set.
std::string moduleDirectory() {
  Dl_info info;
  if (dladdr(reinterpret_cast<const void *>(&moduleDirectory), &info) && info.dli_fname) {
    std::string path = info.dli_fname;
    const std::size_t slash = path.find_last_of('/');
    if (slash != std::string::npos)
      return path.substr(0, slash);
  }
  return ".";
}

// The directory the per-mode copies live in, with a trailing slash. Each mode appends its
// own leaf name.
std::string runtimeDirectory() {
  if (const char *override = std::getenv("WHITEWATER_ORT_LIBRARY_DIR"))
    return std::string(override) + "/";
  // Contents/<arch>/x.ofx -> Contents/Libraries/
  return moduleDirectory() + "/../Libraries/";
}

// ---------------------------------------------------------------------------
// One test of one dlopen mode
// ---------------------------------------------------------------------------

struct Mode {
  const char *name;
  int flags;
  const char *rationale;
  // Each mode opens its OWN copy of the runtime, at a distinct path.
  //
  // This is not tidiness, it is the difference between measuring and not. dlopen on a path
  // that is already loaded returns the existing handle with a bumped refcount -- it does
  // not reload -- and RTLD_DEEPBIND is only honoured at initial load. Measured 2026-08-20
  // in Flame: mode 2 returned mode 1's library, both reported the identical OrtGetApiBase
  // address, and the DEEPBIND column of that verdict meant nothing.
  //
  // dlclose between modes is not a fix: it is advisory, a runtime that registered
  // thread-local state and atexit handlers may well stay mapped, and unloading ORT inside a
  // host process is its own hazard. Distinct files are unambiguous.
  const char *librarySuffix;
};

struct ModeResult {
  bool opened = false;
  bool distinctFromHost = false;
  bool versionReadable = false;
  bool ranInference = false;
  bool arithmeticCorrect = false;
  std::string version;
  std::string failure;
};

// The subset of the ORT C API this needs, fetched through our own handle.
typedef const OrtApiBase *(*GetApiBaseFn)(void);

ModeResult runMode(const Mode &mode, const std::string &libraryBase, const void *hostApiBase) {
  ModeResult result;

  const std::string library = libraryBase + mode.librarySuffix;

  section(mode.name);
  emitf("  %s", mode.rationale);
  emitf("  dlopen(\"%s\", 0x%x)", library.c_str(), mode.flags);

  dlerror();
  void *handle = dlopen(library.c_str(), mode.flags);
  if (!handle) {
    const char *err = dlerror();
    result.failure = err ? err : "dlopen returned null with no error";
    emitf("  FAILED to open: %s", result.failure.c_str());
    return result;
  }
  result.opened = true;
  emitf("  handle %p", handle);

  // --- Level 1: resolution -------------------------------------------------
  dlerror();
  void *ours = dlsym(handle, "OrtGetApiBase");
  if (!ours) {
    const char *err = dlerror();
    result.failure = err ? err : "OrtGetApiBase not found in our handle";
    emitf("  FAILED: %s", result.failure.c_str());
    dlclose(handle);
    return result;
  }

  result.distinctFromHost = (ours != hostApiBase);
  emitf("  OrtGetApiBase: ours %p, host %p -- %s", ours, hostApiBase,
        result.distinctFromHost ? "DISTINCT" : "SAME POINTER (we got the host's)");

  Dl_info info;
  if (dladdr(ours, &info) && info.dli_fname)
    emitf("    ours resolves into %s", info.dli_fname);

  // --- Level 2: internal consistency ---------------------------------------
  // The version string is the real discriminator. Pointer identity can differ while the
  // library underneath is still half-bound to the host's copy.
  GetApiBaseFn getApiBase = reinterpret_cast<GetApiBaseFn>(ours);
  const OrtApiBase *base = getApiBase();
  if (!base) {
    result.failure = "OrtGetApiBase() returned null";
    emitf("  FAILED: %s", result.failure.c_str());
    dlclose(handle);
    return result;
  }
  if (base->GetVersionString) {
    const char *version = base->GetVersionString();
    if (version) {
      result.version = version;
      result.versionReadable = true;
      emitf("  reported version: %s", version);
      emit("    If this matches the host's ONNX Runtime version, the pointers above are");
      emit("    not evidence of isolation -- deliberately ship a different version so this");
      emit("    line discriminates.");
    }
  }

  const OrtApi *api = base->GetApi(ORT_API_VERSION);
  if (!api) {
    result.failure = "GetApi(ORT_API_VERSION) returned null -- API version mismatch";
    emitf("  FAILED: %s", result.failure.c_str());
    dlclose(handle);
    return result;
  }

  // --- Level 3: survival ---------------------------------------------------
  // Everything below is where RTLD_DEEPBIND's real hazards live. Statuses are checked
  // rather than exceptions caught, because a broken unwinder is one of the things being
  // tested and must not be relied on to report itself.
  OrtEnv *env = nullptr;
  OrtStatus *status =
      api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "whitewater-ortprobe", &env);
  if (status) {
    result.failure = std::string("CreateEnv: ") + api->GetErrorMessage(status);
    emitf("  FAILED: %s", result.failure.c_str());
    api->ReleaseStatus(status);
    dlclose(handle);
    return result;
  }
  emit("  CreateEnv: ok");

  OrtSessionOptions *options = nullptr;
  status = api->CreateSessionOptions(&options);
  if (status) {
    result.failure = std::string("CreateSessionOptions: ") + api->GetErrorMessage(status);
    emitf("  FAILED: %s", result.failure.c_str());
    api->ReleaseStatus(status);
    api->ReleaseEnv(env);
    dlclose(handle);
    return result;
  }
  // One thread. The probe is measuring symbol binding, not throughput, and an unbounded
  // pool inside a host that owns all 72 CPUs is rude at best. The status is checked rather
  // than discarded: a swallowed failure here would leave the thread count looking applied.
  status = api->SetIntraOpNumThreads(options, 1);
  if (status) {
    emitf("  SetIntraOpNumThreads failed (continuing): %s", api->GetErrorMessage(status));
    api->ReleaseStatus(status);
  }

  OrtSession *session = nullptr;
  status = api->CreateSessionFromArray(env, kProbeModel,
                                       static_cast<size_t>(kProbeModelSize), options,
                                       &session);
  if (status) {
    result.failure = std::string("CreateSessionFromArray: ") + api->GetErrorMessage(status);
    emitf("  FAILED: %s", result.failure.c_str());
    api->ReleaseStatus(status);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    dlclose(handle);
    return result;
  }
  emit("  CreateSessionFromArray: ok (128-byte embedded Add model)");

  OrtMemoryInfo *memory = nullptr;
  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory);
  if (status) {
    result.failure = std::string("CreateCpuMemoryInfo: ") + api->GetErrorMessage(status);
    emitf("  FAILED: %s", result.failure.c_str());
    api->ReleaseStatus(status);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(options);
    api->ReleaseEnv(env);
    dlclose(handle);
    return result;
  }

  float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
  const int64_t shape[2] = {1, 4};

  OrtValue *inputs[2] = {nullptr, nullptr};
  bool tensorsOk = true;
  float *data[2] = {a, b};
  for (int i = 0; i < 2 && tensorsOk; ++i) {
    status = api->CreateTensorWithDataAsOrtValue(
        memory, data[i], sizeof(float) * 4, shape, 2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]);
    if (status) {
      result.failure = std::string("CreateTensor: ") + api->GetErrorMessage(status);
      emitf("  FAILED: %s", result.failure.c_str());
      api->ReleaseStatus(status);
      tensorsOk = false;
    }
  }

  if (tensorsOk) {
    const char *inputNames[2] = {"a", "b"};
    const char *outputNames[1] = {"c"};
    OrtValue *output = nullptr;
    status = api->Run(session, nullptr, inputNames, inputs, 2, outputNames, 1, &output);
    if (status) {
      result.failure = std::string("Run: ") + api->GetErrorMessage(status);
      emitf("  FAILED: %s", result.failure.c_str());
      api->ReleaseStatus(status);
    } else {
      result.ranInference = true;
      float *out = nullptr;
      status = api->GetTensorMutableData(output, reinterpret_cast<void **>(&out));
      if (status) {
        result.failure = std::string("GetTensorMutableData: ") + api->GetErrorMessage(status);
        api->ReleaseStatus(status);
      } else if (out) {
        emitf("  Run: ok -- [%.1f %.1f %.1f %.1f]", out[0], out[1], out[2], out[3]);
        result.arithmeticCorrect = out[0] == 11.0f && out[1] == 22.0f && out[2] == 33.0f &&
                                   out[3] == 44.0f;
        if (!result.arithmeticCorrect) {
          result.failure = "inference ran but produced the wrong numbers";
          emit("  FAILED: expected [11 22 33 44] -- a wrong answer here means the session");
          emit("  and the kernels did not come from the same library.");
        }
      }
      if (output)
        api->ReleaseValue(output);
    }
  }

  for (int i = 0; i < 2; ++i)
    if (inputs[i])
      api->ReleaseValue(inputs[i]);
  api->ReleaseMemoryInfo(memory);
  api->ReleaseSession(session);
  api->ReleaseSessionOptions(options);
  api->ReleaseEnv(env);
  emit("  teardown: ok");

  // Deliberately left open. dlclose of a runtime that registered thread-local state and
  // atexit handlers inside a host process is its own hazard, and unloading it is not
  // something the shipping plugin would ever do -- so the probe does not do it either.
  emit("  (handle intentionally not dlclosed -- see the note in the source)");
  return result;
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

void runProbe(OfxImageEffectHandle instance) {
  section("White Water ONNX Runtime isolation probe");

  const std::string library = runtimeDirectory();
  emitf("  bundled runtime directory: %s", library.c_str());
  emit("  Each mode opens a separate copy, because dlopen returns the already-loaded");
  emit("  library for a repeated path and RTLD_DEEPBIND only applies at initial load.");
  emitf("  header ORT_API_VERSION: %d", ORT_API_VERSION);

  // What the host already has, and where from. This is the thing we are trying not to bind
  // to, so it is established first and by name.
  section("The host's ONNX Runtime");
  void *hostApiBase = dlsym(RTLD_DEFAULT, "OrtGetApiBase");
  if (!hostApiBase) {
    emit("  OrtGetApiBase is NOT in the global scope.");
    emit("  There is no collision to isolate from in this host -- a bundled runtime can be");
    emit("  linked normally. (This contradicts the 2026-08-20 measurement in Flame, so if");
    emit("  you are seeing it there, something has changed.)");
  } else {
    Dl_info info;
    const char *from = (dladdr(hostApiBase, &info) && info.dli_fname) ? info.dli_fname
                                                                     : "<unknown>";
    emitf("  OrtGetApiBase %p from %s", hostApiBase, from);
    GetApiBaseFn hostFn = reinterpret_cast<GetApiBaseFn>(hostApiBase);
    const OrtApiBase *hostBase = hostFn();
    if (hostBase && hostBase->GetVersionString) {
      const char *v = hostBase->GetVersionString();
      emitf("  host runtime version: %s", v ? v : "<null>");
    }
  }

  std::vector<Mode> modes;
#ifdef __APPLE__
  modes.push_back({"Mode: plain dlopen (macOS two-level namespace)", RTLD_NOW | RTLD_LOCAL,
                   "macOS records which library each symbol came from, so the flat-namespace "
                   "capture this probe exists to measure does not arise. RTLD_DEEPBIND does "
                   "not exist here.",
                   kRuntimeSuffixA});
#else
  // DEEPBIND first. If the two modes ever share a library again through some path this
  // code did not anticipate, the mode that gets genuinely measured should be the one still
  // in question -- not the one already answered.
  modes.push_back({"Mode 1: RTLD_LOCAL | RTLD_DEEPBIND", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND,
                   "DEEPBIND makes the library prefer its own symbols over the global scope. "
                   "Measured 2026-08-20: RTLD_LOCAL alone already sufficed in Flame, so what "
                   "this mode now decides is whether DEEPBIND is safe to use, not whether it "
                   "is necessary.",
                   kRuntimeSuffixB});
  modes.push_back({"Mode 2: RTLD_LOCAL only", RTLD_NOW | RTLD_LOCAL,
                   "RTLD_LOCAL keeps our symbols out of the global scope for later lookups; "
                   "it does not reorder how our own library's relocations resolve. Expected "
                   "to be insufficient on paper, and measured sufficient in Flame -- most "
                   "likely because ONNX Runtime is built with hidden visibility, so its "
                   "internals never consult the global scope and there is nothing to capture.",
                   kRuntimeSuffixA});
#endif

  std::vector<ModeResult> results;
  for (const Mode &mode : modes)
    results.push_back(runMode(mode, library, hostApiBase));

  // --- Verdict -------------------------------------------------------------
  section("VERDICT");
  for (std::size_t i = 0; i < modes.size(); ++i) {
    const ModeResult &r = results[i];
    const bool usable = r.opened && r.distinctFromHost && r.ranInference &&
                        r.arithmeticCorrect;
    emitf("  %-52s %s", modes[i].name, usable ? "USABLE" : "NOT USABLE");
    emitf("      opened %d | distinct pointer %d | version %s | ran %d | arithmetic %d",
          (int)r.opened, (int)r.distinctFromHost,
          r.versionReadable ? r.version.c_str() : "<unread>", (int)r.ranInference,
          (int)r.arithmeticCorrect);
    if (!r.failure.empty())
      emitf("      failure: %s", r.failure.c_str());
  }

  emit("");
  bool anyUsable = false;
  for (const ModeResult &r : results)
    anyUsable = anyUsable || (r.opened && r.distinctFromHost && r.ranInference &&
                              r.arithmeticCorrect);

  if (anyUsable) {
    emit("  At least one mode is usable, so IN-PROCESS inference is available.");
    emit("  Before committing to it, confirm the reported version is NOT the host's --");
    emit("  matching versions make the whole result ambiguous.");
  } else {
    emit("  No mode is usable. Inference goes OUT OF PROCESS, which is already known to");
    emit("  work on this hardware: Mocha Pro ships exactly that shape, and warp-drive has");
    emit("  the supervisor and frame channel to copy (src/ofx/EditorProcess.cpp, src/ipc/).");
  }
  emit("");
  emitf("  Report: %s", reportPath().c_str());

  if (gMessage)
    gMessage->message(instance, kOfxMessageMessage, nullptr,
                      "White Water ORT probe written to:\n%s", reportPath().c_str());
}

// ---------------------------------------------------------------------------
// OFX actions
// ---------------------------------------------------------------------------

OfxStatus onLoad() {
  gEffect = (const OfxImageEffectSuiteV1 *)gHost->fetchSuite(gHost->host,
                                                             kOfxImageEffectSuite, 1);
  gProp = (const OfxPropertySuiteV1 *)gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1);
  gParam = (const OfxParameterSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1);
  gMessage = (const OfxMessageSuiteV1 *)gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1);
  if (!gEffect || !gProp || !gParam)
    return kOfxStatErrMissingHostFeature;

  std::time_t now = std::time(nullptr);
  char stamp[64] = {0};
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  emit("");
  emit("=========================================================================");
  emitf("White Water ORT probe -- session started %s (pid %d)", stamp, (int)getpid());
  emit("=========================================================================");
  return kOfxStatOK;
}

OfxStatus onDescribe(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffect->getPropertySet(effect, &props);
  gProp->propSetString(props, kOfxPropLabel, 0, "White Water ORT Probe");
  gProp->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "White Water");
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 0,
                       kOfxImageEffectContextFilter);
  gProp->propSetString(props, kOfxImageEffectPropSupportedContexts, 1,
                       kOfxImageEffectContextGeneral);
  gProp->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthFloat);
  gProp->propSetString(props, kOfxImageEffectPluginRenderThreadSafety, 0,
                       kOfxImageEffectRenderFullySafe);
  gProp->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  return kOfxStatOK;
}

OfxStatus onDescribeInContext(OfxImageEffectHandle effect) {
  OfxPropertySetHandle clipProps = nullptr;
  gEffect->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &clipProps);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 0,
                       kOfxImageComponentRGBA);
  gEffect->clipDefine(effect, kOfxImageEffectOutputClipName, &clipProps);
  gProp->propSetString(clipProps, kOfxImageEffectPropSupportedComponents, 0,
                       kOfxImageComponentRGBA);

  OfxParamSetHandle paramSet = nullptr;
  gEffect->getParamSet(effect, &paramSet);
  OfxPropertySetHandle paramProps = nullptr;
  gParam->paramDefine(paramSet, kOfxParamTypePushButton, kParamRunProbe, &paramProps);
  if (paramProps) {
    gProp->propSetString(paramProps, kOfxPropLabel, 0, "Run ORT Probe");
    gProp->propSetString(paramProps, kOfxParamPropHint, 0,
                         "Load the bundled ONNX Runtime and report whether it is isolated "
                         "from the host's copy.");
  }
  return kOfxStatOK;
}

// Passthrough, so the probe is harmless in a real comp.
OfxStatus onRender(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs) {
  double time = 0.0;
  gProp->propGetDouble(inArgs, kOfxPropTime, 0, &time);

  OfxImageClipHandle sourceClip = nullptr, outputClip = nullptr;
  gEffect->clipGetHandle(instance, kOfxImageEffectSimpleSourceClipName, &sourceClip, nullptr);
  gEffect->clipGetHandle(instance, kOfxImageEffectOutputClipName, &outputClip, nullptr);

  OfxPropertySetHandle sourceImage = nullptr, outputImage = nullptr;
  if (sourceClip)
    gEffect->clipGetImage(sourceClip, time, nullptr, &sourceImage);
  if (outputClip)
    gEffect->clipGetImage(outputClip, time, nullptr, &outputImage);

  if (outputImage) {
    void *outData = nullptr;
    int outRowBytes = 0, outBounds[4] = {0, 0, 0, 0};
    gProp->propGetPointer(outputImage, kOfxImagePropData, 0, &outData);
    gProp->propGetInt(outputImage, kOfxImagePropRowBytes, 0, &outRowBytes);
    for (int i = 0; i < 4; ++i)
      gProp->propGetInt(outputImage, kOfxImagePropBounds, i, &outBounds[i]);

    void *srcData = nullptr;
    int srcRowBytes = 0, srcBounds[4] = {0, 0, 0, 0};
    if (sourceImage) {
      gProp->propGetPointer(sourceImage, kOfxImagePropData, 0, &srcData);
      gProp->propGetInt(sourceImage, kOfxImagePropRowBytes, 0, &srcRowBytes);
      for (int i = 0; i < 4; ++i)
        gProp->propGetInt(sourceImage, kOfxImagePropBounds, i, &srcBounds[i]);
    }

    const int width = outBounds[2] - outBounds[0];
    const int pixelBytes = 4 * (int)sizeof(float);
    for (int y = outBounds[1]; y < outBounds[3]; ++y) {
      char *dstRow = (char *)outData + (size_t)(y - outBounds[1]) * outRowBytes;
      const bool haveRow = srcData && y >= srcBounds[1] && y < srcBounds[3];
      if (!haveRow) {
        std::memset(dstRow, 0, (size_t)width * pixelBytes);
        continue;
      }
      const char *srcRow = (const char *)srcData + (size_t)(y - srcBounds[1]) * srcRowBytes;
      const int copyFrom = outBounds[0] < srcBounds[0] ? srcBounds[0] : outBounds[0];
      const int copyTo = outBounds[2] < srcBounds[2] ? outBounds[2] : srcBounds[2];
      std::memset(dstRow, 0, (size_t)width * pixelBytes);
      if (copyTo > copyFrom) {
        std::memcpy(dstRow + (size_t)(copyFrom - outBounds[0]) * pixelBytes,
                    srcRow + (size_t)(copyFrom - srcBounds[0]) * pixelBytes,
                    (size_t)(copyTo - copyFrom) * pixelBytes);
      }
    }
  }

  if (sourceImage)
    gEffect->clipReleaseImage(sourceImage);
  if (outputImage)
    gEffect->clipReleaseImage(outputImage);
  return outputImage ? kOfxStatOK : kOfxStatFailed;
}

OfxStatus onInstanceChanged(OfxImageEffectHandle instance, OfxPropertySetHandle inArgs) {
  char *name = nullptr;
  gProp->propGetString(inArgs, kOfxPropName, 0, &name);
  if (name && std::strcmp(name, kParamRunProbe) == 0) {
    runProbe(instance);
    return kOfxStatOK;
  }
  return kOfxStatReplyDefault;
}

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
      return onDescribeInContext(effect);
    if (std::strcmp(action, kOfxImageEffectActionRender) == 0)
      return onRender(effect, inArgs);
    if (std::strcmp(action, kOfxActionInstanceChanged) == 0)
      return onInstanceChanged(effect, inArgs);
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
    kOfxImageEffectPluginApi, 1, "com.mtifilm.whitewater.ortprobe", 1, 0, setHostFunc,
    pluginMain,
};

}  // namespace

// See the note in hostprobe.cpp: ofxCore.h defines OfxExport as a bare `extern` on Unix,
// which is not enough under -fvisibility=hidden.
#define WHITEWATER_OFX_EXPORT extern "C" __attribute__((visibility("default")))

WHITEWATER_OFX_EXPORT int OfxGetNumberOfPlugins(void) { return 1; }

WHITEWATER_OFX_EXPORT OfxPlugin *OfxGetPlugin(int nth) {
  return nth == 0 ? &gPlugin : nullptr;
}

WHITEWATER_OFX_EXPORT OfxStatus OfxSetHost(const OfxHost *host) {
  gHost = const_cast<OfxHost *>(host);
  return kOfxStatOK;
}
