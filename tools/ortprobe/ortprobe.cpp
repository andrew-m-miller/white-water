// White Water ONNX Runtime isolation probe.
//
// Phase 0A established that a plugin can bundle its own ONNX Runtime and use it inside
// Flame even though Flame already has ONNX Runtime 1.22.0 in the global symbol scope.
// Phase 0B extends that measurement to a pinned real optical-flow network on CPU and the
// CUDA execution provider. The tiny embedded Add graph remains as a cheap isolation canary.
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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <memory>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include <dlfcn.h>
#if defined(__linux__)
#include <link.h>
#endif
#include <sys/stat.h>
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
std::atomic<unsigned long long> gCreatedInstances{0};
std::atomic<unsigned long long> gLiveInstances{0};

const char *const kParamRunProbe = "runOrtProbe";

// A check has three outcomes.  "Attempted" is kept separate from the result because an
// unavailable CUDA provider, an absent NVML library, and a deliberately unsafe OOM exercise
// are not failures.  Printing these as bools was how the first report ended up with a page of
// misleading zeroes.
enum class ProbeState { NotTested, Pass, Fail };

const char *probeState(ProbeState state) {
  switch (state) {
  case ProbeState::Pass:
    return "PASS";
  case ProbeState::Fail:
    return "FAIL";
  case ProbeState::NotTested:
    return "NOT TESTED";
  }
  return "NOT TESTED";
}

ProbeState stateFor(bool attempted, bool passed) {
  if (!attempted)
    return ProbeState::NotTested;
  return passed ? ProbeState::Pass : ProbeState::Fail;
}

const char *yesNo(bool value) { return value ? "yes" : "no"; }

std::string errnoDescription(int error) {
  char buffer[256] = {0};
#if defined(__GLIBC__) || defined(__APPLE__)
  // strerror_r has two incompatible signatures.  The GNU variant returns a char pointer;
  // the POSIX variant writes into our buffer.  Both are available on the supported hosts.
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  const char *message = ::strerror_r(error, buffer, sizeof(buffer));
  return message ? message : "unknown error";
#else
  if (::strerror_r(error, buffer, sizeof(buffer)) == 0)
    return buffer;
  return "unknown error";
#endif
#else
  const char *message = std::strerror(error);
  return message ? message : "unknown error";
#endif
}

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

std::string modelPath() {
  if (const char *override = std::getenv("WHITEWATER_ORT_MODEL"))
    return override;
  // Contents/<arch>/x.ofx -> Contents/Resources/models/x.onnx
  return moduleDirectory() + "/../Resources/models/sea-raft-m-opset17.onnx";
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
  // Phase 0A selected plain RTLD_LOCAL. Run the expensive real network only in that mode;
  // the tiny Add model still exercises every mode as an isolation canary.
  bool runRealModel;
};

// The subset of the ORT C API this needs, fetched through our own handle.
typedef const OrtApiBase *(*GetApiBaseFn)(void);

struct FlowRun {
  bool ran = false;
  bool correct = false;
  double sessionMilliseconds = 0.0;
  double firstRunMilliseconds = 0.0;
  double identityMedianEpe = std::numeric_limits<double>::infinity();
  double forwardMedianX = 0.0;
  double forwardMedianY = 0.0;
  double reverseMedianX = 0.0;
  double reverseMedianY = 0.0;
  struct TimingResult {
    int height = 0;
    int width = 0;
    int steadySamples = 0;
    double warmMilliseconds = 0.0;
    double steadyMedianMilliseconds = 0.0;
    double steadyMinimumMilliseconds = 0.0;
    double steadyMaximumMilliseconds = 0.0;
    ProbeState state = ProbeState::NotTested;
    std::string failure;
  };
  struct VramMetrics {
    bool attempted = false;
    bool available = false;
    bool hasSample = false;
    bool hasSteadySample = false;
    std::uint64_t totalBytes = 0;
    std::uint64_t baselineUsedBytes = 0;
    std::uint64_t peakUsedBytes = 0;
    std::uint64_t steadyUsedBytes = 0;
    ProbeState state = ProbeState::NotTested;
    std::string failure;
  };
  bool cancellationAttempted = false;
  bool cancellationObserved = false;
  bool cancellationTimedOut = false;
  ProbeState cancellationState = ProbeState::NotTested;
  std::string cancellationFailure;
  VramMetrics vram;
  std::vector<TimingResult> timings;
  bool leakedResources = false;
  std::string failure;
};

struct LifecycleResult {
  bool attempted = false;
  int requested = 0;
  int completed = 0;
  ProbeState state = ProbeState::NotTested;
  std::string failure;
};

struct FallbackResult {
  bool attempted = false;
  bool providerFailureObserved = false;
  bool cpuFallbackRan = false;
  ProbeState state = ProbeState::NotTested;
  std::string failure;
};

enum class ArenaLimitFailureStage {
  None,
  CreateSessionOptions,
  ConfigureSessionOptions,
  AppendProvider,
  CreateSession,
  CreateMemoryInfo,
  Run,
};

enum class ArenaLimitFailureKind {
  None,
  ProviderInit,
  AllocatorLimit,
  UnexpectedSuccess,
  Other,
};

struct ArenaLimitResult {
  bool attempted = false;
  bool allocatorFailureObserved = false;
  bool cpuRecoveryRan = false;
  bool vramBeforeAvailable = false;
  bool vramAfterAvailable = false;
  std::uint64_t vramBeforeBytes = 0;
  std::uint64_t vramAfterBytes = 0;
  std::size_t gpuMemoryLimitBytes = 0;
  int targetHeight = 0;
  int targetWidth = 0;
  ArenaLimitFailureStage failureStage = ArenaLimitFailureStage::None;
  ArenaLimitFailureKind failureKind = ArenaLimitFailureKind::None;
  ProbeState state = ProbeState::NotTested;
  std::string failure;
};

struct ModeResult {
  bool opened = false;
  bool distinctFromHost = false;
  bool versionReadable = false;
  bool ranInference = false;
  bool arithmeticCorrect = false;
  bool realModelChecked = false;
  bool realModelConfigured = false;
  bool realModelRan = false;
  bool realModelCorrect = false;
  bool cudaAvailable = false;
  bool cudaProviderChecked = false;
  bool cudaRan = false;
  bool cudaCorrect = false;
  // The extended Phase 0B measurements are kept as structured values so the final verdict
  // can say NOT TESTED instead of inventing a zero for a path the host did not offer.
  bool cpuFlowAvailable = false;
  bool cudaFlowAvailable = false;
  FlowRun cpuFlow;
  FlowRun cudaFlow;
  LifecycleResult cpuLifecycle;
  LifecycleResult cudaLifecycle;
  FallbackResult providerFallback;
  ArenaLimitResult gpuArenaLimit;
  ProbeState nodeDuplicationState = ProbeState::NotTested;
  std::string nodeDuplicationFailure;
  std::string version;
  std::string failure;
};

std::string takeStatus(const OrtApi *api, OrtStatus *status) {
  if (!status)
    return {};
  const char *message = api->GetErrorMessage(status);
  std::string result = message ? message : "unknown ONNX Runtime error";
  api->ReleaseStatus(status);
  return result;
}

double median(std::vector<float> values) {
  if (values.empty())
    return std::numeric_limits<double>::quiet_NaN();
  const std::size_t middle = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + middle, values.end());
  if (values.size() % 2)
    return values[middle];
  const float upper = values[middle];
  std::nth_element(values.begin(), values.begin() + middle - 1, values.begin() + middle);
  return 0.5 * (upper + values[middle - 1]);
}

std::vector<float> makeTexture(int height, int width) {
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  std::vector<float> raw(plane * 3), smooth(plane * 3);
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        unsigned value = static_cast<unsigned>(x + 1) * 73856093u;
        value ^= static_cast<unsigned>(y + 1) * 19349663u;
        value ^= static_cast<unsigned>(c + 1) * 83492791u;
        value ^= value >> 13;
        value *= 1274126177u;
        raw[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * width + x] =
            static_cast<float>(value & 0xffffu) * (255.0f / 65535.0f);
      }
    }
  }
  // Match the export test's textured, locally smooth signal without depending on PyTorch's
  // RNG. Zero padding is intentional: avg_pool2d(..., padding=2) uses it too.
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        float sum = 0.0f;
        for (int ky = -2; ky <= 2; ++ky)
          for (int kx = -2; kx <= 2; ++kx)
            if (y + ky >= 0 && y + ky < height && x + kx >= 0 && x + kx < width)
              sum += raw[static_cast<std::size_t>(c) * plane +
                         static_cast<std::size_t>(y + ky) * width + x + kx];
        smooth[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * width + x] =
            sum / 25.0f;
      }
    }
  }
  return smooth;
}

std::vector<float> translateRight(const std::vector<float> &source, int height, int width,
                                  int dx) {
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  std::vector<float> result(source.size(), 0.0f);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = dx; x < width; ++x)
        result[static_cast<std::size_t>(c) * plane + static_cast<std::size_t>(y) * width + x] =
            source[static_cast<std::size_t>(c) * plane +
                   static_cast<std::size_t>(y) * width + x - dx];
  return result;
}

// NVML is deliberately optional and reached through dlopen.  Linking it would add a new
// dependency to the probe bundle and, worse, make a missing driver a load-time failure.  The
// small ABI below is stable across the driver versions supported by Flame; no NVML header is
// needed.  The values are process-wide device counters, so the report also records the Batch
// baseline and does not pretend they are allocations belonging only to this plugin.
#if defined(__linux__)
struct NvmlMemoryInfo {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

using NvmlInitFn = int (*)();
using NvmlShutdownFn = int (*)();
using NvmlDeviceGetHandleByIndexFn = int (*)(unsigned int, void **);
using NvmlDeviceGetMemoryInfoFn = int (*)(void *, NvmlMemoryInfo *);

struct VramMonitor {
  bool attempted = false;
  bool initialized = false;
  bool available = false;
  void *handle = nullptr;
  void *device = nullptr;
  NvmlInitFn init = nullptr;
  NvmlShutdownFn shutdown = nullptr;
  NvmlDeviceGetHandleByIndexFn getDevice = nullptr;
  NvmlDeviceGetMemoryInfoFn getMemory = nullptr;
  std::string failure;

  ~VramMonitor() {
    if (initialized && shutdown)
      shutdown();
    if (handle)
      dlclose(handle);
  }

  bool initialize() {
    if (attempted)
      return available;
    attempted = true;
    const char *overridePath = std::getenv("WHITEWATER_NVML_LIBRARY");
    const char *path = overridePath && overridePath[0] ? overridePath : "libnvidia-ml.so.1";
    dlerror();
    handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char *error = dlerror();
      failure = error ? error : "dlopen returned null with no error";
      return false;
    }
    init = reinterpret_cast<NvmlInitFn>(dlsym(handle, "nvmlInit_v2"));
    if (!init)
      init = reinterpret_cast<NvmlInitFn>(dlsym(handle, "nvmlInit"));
    shutdown = reinterpret_cast<NvmlShutdownFn>(dlsym(handle, "nvmlShutdown"));
    getDevice = reinterpret_cast<NvmlDeviceGetHandleByIndexFn>(
        dlsym(handle, "nvmlDeviceGetHandleByIndex_v2"));
    if (!getDevice)
      getDevice = reinterpret_cast<NvmlDeviceGetHandleByIndexFn>(
          dlsym(handle, "nvmlDeviceGetHandleByIndex"));
    getMemory = reinterpret_cast<NvmlDeviceGetMemoryInfoFn>(
        dlsym(handle, "nvmlDeviceGetMemoryInfo"));
    if (!init || !shutdown || !getDevice || !getMemory) {
      failure = "NVML symbols missing (nvmlInit, nvmlDeviceGetHandleByIndex, "
                "nvmlDeviceGetMemoryInfo, nvmlShutdown)";
      return false;
    }
    const int initStatus = init();
    if (initStatus != 0) {
      char buffer[128];
      std::snprintf(buffer, sizeof(buffer), "nvmlInit returned %d", initStatus);
      failure = buffer;
      return false;
    }
    initialized = true;
    const int deviceStatus = getDevice(0, &device);
    if (deviceStatus != 0 || !device) {
      char buffer[128];
      std::snprintf(buffer, sizeof(buffer), "nvmlDeviceGetHandleByIndex(0) returned %d",
                    deviceStatus);
      failure = buffer;
      return false;
    }
    available = true;
    emitf("    NVML VRAM monitor: available (device 0, %s)", path);
    return true;
  }

  bool sample(NvmlMemoryInfo &memory) {
    if (!available || !getMemory)
      return false;
    const int status = getMemory(device, &memory);
    if (status != 0) {
      char buffer[128];
      std::snprintf(buffer, sizeof(buffer), "nvmlDeviceGetMemoryInfo returned %d", status);
      failure = buffer;
      return false;
    }
    return true;
  }
};
#else
struct VramMonitor {
  bool attempted = false;
  bool available = false;
  std::string failure;

  bool initialize() {
    attempted = true;
    failure = "NVML VRAM sampling is only available on Linux";
    return false;
  }
};
#endif

void sampleVram(VramMonitor *monitor, FlowRun::VramMetrics &metrics, const char *label) {
  if (!monitor)
    return;
  metrics.attempted = true;
  metrics.available = monitor->available;
  if (!monitor->available) {
    metrics.state = ProbeState::NotTested;
    if (metrics.failure.empty())
      metrics.failure = monitor->failure;
    return;
  }
#if defined(__linux__)
  NvmlMemoryInfo sample = {};
  if (!monitor->sample(sample)) {
    metrics.state = ProbeState::Fail;
    metrics.failure = monitor->failure;
    return;
  }
  if (!metrics.hasSample) {
    metrics.totalBytes = sample.total;
    metrics.baselineUsedBytes = sample.used;
    metrics.peakUsedBytes = sample.used;
    metrics.hasSample = true;
  } else {
    metrics.peakUsedBytes =
        std::max(metrics.peakUsedBytes, static_cast<std::uint64_t>(sample.used));
  }
  if (std::strcmp(label, "steady") == 0) {
    metrics.steadyUsedBytes = sample.used;
    metrics.hasSteadySample = true;
  }
  metrics.state = ProbeState::Pass;
  emitf("    VRAM %-14s total %.1f MiB | used %.1f MiB | free %.1f MiB (sample)", label,
        static_cast<double>(sample.total) / (1024.0 * 1024.0),
        static_cast<double>(sample.used) / (1024.0 * 1024.0),
        static_cast<double>(sample.free) / (1024.0 * 1024.0));
#else
  (void)label;
#endif
}

bool sampleVramUsed(VramMonitor *monitor, std::uint64_t &usedBytes) {
  if (!monitor || !monitor->available)
    return false;
#if defined(__linux__)
  NvmlMemoryInfo memory = {};
  if (!monitor->sample(memory))
    return false;
  usedBytes = memory.used;
  return true;
#else
  (void)usedBytes;
  return false;
#endif
}

struct ProbeSize {
  int height = 0;
  int width = 0;
};

std::vector<ProbeSize> usefulProbeSizes() {
  // These are deliberately useful but bounded: 480p exercises a dynamic shape and a
  // host-sized transfer without turning a diagnostic button into a long blocking action.  A
  // caller may select 720p (or other multiples of eight) explicitly, but the hard cap remains
  // in force.  The 720p run is opt-in because SEA-RAFT's CPU cost grows quickly with pixels.
  const char *spec = std::getenv("WHITEWATER_ORT_PROBE_SIZES");
  if (!spec || !spec[0])
    spec = "480x640";

  std::vector<ProbeSize> sizes;
  const std::string value(spec);
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t comma = value.find(',', begin);
    const std::string token = value.substr(begin, comma == std::string::npos
                                                    ? std::string::npos
                                                    : comma - begin);
    const std::size_t x = token.find_first_of("xX");
    char *heightEnd = nullptr;
    char *widthEnd = nullptr;
    long height = 0;
    long width = 0;
    if (x != std::string::npos) {
      const std::string heightText = token.substr(0, x);
      const std::string widthText = token.substr(x + 1);
      height = std::strtol(heightText.c_str(), &heightEnd, 10);
      width = std::strtol(widthText.c_str(), &widthEnd, 10);
    }
    const bool parsed = x != std::string::npos && heightEnd && widthEnd &&
                        *heightEnd == '\0' && *widthEnd == '\0';
    const std::uint64_t pixels = parsed && height > 0 && width > 0
                                     ? static_cast<std::uint64_t>(height) * width
                                     : 0;
    const bool bounded = parsed && height >= 8 && width >= 8 && height <= 2048 && width <= 2048 &&
                         height % 8 == 0 && width % 8 == 0 && pixels <= 2u * 1024u * 1024u;
    if (bounded) {
      sizes.push_back({static_cast<int>(height), static_cast<int>(width)});
    } else {
      emitf("    timing size '%s' rejected (use HxW, multiples of 8, <=2048x2048 and "
            "<=2 Mpix)",
            token.c_str());
    }
    if (comma == std::string::npos)
      break;
    begin = comma + 1;
  }
  return sizes;
}

bool runFlowOnce(const OrtApi *api, OrtSession *session, OrtMemoryInfo *memory,
                 const std::vector<float> &image1, const std::vector<float> &image2,
                 int height, int width, std::vector<float> &flow, double &milliseconds,
                 std::string &failure) {
  const int64_t shape[4] = {1, 3, height, width};
  OrtValue *inputs[2] = {nullptr, nullptr};
  const std::vector<float> *data[2] = {&image1, &image2};
  for (int i = 0; i < 2; ++i) {
    OrtStatus *status = api->CreateTensorWithDataAsOrtValue(
        memory, const_cast<float *>(data[i]->data()), data[i]->size() * sizeof(float), shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[i]);
    if (status) {
      failure = "CreateTensor: " + takeStatus(api, status);
      for (OrtValue *input : inputs)
        if (input)
          api->ReleaseValue(input);
      return false;
    }
  }

  const char *inputNames[2] = {"image1", "image2"};
  const char *outputNames[1] = {"flow"};
  OrtValue *output = nullptr;
  const auto start = std::chrono::steady_clock::now();
  OrtStatus *status = api->Run(session, nullptr, inputNames, inputs, 2, outputNames, 1, &output);
  milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                     .count();
  for (OrtValue *input : inputs)
    api->ReleaseValue(input);
  if (status) {
    failure = "Run: " + takeStatus(api, status);
    if (output)
      api->ReleaseValue(output);
    return false;
  }

  OrtTensorTypeAndShapeInfo *info = nullptr;
  status = api->GetTensorTypeAndShape(output, &info);
  if (status) {
    failure = "GetTensorTypeAndShape: " + takeStatus(api, status);
    api->ReleaseValue(output);
    return false;
  }
  int64_t dimensions[4] = {0, 0, 0, 0};
  status = api->GetDimensions(info, dimensions, 4);
  api->ReleaseTensorTypeAndShapeInfo(info);
  if (status) {
    failure = "GetDimensions: " + takeStatus(api, status);
    api->ReleaseValue(output);
    return false;
  }
  if (dimensions[0] != 1 || dimensions[1] != 2 || dimensions[2] != height ||
      dimensions[3] != width) {
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), "unexpected flow shape [%lld %lld %lld %lld]",
                  (long long)dimensions[0], (long long)dimensions[1],
                  (long long)dimensions[2], (long long)dimensions[3]);
    failure = buffer;
    api->ReleaseValue(output);
    return false;
  }
  float *outputData = nullptr;
  status = api->GetTensorMutableData(output, reinterpret_cast<void **>(&outputData));
  if (status || !outputData) {
    failure = status ? "GetTensorMutableData: " + takeStatus(api, status)
                     : "GetTensorMutableData returned null";
    api->ReleaseValue(output);
    return false;
  }
  flow.assign(outputData, outputData + static_cast<std::size_t>(2) * height * width);
  api->ReleaseValue(output);
  return true;
}

std::vector<FlowRun::TimingResult> runUsefulTimings(const OrtApi *api, OrtSession *session,
                                                    OrtMemoryInfo *memory,
                                                    VramMonitor *vramMonitor = nullptr,
                                                    FlowRun::VramMetrics *vramMetrics = nullptr) {
  std::vector<FlowRun::TimingResult> results;
  const std::vector<ProbeSize> sizes = usefulProbeSizes();
  for (const ProbeSize &size : sizes) {
    FlowRun::TimingResult result;
    result.height = size.height;
    result.width = size.width;
    result.steadySamples = 3;

    const std::vector<float> first = makeTexture(size.height, size.width);
    const std::vector<float> second = translateRight(first, size.height, size.width, 4);
    std::vector<float> flow;
    if (!runFlowOnce(api, session, memory, first, second, size.height, size.width, flow,
                     result.warmMilliseconds, result.failure)) {
      result.state = ProbeState::Fail;
      results.push_back(std::move(result));
      continue;
    }
    if (vramMonitor && vramMetrics)
      sampleVram(vramMonitor, *vramMetrics, "timing warm");

    std::vector<float> steady;
    steady.reserve(static_cast<std::size_t>(result.steadySamples));
    for (int sample = 0; sample < result.steadySamples; ++sample) {
      double milliseconds = 0.0;
      if (!runFlowOnce(api, session, memory, first, second, size.height, size.width, flow,
                       milliseconds, result.failure)) {
        result.state = ProbeState::Fail;
        break;
      }
      if (vramMonitor && vramMetrics)
        sampleVram(vramMonitor, *vramMetrics, "steady");
      steady.push_back(static_cast<float>(milliseconds));
    }
    if (result.state != ProbeState::Fail &&
        static_cast<int>(steady.size()) == result.steadySamples) {
      result.steadyMedianMilliseconds = median(steady);
      result.steadyMinimumMilliseconds = *std::min_element(steady.begin(), steady.end());
      result.steadyMaximumMilliseconds = *std::max_element(steady.begin(), steady.end());
      result.state = ProbeState::Pass;
    }
    results.push_back(std::move(result));
  }
  return results;
}

struct CancellationCall {
  const OrtApi *api = nullptr;
  OrtSession *session = nullptr;
  OrtRunOptions *options = nullptr;
  OrtValue *inputs[2] = {nullptr, nullptr};
  OrtValue *output = nullptr;
  // Keep the backing tensors alive if the bounded join ever has to detach the worker.  A
  // detached diagnostic thread must never retain pointers into runCancellation's stack.
  std::vector<float> first;
  std::vector<float> second;
  std::atomic<bool> started{false};
  std::atomic<bool> finished{false};
  bool runSucceeded = false;
  bool terminationError = false;
  std::string message;
};

bool looksLikeTermination(const std::string &message) {
  std::string lower = message;
  for (char &character : lower) {
    if (character >= 'A' && character <= 'Z')
      character = static_cast<char>(character - 'A' + 'a');
  }
  return lower.find("terminat") != std::string::npos ||
         lower.find("cancel") != std::string::npos || lower.find("abort") != std::string::npos;
}

bool looksLikeExplicitCudaArenaLimitFailure(const std::string &message) {
  std::string lower = message;
  for (char &character : lower) {
    if (character >= 'A' && character <= 'Z')
      character = static_cast<char>(character - 'A' + 'a');
  }
  const bool allocationSpecific = lower.find("out of memory") != std::string::npos ||
                                  lower.find("failed to allocate") != std::string::npos ||
                                  lower.find("memory allocation") != std::string::npos ||
                                  lower.find("available memory") != std::string::npos;
  return lower.find("gpu_mem_limit") != std::string::npos ||
         lower.find("bfcarena") != std::string::npos ||
         (lower.find("arena") != std::string::npos && allocationSpecific);
}

const char *arenaLimitFailureStage(ArenaLimitFailureStage stage) {
  switch (stage) {
  case ArenaLimitFailureStage::None:
    return "none";
  case ArenaLimitFailureStage::CreateSessionOptions:
    return "CreateSessionOptions";
  case ArenaLimitFailureStage::ConfigureSessionOptions:
    return "ConfigureSessionOptions";
  case ArenaLimitFailureStage::AppendProvider:
    return "AppendProvider";
  case ArenaLimitFailureStage::CreateSession:
    return "CreateSession";
  case ArenaLimitFailureStage::CreateMemoryInfo:
    return "CreateMemoryInfo";
  case ArenaLimitFailureStage::Run:
    return "Run";
  }
  return "unknown";
}

const char *arenaLimitFailureKind(ArenaLimitFailureKind kind) {
  switch (kind) {
  case ArenaLimitFailureKind::None:
    return "none";
  case ArenaLimitFailureKind::ProviderInit:
    return "provider-init";
  case ArenaLimitFailureKind::AllocatorLimit:
    return "allocator-limit";
  case ArenaLimitFailureKind::UnexpectedSuccess:
    return "unexpected-success";
  case ArenaLimitFailureKind::Other:
    return "other";
  }
  return "unknown";
}

void releaseCancellationInputs(const std::shared_ptr<CancellationCall> &call) {
  for (OrtValue *&input : call->inputs) {
    if (input) {
      call->api->ReleaseValue(input);
      input = nullptr;
    }
  }
  if (call->output) {
    call->api->ReleaseValue(call->output);
    call->output = nullptr;
  }
}

void runCancellation(const OrtApi *api, OrtSession *session, OrtMemoryInfo *memory,
                     FlowRun &result) {
  result.cancellationAttempted = true;
  constexpr int height = 480;
  constexpr int width = 640;
  const int64_t shape[4] = {1, 3, height, width};
  auto call = std::make_shared<CancellationCall>();
  call->api = api;
  call->session = session;
  call->first = makeTexture(height, width);
  call->second = translateRight(call->first, height, width, 4);
  const std::vector<float> *data[2] = {&call->first, &call->second};
  for (int i = 0; i < 2; ++i) {
    OrtStatus *status = api->CreateTensorWithDataAsOrtValue(
        memory, const_cast<float *>(data[i]->data()), data[i]->size() * sizeof(float), shape, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &call->inputs[i]);
    if (status) {
      result.cancellationState = ProbeState::Fail;
      result.cancellationFailure = "CreateTensor for cancellation: " + takeStatus(api, status);
      releaseCancellationInputs(call);
      return;
    }
  }

  OrtStatus *status = api->CreateRunOptions(&call->options);
  if (status) {
    result.cancellationState = ProbeState::Fail;
    result.cancellationFailure = "CreateRunOptions: " + takeStatus(api, status);
    releaseCancellationInputs(call);
    return;
  }
  status = api->RunOptionsSetRunTag(call->options, "whitewater-phase0b-cancel");
  if (status) {
    result.cancellationState = ProbeState::Fail;
    result.cancellationFailure = "RunOptionsSetRunTag: " + takeStatus(api, status);
    api->ReleaseRunOptions(call->options);
    call->options = nullptr;
    releaseCancellationInputs(call);
    return;
  }

  std::thread worker([call]() {
    call->started.store(true, std::memory_order_release);
    const char *inputNames[2] = {"image1", "image2"};
    const char *outputNames[1] = {"flow"};
    const OrtValue *inputs[2] = {call->inputs[0], call->inputs[1]};
    OrtStatus *runStatus = call->api->Run(call->session, call->options, inputNames, inputs, 2,
                                           outputNames, 1, &call->output);
    if (runStatus) {
      call->message = takeStatus(call->api, runStatus);
      call->terminationError = looksLikeTermination(call->message);
    } else {
      call->runSucceeded = true;
    }
    releaseCancellationInputs(call);
    call->finished.store(true, std::memory_order_release);
  });

  // Wait for the worker to enter Run, then request termination from this host thread.  The
  // bounded waits are intentional: if a provider ignores RunOptionsSetTerminate, leaving the
  // session pinned is safer than hanging Flame's UI.  The normal path joins and cleans up.
  const auto startWait = std::chrono::steady_clock::now();
  while (!call->started.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() - startWait < std::chrono::milliseconds(100))
    std::this_thread::yield();

  status = api->RunOptionsSetTerminate(call->options);
  const bool terminateRequested = !status;
  if (status)
    result.cancellationFailure = "RunOptionsSetTerminate: " + takeStatus(api, status);

  const auto finishWait = std::chrono::steady_clock::now();
  while (!call->finished.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() - finishWait < std::chrono::seconds(5))
    std::this_thread::yield();

  if (!call->finished.load(std::memory_order_acquire)) {
    worker.detach();
    result.cancellationTimedOut = true;
    result.cancellationState = ProbeState::Fail;
    result.cancellationFailure = "RunOptions termination did not return within 5 seconds; "
                                  "session resources intentionally retained";
    result.leakedResources = true;
    return;
  }
  worker.join();
  if (call->options)
    api->ReleaseRunOptions(call->options);
  call->options = nullptr;

  if (!terminateRequested) {
    result.cancellationState = ProbeState::Fail;
    if (result.cancellationFailure.empty())
      result.cancellationFailure = "termination request failed";
  } else if (call->terminationError) {
    result.cancellationObserved = true;
    result.cancellationState = ProbeState::Pass;
    emitf("    cross-thread cancellation: observed provider termination (%s)",
          call->message.c_str());
  } else if (call->runSucceeded) {
    // The model completed before the cancellation landed.  This is an attempted check, but
    // not evidence that cancellation works or fails, so keep it visibly NOT TESTED.
    result.cancellationState = ProbeState::NotTested;
    result.cancellationFailure = "run completed before termination was observed";
    emit("    cross-thread cancellation: NOT TESTED (run won the race)");
  } else {
    result.cancellationState = ProbeState::Fail;
    result.cancellationFailure = call->message.empty() ? "Run failed without a termination "
                                                          "message"
                                                        : call->message;
  }
}

FlowRun runSeaRaft(const OrtApi *api, OrtEnv *env, const std::string &path, bool cuda,
                   bool extended = true, VramMonitor *vramMonitor = nullptr) {
  FlowRun result;
  if (cuda && vramMonitor)
    sampleVram(vramMonitor, result.vram, "before session");
  OrtSessionOptions *options = nullptr;
  OrtStatus *status = api->CreateSessionOptions(&options);
  if (status) {
    result.failure = "CreateSessionOptions: " + takeStatus(api, status);
    return result;
  }
  status = api->SetIntraOpNumThreads(options, 1);
  if (status) {
    result.failure = "SetIntraOpNumThreads: " + takeStatus(api, status);
    api->ReleaseSessionOptions(options);
    return result;
  }
  if (cuda) {
    OrtCUDAProviderOptions cudaOptions{};
    cudaOptions.device_id = 0;
    cudaOptions.arena_extend_strategy = 0;
    cudaOptions.gpu_mem_limit = std::numeric_limits<std::size_t>::max();
    cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cudaOptions.do_copy_in_default_stream = 1;
    status = api->SessionOptionsAppendExecutionProvider_CUDA(options, &cudaOptions);
    if (status) {
      result.failure = "append CUDA provider: " + takeStatus(api, status);
      api->ReleaseSessionOptions(options);
      return result;
    }
  }

  OrtSession *session = nullptr;
  const auto sessionStart = std::chrono::steady_clock::now();
  status = api->CreateSession(env, path.c_str(), options, &session);
  result.sessionMilliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - sessionStart)
          .count();
  api->ReleaseSessionOptions(options);
  if (status) {
    result.failure = "CreateSession: " + takeStatus(api, status);
    return result;
  }
  if (cuda && vramMonitor)
    sampleVram(vramMonitor, result.vram, "after session");

  OrtMemoryInfo *memory = nullptr;
  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory);
  if (status) {
    result.failure = "CreateCpuMemoryInfo: " + takeStatus(api, status);
    api->ReleaseSession(session);
    return result;
  }

  constexpr int height = 128;
  constexpr int width = 192;
  constexpr int dx = 4;
  const std::vector<float> first = makeTexture(height, width);
  const std::vector<float> second = translateRight(first, height, width, dx);
  std::vector<float> identity, forward, reverse;
  double ignored = 0.0;
  if (!runFlowOnce(api, session, memory, first, first, height, width, identity,
                   result.firstRunMilliseconds, result.failure) ||
      !runFlowOnce(api, session, memory, first, second, height, width, forward, ignored,
                   result.failure) ||
      !runFlowOnce(api, session, memory, second, first, height, width, reverse, ignored,
                   result.failure)) {
    api->ReleaseMemoryInfo(memory);
    api->ReleaseSession(session);
    return result;
  }
  if (cuda && vramMonitor)
    sampleVram(vramMonitor, result.vram, "after first run");

  constexpr int border = 16;
  const std::size_t plane = static_cast<std::size_t>(height) * width;
  std::vector<float> identityEpe, forwardX, forwardY, reverseX, reverseY;
  identityEpe.reserve((height - 2 * border) * (width - 2 * border));
  for (int y = border; y < height - border; ++y) {
    for (int x = border; x < width - border; ++x) {
      const std::size_t offset = static_cast<std::size_t>(y) * width + x;
      identityEpe.push_back(std::sqrt(identity[offset] * identity[offset] +
                                     identity[plane + offset] * identity[plane + offset]));
      forwardX.push_back(forward[offset]);
      forwardY.push_back(forward[plane + offset]);
      reverseX.push_back(reverse[offset]);
      reverseY.push_back(reverse[plane + offset]);
    }
  }
  result.identityMedianEpe = median(std::move(identityEpe));
  result.forwardMedianX = median(std::move(forwardX));
  result.forwardMedianY = median(std::move(forwardY));
  result.reverseMedianX = median(std::move(reverseX));
  result.reverseMedianY = median(std::move(reverseY));
  result.ran = true;
  result.correct = result.identityMedianEpe <= 0.75 && result.forwardMedianX >= 2.0 &&
                   result.reverseMedianX <= -2.0 && std::abs(result.forwardMedianY) <= 2.0 &&
                   std::abs(result.reverseMedianY) <= 2.0;
  if (!result.correct)
    result.failure = "inference ran but identity/direction thresholds failed";
  if (extended && result.ran && result.correct) {
    result.timings = runUsefulTimings(api, session, memory, vramMonitor, &result.vram);
    for (const FlowRun::TimingResult &timing : result.timings)
      emitf("    %s %dx%d warm %.1f ms | steady median %.1f ms (min %.1f, max %.1f; %d samples): %s",
            cuda ? "CUDA" : "CPU", timing.height, timing.width, timing.warmMilliseconds,
            timing.steadyMedianMilliseconds, timing.steadyMinimumMilliseconds,
            timing.steadyMaximumMilliseconds, timing.steadySamples, probeState(timing.state));
    if (cuda && vramMonitor)
      sampleVram(vramMonitor, result.vram, "steady");
    runCancellation(api, session, memory, result);
  }
  if (!result.leakedResources) {
    api->ReleaseMemoryInfo(memory);
    api->ReleaseSession(session);
  }
  return result;
}

LifecycleResult runLifecycle(const OrtApi *api, OrtEnv *env, const std::string &path, bool cuda) {
  LifecycleResult result;
  result.attempted = true;
  result.requested = 3;
  for (int iteration = 0; iteration < result.requested; ++iteration) {
    FlowRun run = runSeaRaft(api, env, path, cuda, false, nullptr);
    if (run.leakedResources) {
      result.failure = "lifecycle iteration timed out during cancellation-safe teardown";
      break;
    }
    if (!run.ran || !run.correct) {
      result.failure = run.failure.empty() ? "session did not pass identity/direction" : run.failure;
      break;
    }
    ++result.completed;
  }
  result.state = stateFor(result.attempted, result.completed == result.requested);
  emitf("    repeated create/run/destroy (%s): %d/%d completed: %s", cuda ? "CUDA" : "CPU",
        result.completed, result.requested, probeState(result.state));
  if (!result.failure.empty())
    emitf("      lifecycle detail: %s", result.failure.c_str());
  return result;
}

FallbackResult exerciseProviderFailureFallback(const OrtApi *api, OrtEnv *env,
                                               const std::string &path) {
  FallbackResult result;
  result.attempted = true;
  OrtSessionOptions *options = nullptr;
  OrtStatus *status = api->CreateSessionOptions(&options);
  if (status) {
    result.failure = "CreateSessionOptions: " + takeStatus(api, status);
    result.state = ProbeState::Fail;
    return result;
  }
  status = api->SetIntraOpNumThreads(options, 1);
  if (status) {
    result.failure = "SetIntraOpNumThreads: " + takeStatus(api, status);
    api->ReleaseSessionOptions(options);
    result.state = ProbeState::Fail;
    return result;
  }

  // INT_MAX is a deterministic invalid device selector.  It fails at provider append or
  // session creation without allocating a large arena, unlike trying to consume the remaining
  // device memory.  The CPU path below is the fallback we need to prove.
  OrtCUDAProviderOptions invalidCuda{};
  invalidCuda.device_id = INT_MAX;
  invalidCuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  status = api->SessionOptionsAppendExecutionProvider_CUDA(options, &invalidCuda);
  if (status) {
    result.providerFailureObserved = true;
    result.failure = "invalid CUDA device rejected at provider init: " + takeStatus(api, status);
  } else {
    OrtSession *invalidSession = nullptr;
    status = api->CreateSession(env, path.c_str(), options, &invalidSession);
    if (status) {
      result.providerFailureObserved = true;
      result.failure = "invalid CUDA device rejected at session init: " + takeStatus(api, status);
    } else {
      // A provider that accepts INT_MAX is not a safe failure injection; do not run it.  It is
      // released and the check is reported as failed rather than silently calling CPU.
      result.failure = "invalid CUDA device was accepted; provider failure was not injected";
      api->ReleaseSession(invalidSession);
    }
  }
  api->ReleaseSessionOptions(options);

  FlowRun fallback = runSeaRaft(api, env, path, false, false, nullptr);
  result.cpuFallbackRan = fallback.ran && fallback.correct;
  if (!result.providerFailureObserved) {
    result.state = ProbeState::Fail;
  } else if (result.cpuFallbackRan) {
    result.state = ProbeState::Pass;
  } else {
    result.state = ProbeState::Fail;
    if (!fallback.failure.empty())
      result.failure += "; CPU fallback: " + fallback.failure;
  }
  emitf("    provider-init failure + CPU fallback: injected %s | CPU fallback %s | %s",
        yesNo(result.providerFailureObserved), yesNo(result.cpuFallbackRan),
        probeState(result.state));
  if (!result.failure.empty())
    emitf("      fallback detail: %s", result.failure.c_str());
  return result;
}

ArenaLimitResult exerciseGpuArenaLimitRecovery(const OrtApi *api, OrtEnv *env,
                                               const std::string &path,
                                               VramMonitor *vramMonitor) {
  // gpu_mem_limit bounds ONNX Runtime's CUDA arena, not every CUDA/cuDNN allocation in the
  // process.  This is therefore a controlled arena-limit exercise rather than a device-wide
  // allocation fence.  NVML samples before and after make unexpected physical pressure
  // visible in the report.  A qualifying result must be a recognisable allocator-limit
  // diagnostic from CreateSession or Run; provider setup failures do not count.
  constexpr std::size_t kGpuMemoryLimitBytes = 64u * 1024u * 1024u;
  constexpr int kTargetHeight = 480;
  constexpr int kTargetWidth = 640;

  ArenaLimitResult result;
  result.attempted = true;
  result.gpuMemoryLimitBytes = kGpuMemoryLimitBytes;
  result.targetHeight = kTargetHeight;
  result.targetWidth = kTargetWidth;

  OrtSessionOptions *options = nullptr;
  OrtSession *limitedSession = nullptr;
  OrtMemoryInfo *memory = nullptr;
  bool setupReady = true;

  result.vramBeforeAvailable = sampleVramUsed(vramMonitor, result.vramBeforeBytes);

  result.failureStage = ArenaLimitFailureStage::CreateSessionOptions;
  OrtStatus *status = api->CreateSessionOptions(&options);
  if (status) {
    result.failure = "CreateSessionOptions: " + takeStatus(api, status);
    result.failureKind = ArenaLimitFailureKind::Other;
    setupReady = false;
  }
  if (setupReady) {
    result.failureStage = ArenaLimitFailureStage::ConfigureSessionOptions;
    status = api->SetIntraOpNumThreads(options, 1);
    if (status) {
      result.failure = "SetIntraOpNumThreads: " + takeStatus(api, status);
      result.failureKind = ArenaLimitFailureKind::Other;
      setupReady = false;
    }
  }
  if (setupReady) {
    OrtCUDAProviderOptions limitedCuda{};
    limitedCuda.device_id = 0;
    limitedCuda.arena_extend_strategy = 1; // kSameAsRequested: do not round growth upward.
    limitedCuda.gpu_mem_limit = kGpuMemoryLimitBytes;
    limitedCuda.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
    limitedCuda.do_copy_in_default_stream = 1;
    result.failureStage = ArenaLimitFailureStage::AppendProvider;
    status = api->SessionOptionsAppendExecutionProvider_CUDA(options, &limitedCuda);
    if (status) {
      result.failure = "append limited CUDA provider: " + takeStatus(api, status);
      result.failureKind = ArenaLimitFailureKind::ProviderInit;
      setupReady = false;
    }
  }
  if (setupReady) {
    result.failureStage = ArenaLimitFailureStage::CreateSession;
    status = api->CreateSession(env, path.c_str(), options, &limitedSession);
    if (status) {
      const std::string detail = takeStatus(api, status);
      result.allocatorFailureObserved = looksLikeExplicitCudaArenaLimitFailure(detail);
      result.failureKind = result.allocatorFailureObserved
                               ? ArenaLimitFailureKind::AllocatorLimit
                               : ArenaLimitFailureKind::Other;
      result.failure = "limited CUDA CreateSession: " + detail;
      setupReady = false;
    }
  }
  if (setupReady) {
    result.failureStage = ArenaLimitFailureStage::CreateMemoryInfo;
    status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory);
    if (status) {
      result.failure = "CreateCpuMemoryInfo for limited CUDA run: " + takeStatus(api, status);
      result.failureKind = ArenaLimitFailureKind::Other;
      setupReady = false;
    }
  }
  if (setupReady) {
    result.failureStage = ArenaLimitFailureStage::Run;
    const std::vector<float> first = makeTexture(kTargetHeight, kTargetWidth);
    const std::vector<float> second =
        translateRight(first, kTargetHeight, kTargetWidth, 4);
    std::vector<float> flow;
    double milliseconds = 0.0;
    std::string detail;
    if (!runFlowOnce(api, limitedSession, memory, first, second, kTargetHeight, kTargetWidth,
                     flow, milliseconds, detail)) {
      result.allocatorFailureObserved = looksLikeExplicitCudaArenaLimitFailure(detail);
      result.failureKind = result.allocatorFailureObserved
                               ? ArenaLimitFailureKind::AllocatorLimit
                               : ArenaLimitFailureKind::Other;
      result.failure = "limited CUDA Run: " + detail;
    } else {
      result.failureKind = ArenaLimitFailureKind::UnexpectedSuccess;
      result.failure = "limited CUDA Run unexpectedly succeeded; gpu_mem_limit did not "
                       "produce the required allocator failure";
    }
  }

  if (memory)
    api->ReleaseMemoryInfo(memory);
  if (limitedSession)
    api->ReleaseSession(limitedSession);
  if (options)
    api->ReleaseSessionOptions(options);

  result.vramAfterAvailable = sampleVramUsed(vramMonitor, result.vramAfterBytes);

  // This is process/session recovery, not an automatic production fallback path.  It starts
  // a fresh CPU session after the limited CUDA objects are gone and repeats the real model's
  // 128x192 identity/direction checks as numerical evidence that recovery remains possible.
  FlowRun recovery = runSeaRaft(api, env, path, false, false, nullptr);
  result.cpuRecoveryRan = recovery.ran && recovery.correct;
  if (result.allocatorFailureObserved && result.cpuRecoveryRan) {
    result.state = ProbeState::Pass;
  } else {
    result.state = ProbeState::Fail;
    if (!recovery.failure.empty())
      result.failure += "; CPU recovery: " + recovery.failure;
  }

  emitf("    controlled CUDA arena limit + CPU recovery: gpu_mem_limit %.1f MiB | "
        "target %dx%d | stage %s | kind %s | allocator failure %s | CPU recovery %s | %s",
        static_cast<double>(result.gpuMemoryLimitBytes) / (1024.0 * 1024.0),
        result.targetHeight, result.targetWidth, arenaLimitFailureStage(result.failureStage),
        arenaLimitFailureKind(result.failureKind), yesNo(result.allocatorFailureObserved),
        yesNo(result.cpuRecoveryRan), probeState(result.state));
  if (result.vramBeforeAvailable && result.vramAfterAvailable)
    emitf("      device VRAM used before %.1f MiB | after cleanup %.1f MiB (NVML, device-wide)",
          static_cast<double>(result.vramBeforeBytes) / (1024.0 * 1024.0),
          static_cast<double>(result.vramAfterBytes) / (1024.0 * 1024.0));
  else
    emit("      device VRAM before/after: NOT TESTED (NVML unavailable)");
  if (!result.failure.empty())
    emitf("      arena-limit detail: %s", result.failure.c_str());
  return result;
}

bool providerAvailable(const OrtApi *api, const char *wanted) {
  char **providers = nullptr;
  int count = 0;
  OrtStatus *status = api->GetAvailableProviders(&providers, &count);
  if (status) {
    emitf("  GetAvailableProviders failed: %s", takeStatus(api, status).c_str());
    return false;
  }
  bool found = false;
  emit("  available execution providers:");
  for (int i = 0; i < count; ++i) {
    emitf("    %s", providers[i]);
    found = found || std::strcmp(providers[i], wanted) == 0;
  }
  if (OrtStatus *releaseStatus = api->ReleaseAvailableProviders(providers, count))
    emitf("  ReleaseAvailableProviders failed: %s", takeStatus(api, releaseStatus).c_str());
  return found;
}

#if defined(__linux__)
int collectAccelerationLibrary(struct dl_phdr_info *info, std::size_t, void *opaque) {
  if (!info->dlpi_name || !info->dlpi_name[0])
    return 0;
  const std::string path = info->dlpi_name;
  // Keep this list based on library families, not one release's exact SONAME.  CUDA EP
  // builds commonly pull in curand/cufft/nvrtc/nvjitlink and the original probe silently
  // omitted curand, which made its dependency report incomplete.
  if (path.find("onnxruntime_providers") != std::string::npos ||
      path.find("libcuda") != std::string::npos ||
      path.find("libcudart") != std::string::npos ||
      path.find("libcudnn") != std::string::npos ||
      path.find("libcublas") != std::string::npos ||
      path.find("libcurand") != std::string::npos ||
      path.find("libcufft") != std::string::npos ||
      path.find("libnvrtc") != std::string::npos ||
      path.find("libnvjitlink") != std::string::npos ||
      path.find("libnvjpeg") != std::string::npos ||
      path.find("libnvinfer") != std::string::npos ||
      path.find("libnvonnxparser") != std::string::npos ||
      path.find("libnccl") != std::string::npos ||
      path.find("libnvToolsExt") != std::string::npos)
    static_cast<std::vector<std::string> *>(opaque)->push_back(path);
  return 0;
}

void reportAccelerationLibraries(const char *when) {
  std::vector<std::string> libraries;
  dl_iterate_phdr(collectAccelerationLibrary, &libraries);
  std::sort(libraries.begin(), libraries.end());
  libraries.erase(std::unique(libraries.begin(), libraries.end()), libraries.end());
  emitf("    mapped CUDA/provider libraries %s:", when);
  if (libraries.empty()) {
    emit("      <none>");
    return;
  }
  std::uint64_t totalBytes = 0;
  for (const std::string &library : libraries) {
    struct stat fileStat = {};
    if (::stat(library.c_str(), &fileStat) == 0 && fileStat.st_size >= 0) {
      totalBytes += static_cast<std::uint64_t>(fileStat.st_size);
      emitf("      %s | on-disk bytes %lld", library.c_str(),
            static_cast<long long>(fileStat.st_size));
    } else {
      const int error = errno;
      emitf("      %s | on-disk size unavailable: errno=%d (%s)", library.c_str(), error,
            errnoDescription(error).c_str());
    }
  }
  emitf("    mapped CUDA/provider file total: %llu bytes (%.1f MiB)",
        static_cast<unsigned long long>(totalBytes),
        static_cast<double>(totalBytes) / (1024.0 * 1024.0));
}
#else
void reportAccelerationLibraries(const char *) {}
#endif

ModeResult runMode(const Mode &mode, const std::string &libraryBase, const void *hostApiBase,
                   bool runExtendedMeasurements) {
  ModeResult result;

  const std::string library = libraryBase + mode.librarySuffix;

  section(mode.name);
  emitf("  %s", mode.rationale);
  emitf("  dlopen(\"%s\", 0x%x)", library.c_str(), mode.flags);
  if (access(library.c_str(), R_OK) != 0) {
    const int error = errno;
    emitf("  runtime access check failed: errno=%d (%s)", error,
          errnoDescription(error).c_str());
  }

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

  if (mode.runRealModel && result.arithmeticCorrect) {
    const std::string seaRaftPath = modelPath();
    emit("");
    emit("  SEA-RAFT M real-network probe (selected plain RTLD_LOCAL mode)");
    emitf("    model: %s", seaRaftPath.c_str());
    result.realModelChecked = true;
    if (access(seaRaftPath.c_str(), R_OK) != 0) {
      const int error = errno;
      emitf("    NOT TESTED: model is not readable: errno=%d (%s)", error,
            errnoDescription(error).c_str());
      api->ReleaseEnv(env);
      emit("  teardown: ok");
      emit("  (handle intentionally not dlclosed -- see the note in the source)");
      return result;
    }
    result.realModelConfigured = true;
    emitf("    model readable: yes (bytes checked below by ORT; path access errno diagnostics "
          "are enabled)");
    result.cpuFlow =
        runSeaRaft(api, env, seaRaftPath, false, runExtendedMeasurements, nullptr);
    result.cpuFlowAvailable = true;
    const FlowRun &cpu = result.cpuFlow;
    result.realModelRan = cpu.ran;
    result.realModelCorrect = cpu.correct;
    if (cpu.ran) {
      emitf("    CPU session %.1f ms | first run %.1f ms", cpu.sessionMilliseconds,
            cpu.firstRunMilliseconds);
      emitf("    CPU identity median EPE %.4f px", cpu.identityMedianEpe);
      emitf("    CPU forward median (%.4f, %.4f) | reverse (%.4f, %.4f)",
            cpu.forwardMedianX, cpu.forwardMedianY, cpu.reverseMedianX, cpu.reverseMedianY);
      emitf("    CPU direction/identity: %s", cpu.correct ? "CORRECT" : "FAILED");
    } else {
      emitf("    CPU FAILED: %s", cpu.failure.c_str());
    }
    if (!cpu.failure.empty() && result.failure.empty())
      result.failure = "SEA-RAFT CPU: " + cpu.failure;

    if (cpu.leakedResources) {
      const char *reason = "NOT TESTED because the CPU cancellation probe retained live "
                           "resources";
      result.cpuLifecycle.failure = reason;
      result.cudaLifecycle.failure = reason;
      result.providerFallback.failure = reason;
      result.gpuArenaLimit.failure = reason;
      emitf("    CUDA and follow-on lifecycle/recovery checks: %s", reason);
    } else {
      result.cudaProviderChecked = true;
      result.cudaAvailable = providerAvailable(api, "CUDAExecutionProvider");
      if (result.cudaAvailable) {
        VramMonitor vram;
        vram.initialize();
        reportAccelerationLibraries("before CUDA session creation");
        result.cudaFlow =
            runSeaRaft(api, env, seaRaftPath, true, runExtendedMeasurements, &vram);
        result.cudaFlowAvailable = true;
        const FlowRun &cuda = result.cudaFlow;
        reportAccelerationLibraries("after CUDA session/run/teardown");
        result.cudaRan = cuda.ran;
        result.cudaCorrect = cuda.correct;
        if (cuda.ran) {
          emitf("    CUDA session %.1f ms | first run %.1f ms", cuda.sessionMilliseconds,
                cuda.firstRunMilliseconds);
          emitf("    CUDA identity median EPE %.4f px", cuda.identityMedianEpe);
          emitf("    CUDA forward median (%.4f, %.4f) | reverse (%.4f, %.4f)",
                cuda.forwardMedianX, cuda.forwardMedianY, cuda.reverseMedianX,
                cuda.reverseMedianY);
          emitf("    CUDA direction/identity: %s", cuda.correct ? "CORRECT" : "FAILED");
        } else {
          emitf("    CUDA FAILED: %s", cuda.failure.c_str());
        }
        if (!cuda.failure.empty() && result.failure.empty())
          result.failure = "SEA-RAFT CUDA: " + cuda.failure;

        if (cuda.vram.attempted && !cuda.vram.available)
          emitf("    VRAM result: %s (%s)", probeState(cuda.vram.state),
                cuda.vram.failure.empty() ? "NVML unavailable" : cuda.vram.failure.c_str());
        else if (cuda.vram.hasSample) {
          const double observedDeltaMiB =
              (static_cast<double>(cuda.vram.peakUsedBytes) -
               static_cast<double>(cuda.vram.baselineUsedBytes)) /
              (1024.0 * 1024.0);
          emitf("    VRAM result: %s | baseline %.1f MiB | observed peak %.1f MiB | steady %.1f MiB | "
                "observed delta %.1f MiB",
                probeState(cuda.vram.state),
                static_cast<double>(cuda.vram.baselineUsedBytes) / (1024.0 * 1024.0),
                static_cast<double>(cuda.vram.peakUsedBytes) / (1024.0 * 1024.0),
                static_cast<double>(cuda.vram.steadyUsedBytes) / (1024.0 * 1024.0),
                observedDeltaMiB);
        }

        if (runExtendedMeasurements && cuda.ran && cuda.correct &&
            !cuda.leakedResources) {
          result.cudaLifecycle = runLifecycle(api, env, seaRaftPath, true);
          result.providerFallback = exerciseProviderFailureFallback(api, env, seaRaftPath);
          result.gpuArenaLimit =
              exerciseGpuArenaLimitRecovery(api, env, seaRaftPath, &vram);
        } else if (runExtendedMeasurements) {
          const char *reason = cuda.leakedResources
                                   ? "NOT TESTED because the CUDA cancellation probe retained "
                                     "live GPU resources"
                                   : "NOT TESTED because baseline CUDA inference did not pass";
          result.cudaLifecycle.failure = reason;
          result.providerFallback.failure = reason;
          result.gpuArenaLimit.failure = reason;
          if (cuda.leakedResources)
            result.cpuLifecycle.failure = reason;
          emitf("    CUDA lifecycle, provider-failure and arena-limit recovery: %s", reason);
        }
      } else {
        emit("    CUDAExecutionProvider not present in this runtime -- CUDA lifecycle, VRAM, "
             "provider-failure and arena-limit checks are NOT TESTED.");
      }
    }
    if (runExtendedMeasurements && !cpu.leakedResources &&
        !result.cudaFlow.leakedResources)
      result.cpuLifecycle = runLifecycle(api, env, seaRaftPath, false);
    else if (!runExtendedMeasurements)
      emit("    extended Phase 0B measurements disabled (set WHITEWATER_ORT_EXTENDED=1 to run)");
    result.nodeDuplicationState = ProbeState::NotTested;
    result.nodeDuplicationFailure =
        "duplicate the OFX node in Flame and run this probe on each copy; compare the "
        "instance handles, process counters and verdicts printed in both reports";
    if (!result.gpuArenaLimit.attempted)
      emit("    controlled CUDA arena limit + CPU recovery: NOT TESTED");
    emitf("    node duplication equivalence: %s (%s)", probeState(result.nodeDuplicationState),
          result.nodeDuplicationFailure.c_str());
  }

  bool keepEnvironment = result.cpuFlow.leakedResources || result.cudaFlow.leakedResources;
  if (!keepEnvironment)
    api->ReleaseEnv(env);
  else
    emit("  teardown: session cancellation timed out; environment intentionally retained");
  emit("  teardown: ok");

  // Deliberately left open. dlclose of a runtime that registered thread-local state and
  // atexit handlers inside a host process is its own hazard, and unloading it is not
  // something the shipping plugin would ever do -- so the probe does not do it either.
  emit("  (handle intentionally not dlclosed -- see the note in the source)");
  return result;
}

void emitFlowExtendedVerdict(const char *label, const FlowRun &flow, bool available) {
  if (!available) {
    emitf("      %s extended measurements: NOT TESTED (session did not initialise)", label);
    return;
  }
  emitf("      %s inference: %s | identity/direction: %s", label,
        probeState(stateFor(available, flow.ran)), probeState(stateFor(flow.ran, flow.correct)));
  if (!flow.timings.empty()) {
    for (const FlowRun::TimingResult &timing : flow.timings)
      emitf("      %s %dx%d warm %.1f ms | steady median %.1f ms | samples %d | %s", label,
            timing.height, timing.width, timing.warmMilliseconds,
            timing.steadyMedianMilliseconds, timing.steadySamples, probeState(timing.state));
  } else {
    emitf("      %s useful-size timing: NOT TESTED", label);
  }
  emitf("      %s cross-thread cancellation: attempted %s | %s", label,
        yesNo(flow.cancellationAttempted), probeState(flow.cancellationState));
  if (!flow.cancellationFailure.empty())
    emitf("        cancellation detail: %s", flow.cancellationFailure.c_str());
  if (flow.vram.attempted) {
    if (flow.vram.hasSample)
      emitf("      %s VRAM: attempted yes | %s | baseline %.1f MiB | observed peak %.1f MiB | steady %.1f MiB",
            label, probeState(flow.vram.state),
            static_cast<double>(flow.vram.baselineUsedBytes) / (1024.0 * 1024.0),
            static_cast<double>(flow.vram.peakUsedBytes) / (1024.0 * 1024.0),
            static_cast<double>(flow.vram.steadyUsedBytes) / (1024.0 * 1024.0));
    else
      emitf("      %s VRAM: attempted yes | %s (%s)", label, probeState(flow.vram.state),
            flow.vram.failure.empty() ? "no sample" : flow.vram.failure.c_str());
  } else {
    emitf("      %s VRAM: NOT TESTED (CPU path)", label);
  }
}

// ---------------------------------------------------------------------------
// The probe
// ---------------------------------------------------------------------------

bool runProbe(OfxImageEffectHandle instance) {
  section("White Water ONNX Runtime isolation probe");

  emitf("  plugin instance %p | instances created in process %llu | currently live %llu",
        instance, gCreatedInstances.load(std::memory_order_relaxed),
        gLiveInstances.load(std::memory_order_relaxed));

  bool runExtendedMeasurements = true;
  if (const char *extended = std::getenv("WHITEWATER_ORT_EXTENDED"))
    runExtendedMeasurements = std::strcmp(extended, "0") != 0;
  emitf("  extended Phase 0B measurements: %s",
        runExtendedMeasurements ? "enabled" : "disabled");

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
                   kRuntimeSuffixA, true});
#else
  // DEEPBIND first. If the two modes ever share a library again through some path this
  // code did not anticipate, the mode that gets genuinely measured should be the one still
  // in question -- not the one already answered.
  modes.push_back({"Mode 1: RTLD_LOCAL | RTLD_DEEPBIND", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND,
                   "DEEPBIND makes the library prefer its own symbols over the global scope. "
                   "Measured 2026-08-20: RTLD_LOCAL alone already sufficed in Flame, so what "
                   "this mode now decides is whether DEEPBIND is safe to use, not whether it "
                   "is necessary.",
                   kRuntimeSuffixB, false});
  modes.push_back({"Mode 2: RTLD_LOCAL only", RTLD_NOW | RTLD_LOCAL,
                   "RTLD_LOCAL keeps our symbols out of the global scope for later lookups; "
                   "it does not reorder how our own library's relocations resolve. Expected "
                   "to be insufficient on paper, and measured sufficient in Flame -- most "
                   "likely because ONNX Runtime is built with hidden visibility, so its "
                   "internals never consult the global scope and there is nothing to capture.",
                   kRuntimeSuffixA, true});
#endif

  std::vector<ModeResult> results;
  for (const Mode &mode : modes)
    results.push_back(runMode(mode, library, hostApiBase, runExtendedMeasurements));

  // --- Verdict -------------------------------------------------------------
  section("VERDICT");
  for (std::size_t i = 0; i < modes.size(); ++i) {
    const ModeResult &r = results[i];
    const bool usable = r.opened && r.distinctFromHost && r.ranInference &&
                        r.arithmeticCorrect;
    emitf("  %-52s %s", modes[i].name, usable ? "USABLE" : "NOT USABLE");
    emitf("      opened: attempted yes | %s", probeState(stateFor(true, r.opened)));
    emitf("      distinct pointer: attempted %s | %s", yesNo(r.opened),
          probeState(stateFor(r.opened, r.distinctFromHost)));
    emitf("      version: attempted %s | %s (%s)", yesNo(r.opened),
          probeState(stateFor(r.opened, r.versionReadable)),
          r.versionReadable ? r.version.c_str() : "unreadable");
    emitf("      Add inference: attempted %s | %s", yesNo(r.opened),
          probeState(stateFor(r.opened, r.ranInference)));
    emitf("      Add arithmetic: attempted %s | %s", yesNo(r.ranInference),
          probeState(stateFor(r.ranInference, r.arithmeticCorrect)));
    if (modes[i].runRealModel) {
      emitf("      SEA-RAFT model readability: attempted %s | %s",
            yesNo(r.realModelChecked),
            probeState(stateFor(r.realModelChecked, r.realModelConfigured)));
      emitf("      SEA-RAFT CPU inference: attempted %s | %s",
            yesNo(r.realModelConfigured),
            probeState(stateFor(r.realModelConfigured, r.realModelRan)));
      emitf("      CPU direction/identity: attempted %s | %s", yesNo(r.realModelRan),
            probeState(stateFor(r.realModelRan, r.realModelCorrect)));
      if (!r.cudaProviderChecked)
        emit("      CUDA provider availability: NOT TESTED");
      else
        emitf("      CUDA provider availability: %s", r.cudaAvailable ? "AVAILABLE" : "ABSENT");
      emitf("      CUDA direction/identity: attempted %s | %s", yesNo(r.cudaRan),
            probeState(stateFor(r.cudaRan, r.cudaCorrect)));
      emitFlowExtendedVerdict("CPU", r.cpuFlow, r.cpuFlowAvailable);
      emitFlowExtendedVerdict("CUDA", r.cudaFlow, r.cudaFlowAvailable);
      emitf("      CPU repeated lifecycle: attempted %s | %s (%d/%d)",
            yesNo(r.cpuLifecycle.attempted), probeState(r.cpuLifecycle.state),
            r.cpuLifecycle.completed, r.cpuLifecycle.requested);
      emitf("      CUDA repeated lifecycle: attempted %s | %s (%d/%d)",
            yesNo(r.cudaLifecycle.attempted), probeState(r.cudaLifecycle.state),
            r.cudaLifecycle.completed, r.cudaLifecycle.requested);
      emitf("      provider-init failure + CPU fallback: attempted %s | %s",
            yesNo(r.providerFallback.attempted), probeState(r.providerFallback.state));
      if (r.gpuArenaLimit.attempted)
        emitf("      controlled CUDA arena limit + CPU recovery: attempted yes | %s | "
              "gpu_mem_limit %.1f MiB | target %dx%d | stage %s | kind %s | "
              "allocator failure %s | CPU recovery %s",
              probeState(r.gpuArenaLimit.state),
              static_cast<double>(r.gpuArenaLimit.gpuMemoryLimitBytes) / (1024.0 * 1024.0),
              r.gpuArenaLimit.targetHeight, r.gpuArenaLimit.targetWidth,
              arenaLimitFailureStage(r.gpuArenaLimit.failureStage),
              arenaLimitFailureKind(r.gpuArenaLimit.failureKind),
              yesNo(r.gpuArenaLimit.allocatorFailureObserved),
              yesNo(r.gpuArenaLimit.cpuRecoveryRan));
      else
        emit("      controlled CUDA arena limit + CPU recovery: attempted no | NOT TESTED");
      if (r.gpuArenaLimit.vramBeforeAvailable && r.gpuArenaLimit.vramAfterAvailable)
        emitf("        device VRAM used before %.1f MiB | after cleanup %.1f MiB "
              "(NVML, device-wide)",
              static_cast<double>(r.gpuArenaLimit.vramBeforeBytes) / (1024.0 * 1024.0),
              static_cast<double>(r.gpuArenaLimit.vramAfterBytes) / (1024.0 * 1024.0));
      if (!r.gpuArenaLimit.failure.empty())
        emitf("        arena-limit detail: %s", r.gpuArenaLimit.failure.c_str());
      emitf("      node duplication equivalence: attempted %s | %s (%s)",
            r.nodeDuplicationState == ProbeState::NotTested ? "no" : "yes",
            probeState(r.nodeDuplicationState),
            r.nodeDuplicationFailure.empty() ? "requires manual Flame action"
                                              : r.nodeDuplicationFailure.c_str());
    }
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
  for (std::size_t i = 0; i < modes.size(); ++i) {
    if (!modes[i].runRealModel)
      continue;
    const ModeResult &r = results[i];
    emit("");
    if (!r.realModelConfigured) {
      emit("  Phase 0B real-network result: NOT TESTED (model unreadable or not configured; "
           "see the access diagnostic above)");
    } else {
      emitf("  Phase 0B CPU real-network result: attempted %s | %s", yesNo(r.realModelRan),
            probeState(stateFor(r.realModelRan, r.realModelCorrect)));
    }
    if (r.realModelConfigured && !r.cudaProviderChecked)
      emit("  Phase 0B CUDA result: NOT TESTED (provider check skipped; see mode detail)");
    else if (r.realModelConfigured && !r.cudaAvailable)
      emit("  Phase 0B CUDA result: NOT TESTED (CUDA EP absent from this runtime)");
    else if (r.realModelConfigured)
      emitf("  Phase 0B CUDA real-network result: attempted %s | %s", yesNo(r.cudaRan),
            probeState(stateFor(r.cudaRan, r.cudaCorrect)));
  }
  emit("");
  emitf("  Report: %s", reportPath().c_str());

  if (gMessage)
    gMessage->message(instance, kOfxMessageMessage, nullptr,
                      "White Water ORT probe written to:\n%s", reportPath().c_str());

  bool realModelPassed = false;
  for (std::size_t i = 0; i < modes.size(); ++i)
    if (modes[i].runRealModel)
      realModelPassed = results[i].realModelRan && results[i].realModelCorrect;
  const char *required = std::getenv("WHITEWATER_ORT_REQUIRE_REAL_MODEL");
  if (required && std::strcmp(required, "0") != 0 && !realModelPassed)
    return false;
  const char *requireLifecycle = std::getenv("WHITEWATER_ORT_REQUIRE_LIFECYCLE");
  if (requireLifecycle && std::strcmp(requireLifecycle, "0") != 0) {
    for (std::size_t i = 0; i < modes.size(); ++i) {
      if (modes[i].runRealModel && results[i].realModelConfigured &&
          results[i].cpuLifecycle.state != ProbeState::Pass)
        return false;
    }
  }
  const char *requireGpuMemLimit = std::getenv("WHITEWATER_ORT_REQUIRE_GPU_MEM_LIMIT");
  if (requireGpuMemLimit && std::strcmp(requireGpuMemLimit, "0") != 0) {
    bool arenaLimitPassed = false;
    for (std::size_t i = 0; i < modes.size(); ++i)
      if (modes[i].runRealModel)
        arenaLimitPassed = results[i].gpuArenaLimit.state == ProbeState::Pass;
    emitf("  Required CUDA arena-limit gate: %s", arenaLimitPassed ? "PASS" : "FAIL");
    if (!arenaLimitPassed)
      return false;
  }
  return true;
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
                         "Run isolation checks plus the pinned SEA-RAFT CPU/CUDA probe and "
                         "write the Phase 0B report.");
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
    return runProbe(instance) ? kOfxStatOK : kOfxStatFailed;
  }
  return kOfxStatReplyDefault;
}

OfxStatus onCreateInstance(OfxImageEffectHandle instance) {
  const unsigned long long created =
      gCreatedInstances.fetch_add(1, std::memory_order_relaxed) + 1;
  const unsigned long long live = gLiveInstances.fetch_add(1, std::memory_order_relaxed) + 1;
  emitf("ORT probe instance created: handle %p | created in process %llu | live %llu", instance,
        created, live);
  return kOfxStatOK;
}

OfxStatus onDestroyInstance(OfxImageEffectHandle instance) {
  const unsigned long long previous = gLiveInstances.fetch_sub(1, std::memory_order_relaxed);
  const unsigned long long live = previous > 0 ? previous - 1 : 0;
  if (previous == 0)
    gLiveInstances.store(0, std::memory_order_relaxed);
  emitf("ORT probe instance destroyed: handle %p | live %llu", instance, live);
  return kOfxStatOK;
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
    if (std::strcmp(action, kOfxActionCreateInstance) == 0)
      return onCreateInstance(effect);
    if (std::strcmp(action, kOfxActionDestroyInstance) == 0)
      return onDestroyInstance(effect);
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
