// Evaluation-only ONNX Runtime C API bridge.
//
// This file is deliberately outside src/infer and the shipping plugin.  P25-5 carries the
// official ONNX Runtime 1.29.0 CUDA-12 C/C++ archive in its relocated conda environment and
// uses this small C ABI from Python.  Keeping the provider/session boundary here avoids making
// the air-gapped evaluator depend on a PyPI wheel whose 1.29.0 build is CUDA 13.

#include "onnxruntime_c_api.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#if defined(__GNUC__)
#define WW_ORT_EXPORT __attribute__((visibility("default")))
#else
#define WW_ORT_EXPORT
#endif

namespace {

thread_local std::string last_error;
thread_local std::string scratch;

// This library is called through ctypes.  Keep C++ exceptions inside the bridge, including
// allocation failures from metadata formatting or session bookkeeping.  Error reporting is
// best-effort so an allocation failure while recording an error cannot throw again.
void set_error(const char* message) noexcept {
  try {
    last_error = message == nullptr ? "native ONNX Runtime bridge failure" : message;
  } catch (...) {
  }
}

void set_error(const std::string& message) noexcept {
  try {
    last_error = message;
  } catch (...) {
  }
}

const OrtApi* get_api() {
  static const OrtApi* api = []() -> const OrtApi* {
    const OrtApiBase* base = OrtGetApiBase();
    return base == nullptr ? nullptr : base->GetApi(ORT_API_VERSION);
  }();
  return api;
}

bool fail(const std::string& message) noexcept {
  set_error(message);
  return false;
}

template <typename Function, typename Result>
Result guarded(Function&& function, Result fallback) noexcept {
  try {
    return function();
  } catch (const std::exception& exception) {
    set_error(exception.what());
    return fallback;
  } catch (...) {
    set_error("unknown C++ exception in native ONNX Runtime bridge");
    return fallback;
  }
}

template <typename Function>
void guarded_void(Function&& function) noexcept {
  try {
    function();
  } catch (const std::exception& exception) {
    set_error(exception.what());
  } catch (...) {
    set_error("unknown C++ exception in native ONNX Runtime bridge");
  }
}

bool check_status(OrtStatus* status, const char* operation) {
  if (status == nullptr) {
    return true;
  }
  const OrtApi* api = get_api();
  const char* detail = api == nullptr ? nullptr : api->GetErrorMessage(status);
  std::string message = operation == nullptr ? "ONNX Runtime call failed" : operation;
  if (detail != nullptr && *detail != '\0') {
    message += ": ";
    message += detail;
  }
  if (api != nullptr) {
    api->ReleaseStatus(status);
  }
  return fail(message);
}

bool check_api() {
  if (get_api() == nullptr) {
    return fail("ONNX Runtime 1.29 C API is unavailable");
  }
  return true;
}

std::string quote_json(const std::string& value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20) {
          out << "\\u00";
          const char* hex = "0123456789abcdef";
          out << hex[(character >> 4) & 0xf] << hex[character & 0xf];
        } else {
          out << static_cast<char>(character);
        }
        break;
    }
  }
  out << '"';
  return out.str();
}

struct ww_ort_session {
  OrtEnv* env = nullptr;
  OrtSession* session = nullptr;
  // Canonical Python-runtime provider spelling, not the internal construction token.
  const char* provider = nullptr;
};

std::mutex env_mutex;
OrtEnv* shared_env = nullptr;

OrtEnv* get_env() {
  const OrtApi* api = get_api();
  std::lock_guard<std::mutex> lock(env_mutex);
  if (shared_env != nullptr) {
    return shared_env;
  }
  OrtEnv* env = nullptr;
  if (!check_status(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "whitewater-p25-5", &env), "CreateEnv")) {
    return nullptr;
  }
  // The evaluator must not generate telemetry files in an air-gapped runtime.  A failure here
  // is intentionally fatal rather than silently changing the runtime's identity.
  if (!check_status(api->DisableTelemetryEvents(env), "DisableTelemetryEvents")) {
    api->ReleaseEnv(env);
    return nullptr;
  }
  shared_env = env;
  return shared_env;
}

const char* element_type_name(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "tensor(float)";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "tensor(double)";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "tensor(float16)";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "tensor(int64)";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "tensor(int32)";
    default: return "tensor(unknown)";
  }
}

bool append_shape(std::ostringstream& out, const OrtTensorTypeAndShapeInfo* tensor_info) {
  const OrtApi* api = get_api();
  size_t rank = 0;
  if (!check_status(api->GetDimensionsCount(tensor_info, &rank), "GetDimensionsCount")) {
    return false;
  }
  std::vector<int64_t> dimensions(rank);
  if (!check_status(api->GetDimensions(tensor_info, dimensions.data(), rank), "GetDimensions")) {
    return false;
  }
  std::vector<const char*> symbolic(rank, nullptr);
  // Symbolic names are advisory.  Some older provider/type-info combinations do not expose
  // them even though GetDimensions succeeds; represent those dimensions as a stable token.
  const bool symbols_ok = check_status(
      api->GetSymbolicDimensions(tensor_info, symbolic.data(), rank), "GetSymbolicDimensions");
  if (!symbols_ok) {
    last_error.clear();
  }
  out << '[';
  for (size_t index = 0; index < rank; ++index) {
    if (index != 0) {
      out << ',';
    }
    if (dimensions[index] >= 0) {
      out << dimensions[index];
    } else if (symbols_ok && symbolic[index] != nullptr && *symbolic[index] != '\0') {
      out << quote_json(symbolic[index]);
    } else {
      out << quote_json("dynamic");
    }
  }
  out << ']';
  return true;
}

bool append_tensor_meta(
    std::ostringstream& out,
    const OrtSession* session,
    size_t index,
    bool input,
    OrtAllocator* allocator) {
  const OrtApi* api = get_api();
  char* name = nullptr;
  OrtTypeInfo* type_info = nullptr;
  const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
  OrtStatus* status = nullptr;
  if (input) {
    status = api->SessionGetInputName(session, index, allocator, &name);
  } else {
    status = api->SessionGetOutputName(session, index, allocator, &name);
  }
  if (!check_status(status, input ? "SessionGetInputName" : "SessionGetOutputName")) {
    return false;
  }
  const bool name_ok = name != nullptr;
  std::string name_copy = name_ok ? std::string(name) : std::string();
  if (name != nullptr) {
    allocator->Free(allocator, name);
  }
  if (!name_ok) {
    return fail("ONNX Runtime returned an empty tensor name");
  }
  if (input) {
    status = api->SessionGetInputTypeInfo(session, index, &type_info);
  } else {
    status = api->SessionGetOutputTypeInfo(session, index, &type_info);
  }
  if (!check_status(status, input ? "SessionGetInputTypeInfo" : "SessionGetOutputTypeInfo")) {
    return false;
  }
  if (!check_status(api->CastTypeInfoToTensorInfo(type_info, &tensor_info), "CastTypeInfoToTensorInfo")) {
    api->ReleaseTypeInfo(type_info);
    return false;
  }
  if (tensor_info == nullptr) {
    api->ReleaseTypeInfo(type_info);
    return fail("ONNX Runtime tensor metadata is not a tensor");
  }
  ONNXTensorElementDataType element_type{};
  if (!check_status(api->GetTensorElementType(tensor_info, &element_type), "GetTensorElementType")) {
    api->ReleaseTypeInfo(type_info);
    return false;
  }
  out << "{\"name\":" << quote_json(name_copy)
      << ",\"type\":" << quote_json(element_type_name(element_type))
      << ",\"shape\":";
  const bool shape_ok = append_shape(out, tensor_info);
  out << '}';
  api->ReleaseTypeInfo(type_info);
  return shape_ok;
}

}  // namespace

extern "C" {

WW_ORT_EXPORT const char* ww_ort_last_error() noexcept {
  return last_error.c_str();
}

WW_ORT_EXPORT const char* ww_ort_version() noexcept {
  return guarded([&]() -> const char* {
    if (!check_api()) {
      return "";
    }
    return OrtGetApiBase()->GetVersionString();
  }, "");
}

WW_ORT_EXPORT const char* ww_ort_available_providers() noexcept {
  return guarded([&]() -> const char* {
    scratch.clear();
    if (!check_api()) {
      return "[]";
    }
    char** providers = nullptr;
    int provider_count = 0;
    if (!check_status(get_api()->GetAvailableProviders(&providers, &provider_count), "GetAvailableProviders")) {
      return "[]";
    }
    scratch = "[";
    for (int index = 0; index < provider_count; ++index) {
      if (index != 0) {
        scratch += ',';
      }
      scratch += quote_json(providers[index] == nullptr ? "" : providers[index]);
    }
    scratch += ']';
    check_status(get_api()->ReleaseAvailableProviders(providers, provider_count),
                 "ReleaseAvailableProviders");
    return scratch.c_str();
  }, "[]");
}

WW_ORT_EXPORT ww_ort_session* ww_ort_session_create(
    const char* model_path, const char* provider) noexcept {
  return guarded([&]() -> ww_ort_session* {
    if (!check_api()) {
      return nullptr;
    }
    if (model_path == nullptr || *model_path == '\0') {
      fail("model path is empty");
      return nullptr;
    }
    if (provider == nullptr || (std::strcmp(provider, "cpu") != 0 && std::strcmp(provider, "cuda") != 0)) {
      fail("provider must be 'cpu' or 'cuda'");
      return nullptr;
    }
    OrtEnv* env = get_env();
    if (env == nullptr) {
      return nullptr;
    }
    const OrtApi* api = get_api();
    OrtSessionOptions* options = nullptr;
    if (!check_status(api->CreateSessionOptions(&options), "CreateSessionOptions")) {
      return nullptr;
    }
    if (std::strcmp(provider, "cuda") == 0) {
    // The official 1.29.0 CUDA-12 archive carries the public C API header at include/
    // onnxruntime_c_api.h; it does not carry cuda_provider_factory.h.  Use the versioned API
    // table entry and its public options struct, which loads the adjacent CUDA provider shared
    // object and returns a status if CUDA/cuDNN is absent.  Keep ORT's ordinary lower-priority
    // CPU fallback: SEA-RAFT has shape/housekeeping nodes that ORT intentionally assigns there,
    // while appending CUDA first keeps it the requested execution provider.
      OrtCUDAProviderOptions cuda_options{};
      cuda_options.device_id = 0;
      if (!check_status(api->SessionOptionsAppendExecutionProvider_CUDA(options, &cuda_options),
                        "SessionOptionsAppendExecutionProvider_CUDA")) {
        api->ReleaseSessionOptions(options);
        return nullptr;
      }
    }
    OrtSession* session = nullptr;
    if (!check_status(api->CreateSession(env, model_path, options, &session), "CreateSession")) {
      api->ReleaseSessionOptions(options);
      return nullptr;
    }
    api->ReleaseSessionOptions(options);
    auto* result = new (std::nothrow) ww_ort_session;
    if (result == nullptr) {
      api->ReleaseSession(session);
      fail("could not allocate native ONNX Runtime session handle");
      return nullptr;
    }
    result->env = env;
    result->session = session;
    result->provider = std::strcmp(provider, "cpu") == 0
                           ? "CPUExecutionProvider"
                           : "CUDAExecutionProvider";
    last_error.clear();
    return result;
  }, static_cast<ww_ort_session*>(nullptr));
}

WW_ORT_EXPORT void ww_ort_session_release(ww_ort_session* session) noexcept {
  guarded_void([&]() {
    if (session == nullptr) {
      return;
    }
    if (session->session != nullptr && get_api() != nullptr) {
      get_api()->ReleaseSession(session->session);
    }
    delete session;
  });
}

WW_ORT_EXPORT const char* ww_ort_session_providers(const ww_ort_session* session) noexcept {
  return guarded([&]() -> const char* {
    scratch = "[]";
    if (session == nullptr) {
      fail("session is null");
      return scratch.c_str();
    }
    scratch = "[" + quote_json(session->provider) + "]";
    return scratch.c_str();
  }, "[]");
}

WW_ORT_EXPORT const char* ww_ort_session_metadata(const ww_ort_session* session) noexcept {
  return guarded([&]() -> const char* {
  scratch.clear();
  if (session == nullptr || session->session == nullptr) {
    fail("session is null");
    return "{}";
  }
  const OrtApi* api = get_api();
  OrtAllocator* allocator = nullptr;
  if (!check_status(api->GetAllocatorWithDefaultOptions(&allocator), "GetAllocatorWithDefaultOptions")) {
    return "{}";
  }
  size_t input_count = 0;
  size_t output_count = 0;
  if (!check_status(api->SessionGetInputCount(session->session, &input_count), "SessionGetInputCount") ||
      !check_status(api->SessionGetOutputCount(session->session, &output_count), "SessionGetOutputCount")) {
    return "{}";
  }
  std::ostringstream out;
  out << "{\"inputs\":[";
  for (size_t index = 0; index < input_count; ++index) {
    if (index != 0) {
      out << ',';
    }
    if (!append_tensor_meta(out, session->session, index, true, allocator)) {
      return "{}";
    }
  }
  out << "],\"outputs\":[";
  for (size_t index = 0; index < output_count; ++index) {
    if (index != 0) {
      out << ',';
    }
    if (!append_tensor_meta(out, session->session, index, false, allocator)) {
      return "{}";
    }
  }
  out << "]}";
  scratch = out.str();
  last_error.clear();
  return scratch.c_str();
  }, "{}");
}

WW_ORT_EXPORT int ww_ort_session_run(
    ww_ort_session* session,
    const float* first,
    const float* second,
    const int64_t* shape,
    size_t rank,
    float** output,
    size_t* output_count,
    int64_t* output_shape,
    size_t output_shape_capacity,
    size_t* output_rank) noexcept {
  return guarded([&]() -> int {
  if (session == nullptr || session->session == nullptr || first == nullptr || second == nullptr ||
      shape == nullptr || output == nullptr || output_count == nullptr || output_shape == nullptr ||
      output_rank == nullptr) {
    fail("invalid null argument to ww_ort_session_run");
    return 0;
  }
  if (rank == 0 || rank > 8 || output_shape_capacity < rank) {
    fail("input rank must be between one and eight and fit the output shape buffer");
    return 0;
  }
  size_t elements = 1;
  for (size_t index = 0; index < rank; ++index) {
    if (shape[index] <= 0 || static_cast<uint64_t>(shape[index]) >
        std::numeric_limits<size_t>::max() / elements) {
      fail("input shape contains an invalid or overflowing dimension");
      return 0;
    }
    elements *= static_cast<size_t>(shape[index]);
  }
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    fail("input tensor byte size overflows size_t");
    return 0;
  }

  const OrtApi* api = get_api();
  OrtMemoryInfo* memory_info = nullptr;
  OrtValue* first_value = nullptr;
  OrtValue* second_value = nullptr;
  OrtValue* output_value = nullptr;
  OrtTensorTypeAndShapeInfo* output_info = nullptr;
  auto release_inputs = [&]() {
    if (first_value != nullptr) api->ReleaseValue(first_value);
    if (second_value != nullptr) api->ReleaseValue(second_value);
    if (memory_info != nullptr) api->ReleaseMemoryInfo(memory_info);
  };
  if (!check_status(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info),
                    "CreateCpuMemoryInfo")) {
    return 0;
  }
  if (!check_status(api->CreateTensorWithDataAsOrtValue(
                        memory_info, const_cast<float*>(first), elements * sizeof(float), shape, rank,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &first_value),
                    "CreateTensorWithDataAsOrtValue(first)")) {
    release_inputs();
    return 0;
  }
  if (!check_status(api->CreateTensorWithDataAsOrtValue(
                        memory_info, const_cast<float*>(second), elements * sizeof(float), shape, rank,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &second_value),
                    "CreateTensorWithDataAsOrtValue(second)")) {
    release_inputs();
    return 0;
  }
  const char* input_names[] = {"image1", "image2"};
  const OrtValue* inputs[] = {first_value, second_value};
  const char* output_names[] = {"flow"};
  if (!check_status(api->Run(session->session, nullptr, input_names, inputs, 2, output_names, 1, &output_value),
                    "Run")) {
    release_inputs();
    return 0;
  }
  release_inputs();
  if (!check_status(api->GetTensorTypeAndShape(output_value, &output_info), "GetTensorTypeAndShape")) {
    api->ReleaseValue(output_value);
    return 0;
  }
  ONNXTensorElementDataType output_type{};
  if (!check_status(api->GetTensorElementType(output_info, &output_type), "GetTensorElementType(output)")) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    return 0;
  }
  if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    fail("SEA-RAFT output tensor is not float32");
    return 0;
  }
  size_t result_rank = 0;
  if (!check_status(api->GetDimensionsCount(output_info, &result_rank), "GetDimensionsCount(output)")) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    return 0;
  }
  if (result_rank > output_shape_capacity) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    fail("output rank exceeds the supplied shape buffer");
    return 0;
  }
  int64_t dimensions[8] = {};
  if (!check_status(api->GetDimensions(output_info, dimensions, result_rank), "GetDimensions(output)")) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    return 0;
  }
  for (size_t index = 0; index < result_rank; ++index) {
    if (dimensions[index] < 0) {
      api->ReleaseTensorTypeAndShapeInfo(output_info);
      api->ReleaseValue(output_value);
      fail("SEA-RAFT output tensor has a dynamic dimension after inference");
      return 0;
    }
  }
  size_t result_elements = 0;
  if (!check_status(api->GetTensorShapeElementCount(output_info, &result_elements),
                    "GetTensorShapeElementCount")) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    return 0;
  }
  if (result_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    fail("output tensor byte size overflows size_t");
    return 0;
  }
  void* result_data = nullptr;
  if (!check_status(api->GetTensorMutableData(output_value, &result_data), "GetTensorMutableData")) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    return 0;
  }
  if (result_elements != 0 && result_data == nullptr) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    fail("ONNX Runtime returned a null output tensor buffer");
    return 0;
  }
  auto* copied = new (std::nothrow) float[result_elements];
  if (copied == nullptr) {
    api->ReleaseTensorTypeAndShapeInfo(output_info);
    api->ReleaseValue(output_value);
    fail("could not allocate output tensor copy");
    return 0;
  }
  std::memcpy(copied, result_data, result_elements * sizeof(float));
  std::copy(dimensions, dimensions + result_rank, output_shape);
  *output = copied;
  *output_count = result_elements;
  *output_rank = result_rank;
  api->ReleaseTensorTypeAndShapeInfo(output_info);
  api->ReleaseValue(output_value);
  last_error.clear();
  return 1;
  }, 0);
}

WW_ORT_EXPORT void ww_ort_free(void* pointer) noexcept {
  guarded_void([&]() { delete[] static_cast<float*>(pointer); });
}

}  // extern "C"
