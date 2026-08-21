#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace whitewater::ortprobe {

// These are diagnostic guardrails for the disposable Phase 0B probe, not limits on the
// product.  The arena maximum is chosen for the current 24 GiB qualification GPU; a future
// artifact can raise it without changing any shipping interface.
constexpr std::uint64_t kHighResolutionMaximumPixels = 32u * 1024u * 1024u;
constexpr std::uint64_t kHighResolutionMaximumDimension = 8192u;
constexpr std::uint64_t kHighResolutionMinimumArenaMiB = 64u;
constexpr std::uint64_t kHighResolutionMaximumArenaMiB = 16384u;

struct HighResolutionConfig {
  bool enabled = false;
  bool valid = false;
  int sourceHeight = 0;
  int sourceWidth = 0;
  int tensorHeight = 0;
  int tensorWidth = 0;
  std::size_t gpuMemoryLimitBytes = 0;
  std::string failure;
};

enum class HighResolutionOutcome {
  None,
  InferencePass,
  BoundedAllocationStop,
  OtherFailure,
};

inline bool highResolutionMeasurementResultCollected(bool configValid, bool attempted,
                                                      bool telemetryComplete,
                                                      HighResolutionOutcome outcome) {
  return configValid && attempted && telemetryComplete &&
         (outcome == HighResolutionOutcome::InferencePass ||
          outcome == HighResolutionOutcome::BoundedAllocationStop);
}

inline bool parseUnsignedDecimal(const std::string &text, std::uint64_t &value) {
  if (text.empty())
    return false;
  std::uint64_t parsed = 0;
  for (const char character : text) {
    if (character < '0' || character > '9')
      return false;
    const std::uint64_t digit = static_cast<std::uint64_t>(character - '0');
    if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10u)
      return false;
    parsed = parsed * 10u + digit;
  }
  value = parsed;
  return true;
}

inline HighResolutionConfig parseHighResolutionConfig(const char *enabledText,
                                                       const char *sizeText,
                                                       const char *arenaMiBText) {
  HighResolutionConfig config;
  config.enabled = enabledText && enabledText[0] && std::string(enabledText) != "0";
  if (!config.enabled)
    return config;

  if (!sizeText || !sizeText[0]) {
    config.failure = "WHITEWATER_ORT_HIGHRES_SIZE is required in HxW order";
    return config;
  }
  const std::string size(sizeText);
  const std::size_t separator = size.find_first_of("xX");
  if (separator == std::string::npos || size.find_first_of("xX", separator + 1) !=
                                            std::string::npos) {
    config.failure = "WHITEWATER_ORT_HIGHRES_SIZE must contain exactly one x in HxW order";
    return config;
  }

  std::uint64_t height = 0;
  std::uint64_t width = 0;
  if (!parseUnsignedDecimal(size.substr(0, separator), height) ||
      !parseUnsignedDecimal(size.substr(separator + 1), width) || height == 0 || width == 0) {
    config.failure = "WHITEWATER_ORT_HIGHRES_SIZE must contain positive decimal HxW values";
    return config;
  }
  if (height > kHighResolutionMaximumDimension || width > kHighResolutionMaximumDimension) {
    config.failure = "requested size exceeds this diagnostic artifact's 8192-pixel dimension guard";
    return config;
  }
  const bool approvedTarget =
      (height == 2160u && width == 3840u) || (height == 2160u && width == 4096u) ||
      (height == 3164u && width == 4608u);
  if (!approvedTarget) {
    config.failure = "this artifact accepts one Phase 0B target: 2160x3840, 2160x4096, or 3164x4608";
    return config;
  }

  const std::uint64_t tensorHeight = (height + 7u) & ~std::uint64_t{7u};
  const std::uint64_t tensorWidth = (width + 7u) & ~std::uint64_t{7u};
  if (tensorHeight > kHighResolutionMaximumPixels / tensorWidth) {
    config.failure = "padded tensor exceeds this diagnostic artifact's 32-Mpixel safety guard";
    return config;
  }

  if (!arenaMiBText || !arenaMiBText[0]) {
    config.failure = "WHITEWATER_ORT_GPU_MEM_LIMIT_MIB is required in high-resolution mode";
    return config;
  }
  std::uint64_t arenaMiB = 0;
  if (!parseUnsignedDecimal(arenaMiBText, arenaMiB) ||
      arenaMiB < kHighResolutionMinimumArenaMiB ||
      arenaMiB > kHighResolutionMaximumArenaMiB) {
    config.failure = "WHITEWATER_ORT_GPU_MEM_LIMIT_MIB must be an integer from 64 through 16384";
    return config;
  }
  constexpr std::uint64_t bytesPerMiB = 1024u * 1024u;
  if (arenaMiB > std::numeric_limits<std::size_t>::max() / bytesPerMiB) {
    config.failure = "WHITEWATER_ORT_GPU_MEM_LIMIT_MIB overflows size_t";
    return config;
  }

  config.sourceHeight = static_cast<int>(height);
  config.sourceWidth = static_cast<int>(width);
  config.tensorHeight = static_cast<int>(tensorHeight);
  config.tensorWidth = static_cast<int>(tensorWidth);
  config.gpuMemoryLimitBytes = static_cast<std::size_t>(arenaMiB * bytesPerMiB);
  config.valid = true;
  return config;
}

} // namespace whitewater::ortprobe
