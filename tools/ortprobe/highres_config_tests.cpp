#include "highres_config.h"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}

void expectInvalid(const char *size, const char *limit, const char *message) {
  const auto config =
      whitewater::ortprobe::parseHighResolutionConfig("1", size, limit);
  check(config.enabled && !config.valid && !config.failure.empty(), message);
}

} // namespace

int main() {
  using whitewater::ortprobe::parseHighResolutionConfig;

  const auto disabled = parseHighResolutionConfig(nullptr, nullptr, nullptr);
  check(!disabled.enabled && !disabled.valid, "unset mode stays disabled");

  const auto uhd = parseHighResolutionConfig("1", "2160x3840", "16384");
  check(uhd.valid && uhd.sourceHeight == 2160 && uhd.sourceWidth == 3840,
        "UHD parses in HxW order");
  check(uhd.tensorHeight == 2160 && uhd.tensorWidth == 3840,
        "UHD needs no padding");
  check(uhd.gpuMemoryLimitBytes == std::size_t{16384} * 1024u * 1024u,
        "arena MiB converts to bytes");

  const auto dci = parseHighResolutionConfig("yes", "2160X4096", "4096");
  check(dci.valid && dci.tensorHeight == 2160 && dci.tensorWidth == 4096,
        "uppercase separator parses");

  const auto alexa = parseHighResolutionConfig("1", "3164x4608", "16384");
  check(alexa.valid && alexa.sourceHeight == 3164 && alexa.sourceWidth == 4608,
        "Alexa source dimensions are preserved");
  check(alexa.tensorHeight == 3168 && alexa.tensorWidth == 4608,
        "Alexa height pads upward to a multiple of eight");

  expectInvalid(nullptr, "16384", "missing size is rejected");
  expectInvalid("", "16384", "empty size is rejected");
  expectInvalid("2160", "16384", "missing separator is rejected");
  expectInvalid("2160x3840x1", "16384", "extra separator is rejected");
  expectInvalid("2160x", "16384", "empty width is rejected");
  expectInvalid("-2160x3840", "16384", "negative height is rejected");
  expectInvalid("2160x3840,2160x4096", "16384", "multiple sizes are rejected");
  expectInvalid("999999999999999999999x1", "16384", "numeric overflow is rejected");
  expectInvalid("1080x1920", "16384", "non-qualification target is rejected");
  expectInvalid("8192x8192", "16384", "unsupported large target is rejected");
  expectInvalid("2160x3840", nullptr, "missing arena limit is rejected");
  expectInvalid("2160x3840", "0", "zero arena limit is rejected");
  expectInvalid("2160x3840", "16MiB", "arena suffix is rejected");
  expectInvalid("2160x3840", "16385", "probe arena maximum is enforced");

  using whitewater::ortprobe::HighResolutionOutcome;
  using whitewater::ortprobe::highResolutionMeasurementResultCollected;
  check(highResolutionMeasurementResultCollected(
            true, true, true, HighResolutionOutcome::InferencePass),
        "successful inference is a collected measurement result");
  check(highResolutionMeasurementResultCollected(
            true, true, true, HighResolutionOutcome::BoundedAllocationStop),
        "bounded allocation stop is a collected measurement result");
  check(!highResolutionMeasurementResultCollected(
            true, true, true, HighResolutionOutcome::OtherFailure),
        "unclassified failure does not pass the measurement-result gate");
  check(!highResolutionMeasurementResultCollected(
            true, false, true, HighResolutionOutcome::InferencePass),
        "not-attempted result does not pass the gate");
  check(!highResolutionMeasurementResultCollected(
            true, true, false, HighResolutionOutcome::InferencePass),
        "missing telemetry does not pass the gate");
  check(!highResolutionMeasurementResultCollected(
            false, true, true, HighResolutionOutcome::InferencePass),
        "invalid configuration does not pass the gate");

  if (failures)
    return 1;
  std::puts("high-resolution configuration tests passed");
  return 0;
}
