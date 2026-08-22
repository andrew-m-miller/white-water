#include "Pfm.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
  }
}

}  // namespace

int main() {
  using whitewater::tool::PfmImage;
  const std::filesystem::path path =
      std::filesystem::temp_directory_path() / "whitewater-pfm-roundtrip.pfm";

  PfmImage source;
  source.width = 3;
  source.height = 2;
  source.channels = 3;
  source.pixels = {0.0f, 1.0f, -2.0f, 3.5f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                   9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f};
  std::string error;
  check(whitewater::tool::writePfm(path.string(), source, &error),
        "write a three-channel PFM: " + error);

  PfmImage decoded;
  check(whitewater::tool::readPfm(path.string(), &decoded, &error),
        "read the written PFM: " + error);
  check(decoded.width == source.width && decoded.height == source.height &&
            decoded.channels == source.channels,
        "round trip preserves dimensions and channel count");
  check(decoded.pixels == source.pixels, "round trip preserves signed float samples exactly");

  PfmImage invalid;
  invalid.width = 1;
  invalid.height = 1;
  invalid.channels = 2;
  invalid.pixels = {1.0f, 2.0f};
  check(!whitewater::tool::writePfm(path.string(), invalid, &error),
        "PFM writer rejects an unsupported two-channel payload");

  std::error_code ignored;
  std::filesystem::remove(path, ignored);
  if (failures == 0) std::cout << "PFM tests passed\n";
  return failures == 0 ? 0 : 1;
}
