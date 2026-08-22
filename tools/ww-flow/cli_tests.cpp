#include "Pfm.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
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

bool near(float a, double b) { return std::fabs(static_cast<double>(a) - b) <= 1.0e-6; }

std::string shellQuote(const std::string &value) {
  std::string result = "'";
  for (char character : value) {
    if (character == '\'')
      result += "'\\''";
    else
      result += character;
  }
  return result + "'";
}

bool run(const std::string &executable, const std::string &arguments) {
  return std::system((shellQuote(executable) + " " + arguments).c_str()) == 0;
}

whitewater::tool::PfmImage image(float offset) {
  whitewater::tool::PfmImage result;
  result.width = 3;
  result.height = 2;
  result.channels = 3;
  result.pixels.resize(18);
  for (std::size_t index = 0; index < result.pixels.size(); ++index)
    result.pixels[index] = offset + static_cast<float>(index) * 0.125f;
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ww_flow_cli_tests /path/to/ww-flow\n";
    return 2;
  }
  const std::string executable = argv[1];
  const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("whitewater-ww-flow-cli-tests-" + std::to_string(unique));
  std::error_code ignored;
  std::filesystem::create_directories(directory);

  const std::filesystem::path firstPath = directory / "first.pfm";
  const std::filesystem::path secondPath = directory / "second.pfm";
  const std::filesystem::path flowPath = directory / "flow.pfm";
  const std::filesystem::path stPath = directory / "st.pfm";
  const std::filesystem::path warpPath = directory / "warp.pfm";
  const whitewater::tool::PfmImage first = image(1.0f);
  const whitewater::tool::PfmImage second = image(10.0f);
  std::string error;
  check(whitewater::tool::writePfm(firstPath.string(), first, &error), "write first input");
  check(whitewater::tool::writePfm(secondPath.string(), second, &error), "write second input");

  const std::string inputs = shellQuote(firstPath.string()) + " " + shellQuote(secondPath.string());
  check(run(executable, "--mode flow --translate 1.5 -2 --output " +
                            shellQuote(flowPath.string()) + " " + inputs),
        "flow mode runs through NullPairwiseEstimator");
  whitewater::tool::PfmImage flow;
  check(whitewater::tool::readPfm(flowPath.string(), &flow, &error), "read flow output");
  for (std::size_t index = 0; index + 2 < flow.pixels.size(); index += 3) {
    check(near(flow.pixels[index], 1.5) && near(flow.pixels[index + 1], -2.0) &&
              near(flow.pixels[index + 2], 0.0),
          "flow PFM stores signed U/V and optional confidence channel");
  }

  check(run(executable, "--mode st --output " + shellQuote(stPath.string()) + " " + inputs),
        "ST mode runs through the null-estimator identity field");
  whitewater::tool::PfmImage st;
  check(whitewater::tool::readPfm(stPath.string(), &st, &error), "read ST output");
  for (int y = 0; y < st.height; ++y) {
    for (int x = 0; x < st.width; ++x) {
      const float *pixel = st.pixels.data() + (static_cast<std::size_t>(y) * st.width + x) * 3;
      check(near(pixel[0], (x + 0.5) / st.width) && near(pixel[1], (y + 0.5) / st.height),
            "absolute ST output uses Flame's measured half-pixel bottom-left convention");
    }
  }

  check(run(executable,
            "--mode warp --output " + shellQuote(warpPath.string()) + " " + inputs),
        "warp mode runs through the null-estimator identity field");
  whitewater::tool::PfmImage warped;
  check(whitewater::tool::readPfm(warpPath.string(), &warped, &error), "read warped output");
  check(warped.pixels == second.pixels,
        "identity flow warps the second input into the first frame exactly");

  std::filesystem::remove_all(directory, ignored);
  if (failures == 0) std::cout << "ww-flow CLI tests passed\n";
  return failures == 0 ? 0 : 1;
}
