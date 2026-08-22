#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Pfm.h"
#include "core/flow/FlowWarpMap.h"
#include "core/flow/StMap.h"
#include "core/warp/Resampler.h"
#include "infer/NullPairwiseEstimator.h"

namespace {

enum class OutputMode { kFlow, kSt, kWarp };

struct Options {
  OutputMode mode = OutputMode::kFlow;
  std::string firstPath;
  std::string secondPath;
  std::string outputPath;
  whitewater::Vec2 translation;
  whitewater::StMapMode stMode = whitewater::StMapMode::kAbsoluteUV;
  whitewater::StMapOrigin stOrigin = whitewater::StMapOrigin::kBottomLeft;
};

void usage(std::ostream &stream) {
  stream << "usage: ww-flow [options] FIRST.pfm SECOND.pfm\n"
            "  --output PATH                 output PFM (required)\n"
            "  --mode flow|st|warp          output kind (default: flow)\n"
            "  --translate DX DY            deterministic backward displacement\n"
            "  --st-mode absolute|relative  ST representation (default: absolute)\n"
            "  --st-origin bottom|top       ST origin (default: bottom)\n"
            "\n"
            "Phase 2 uses the analytic null estimator. The model bake-off and ONNX\n"
            "backend are deliberately not part of this build.\n";
}

double parseFinite(const std::string &value, const char *name) {
  std::size_t consumed = 0;
  double parsed = 0.0;
  try {
    parsed = std::stod(value, &consumed);
  } catch (...) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  if (consumed != value.size() || !std::isfinite(parsed))
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  return parsed;
}

Options parseOptions(int argc, char **argv) {
  Options options;
  std::vector<std::string> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto requireValue = [&](const char *option) -> std::string {
      if (++index >= argc) throw std::invalid_argument(std::string(option) + " needs a value");
      return argv[index];
    };

    if (argument == "--help" || argument == "-h") {
      usage(std::cout);
      std::exit(0);
    } else if (argument == "--output" || argument == "-o") {
      options.outputPath = requireValue("--output");
    } else if (argument == "--mode") {
      const std::string value = requireValue("--mode");
      if (value == "flow")
        options.mode = OutputMode::kFlow;
      else if (value == "st")
        options.mode = OutputMode::kSt;
      else if (value == "warp")
        options.mode = OutputMode::kWarp;
      else
        throw std::invalid_argument("--mode must be flow, st, or warp");
    } else if (argument == "--translate") {
      const std::string dx = requireValue("--translate");
      const std::string dy = requireValue("--translate");
      options.translation = whitewater::Vec2(parseFinite(dx, "DX"), parseFinite(dy, "DY"));
    } else if (argument == "--st-mode") {
      const std::string value = requireValue("--st-mode");
      if (value == "absolute")
        options.stMode = whitewater::StMapMode::kAbsoluteUV;
      else if (value == "relative")
        options.stMode = whitewater::StMapMode::kRelativePixels;
      else
        throw std::invalid_argument("--st-mode must be absolute or relative");
    } else if (argument == "--st-origin") {
      const std::string value = requireValue("--st-origin");
      if (value == "bottom")
        options.stOrigin = whitewater::StMapOrigin::kBottomLeft;
      else if (value == "top")
        options.stOrigin = whitewater::StMapOrigin::kTopLeft;
      else
        throw std::invalid_argument("--st-origin must be bottom or top");
    } else if (!argument.empty() && argument[0] == '-') {
      throw std::invalid_argument("unknown option: " + argument);
    } else {
      positional.push_back(argument);
    }
  }
  if (positional.size() != 2) throw std::invalid_argument("two input PFM paths are required");
  if (options.outputPath.empty()) throw std::invalid_argument("--output is required");
  options.firstPath = positional[0];
  options.secondPath = positional[1];
  return options;
}

whitewater::OwnedFrame toFrame(const whitewater::tool::PfmImage &input) {
  whitewater::OwnedFrame frame;
  frame.bounds = {0, 0, input.width, input.height};
  frame.origin = whitewater::Vec2();
  frame.rowStride = static_cast<std::ptrdiff_t>(input.width) * whitewater::kImageChannels;
  frame.rgba.resize(static_cast<std::size_t>(input.width) * input.height *
                    whitewater::kImageChannels);
  frame.sourceFormat.depth = whitewater::CapturedPixelDepth::kFloat;
  frame.sourceFormat.components = input.channels == 1
                                      ? whitewater::CapturedPixelComponents::kAlpha
                                      : whitewater::CapturedPixelComponents::kRGB;
  frame.sourceFormat.alpha = whitewater::CapturedAlphaAssociation::kOpaque;
  frame.pixelAspectRatio = 1.0;
  for (int y = 0; y < input.height; ++y) {
    const float *source = input.pixels.data() +
                          static_cast<std::size_t>(y) * input.width * input.channels;
    float *destination = frame.rgba.data() +
                         static_cast<std::size_t>(y) * frame.rowStride;
    for (int x = 0; x < input.width; ++x) {
      if (input.channels == 1) {
        destination[0] = destination[1] = destination[2] = source[0];
      } else {
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
      }
      destination[3] = 1.0f;
      source += input.channels;
      destination += whitewater::kImageChannels;
    }
  }
  return frame;
}

whitewater::Image frameImage(const whitewater::OwnedFrame &frame) {
  whitewater::Image image(frame.width(), frame.height());
  whitewater::ImageView destination = image.view();
  for (int y = 0; y < frame.height(); ++y) {
    const float *source = frame.row(y);
    float *out = destination.row(y);
    std::copy(source, source + static_cast<std::ptrdiff_t>(frame.width()) *
                                   whitewater::kImageChannels,
              out);
  }
  return image;
}

whitewater::tool::PfmImage rgbPfm(const whitewater::Image &image) {
  whitewater::tool::PfmImage output;
  output.width = image.width();
  output.height = image.height();
  output.channels = 3;
  output.pixels.resize(static_cast<std::size_t>(output.width) * output.height * 3);
  const whitewater::ConstImageView source = image.view();
  for (int y = 0; y < output.height; ++y) {
    const float *pixel = source.row(y);
    float *out = output.pixels.data() + static_cast<std::size_t>(y) * output.width * 3;
    for (int x = 0; x < output.width; ++x) {
      out[0] = pixel[0];
      out[1] = pixel[1];
      out[2] = pixel[2];
      pixel += whitewater::kImageChannels;
      out += 3;
    }
  }
  return output;
}

whitewater::tool::PfmImage flowPfm(const whitewater::FlowResult &result) {
  const whitewater::FlowField &field = result.link.field();
  whitewater::tool::PfmImage output;
  output.width = field.columns();
  output.height = field.rows();
  output.channels = 3;
  output.pixels.resize(static_cast<std::size_t>(output.width) * output.height * 3);
  for (int y = 0; y < output.height; ++y) {
    for (int x = 0; x < output.width; ++x) {
      const whitewater::Vec2 flow = whitewater::flowNode(field, x, y);
      float *out = output.pixels.data() +
                   (static_cast<std::size_t>(y) * output.width + x) * 3;
      out[0] = static_cast<float>(flow.x);
      out[1] = static_cast<float>(flow.y);
      out[2] = result.confidence ? result.confidence->node(x, y)[0] : 0.0f;
    }
  }
  return output;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parseOptions(argc, argv);
    whitewater::tool::PfmImage firstInput;
    whitewater::tool::PfmImage secondInput;
    std::string error;
    if (!whitewater::tool::readPfm(options.firstPath, &firstInput, &error) ||
        !whitewater::tool::readPfm(options.secondPath, &secondInput, &error)) {
      throw std::runtime_error(error);
    }
    if (firstInput.width != secondInput.width || firstInput.height != secondInput.height)
      throw std::runtime_error("input PFM dimensions must match");

    const whitewater::OwnedFrame first = toFrame(firstInput);
    const whitewater::OwnedFrame second = toFrame(secondInput);
    whitewater::FlowRequest request;
    request.fromTime = 1;
    request.toTime = 0;
    request.columns = first.width();
    request.rows = first.height();
    request.fromGeometry = whitewater::FieldGeometry::forPixels(first.bounds.x1, first.bounds.y1);
    request.toGeometry = whitewater::FieldGeometry::forPixels(second.bounds.x1, second.bounds.y1);

    whitewater::NullFlowParameters parameters;
    parameters.pattern = options.translation == whitewater::Vec2()
                             ? whitewater::NullFlowPattern::kIdentity
                             : whitewater::NullFlowPattern::kTranslation;
    parameters.translation = options.translation;
    const whitewater::FlowResult flow =
        whitewater::NullPairwiseEstimator(parameters).estimate(first, second, request);
    if (!flow.succeeded()) throw std::runtime_error(flow.message);

    whitewater::tool::PfmImage output;
    if (options.mode == OutputMode::kFlow) {
      output = flowPfm(flow);
    } else if (options.mode == OutputMode::kSt) {
      whitewater::StMapOptions stOptions;
      stOptions.mode = options.stMode;
      stOptions.origin = options.stOrigin;
      stOptions.sourceBounds = second.bounds;
      stOptions.destinationBounds = first.bounds;
      output = rgbPfm(whitewater::fieldToStMap(flow.link.field(), stOptions));
    } else {
      const whitewater::Image source = frameImage(second);
      whitewater::Image warped(first.width(), first.height());
      const whitewater::FlowWarpMap map(flow.link);
      whitewater::ResampleOptions resampleOptions(whitewater::AlphaMode::kUnpremultiplied);
      resampleOptions.filter = whitewater::ResampleFilter::kBilinear;
      resampleOptions.edge = whitewater::EdgeMode::kBlack;
      whitewater::resample(source.view(), map, resampleOptions, whitewater::ResampleGeometry(),
                           warped.view());
      output = rgbPfm(warped);
    }
    if (!whitewater::tool::writePfm(options.outputPath, output, &error))
      throw std::runtime_error(error);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ww-flow: " << error.what() << '\n';
    usage(std::cerr);
    return 2;
  }
}
