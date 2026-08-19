// Vendored from warp-drive 887a123, src/ofx/FrameCapture.cpp. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "ofx/FrameCapture.h"

#include <algorithm>
#include <cmath>
#include <memory>

namespace whitewater {
namespace ofx {
namespace {

CapturedPixelDepth capturedDepth(OFX::BitDepthEnum depth) {
  switch (depth) {
    case OFX::eBitDepthUByte: return CapturedPixelDepth::kByte;
    case OFX::eBitDepthUShort: return CapturedPixelDepth::kShort;
    case OFX::eBitDepthHalf: return CapturedPixelDepth::kHalf;
    case OFX::eBitDepthFloat: return CapturedPixelDepth::kFloat;
    case OFX::eBitDepthCustom: return CapturedPixelDepth::kCustom;
    default: return CapturedPixelDepth::kUnknown;
  }
}

CapturedPixelComponents capturedComponents(OFX::PixelComponentEnum components) {
  switch (components) {
    case OFX::ePixelComponentRGBA: return CapturedPixelComponents::kRGBA;
    case OFX::ePixelComponentRGB: return CapturedPixelComponents::kRGB;
    case OFX::ePixelComponentAlpha: return CapturedPixelComponents::kAlpha;
    case OFX::ePixelComponentCustom: return CapturedPixelComponents::kCustom;
    default: return CapturedPixelComponents::kUnknown;
  }
}

CapturedAlphaAssociation capturedAlpha(OFX::PreMultiplicationEnum premultiplication) {
  switch (premultiplication) {
    case OFX::eImageOpaque: return CapturedAlphaAssociation::kOpaque;
    case OFX::eImagePreMultiplied: return CapturedAlphaAssociation::kPremultiplied;
    case OFX::eImageUnPreMultiplied: return CapturedAlphaAssociation::kUnpremultiplied;
  }
  return CapturedAlphaAssociation::kUnpremultiplied;
}

}  // namespace

FrameCaptureResult captureHostImage(const HostImageData &image) {
  HostSourceImage source;
  std::string error;
  if (!source.attach(image, &error)) return {std::nullopt, error};

  const ConstImageView view = source.view();
  const CapturedPixelBounds bounds = {image.bounds.x1, image.bounds.y1, image.bounds.x2,
                                      image.bounds.y2};
  if (view.isEmpty()) {
    return {std::nullopt, "source image has empty bounds or no pixel data"};
  }

  CapturedFrame frame;
  frame.bounds = bounds;
  frame.origin = source.origin();
  frame.rowStride = static_cast<std::ptrdiff_t>(view.width()) * kImageChannels;
  frame.sourceFormat = {capturedDepth(image.depth), capturedComponents(image.components),
                        capturedAlpha(image.premultiplication)};
  frame.pixelAspectRatio = image.pixelAspectRatio;
  if (!(frame.pixelAspectRatio > 0.0) || !std::isfinite(frame.pixelAspectRatio)) {
    return {std::nullopt, "source image has an invalid pixel aspect ratio"};
  }

  frame.rgba.resize(static_cast<std::size_t>(view.height()) *
                    static_cast<std::size_t>(frame.rowStride));
  for (int y = 0; y < view.height(); ++y) {
    const float *input = view.row(y);
    float *output = frame.rgba.data() + static_cast<std::ptrdiff_t>(y) * frame.rowStride;
    std::copy(input, input + frame.rowStride, output);
  }
  return {std::move(frame), std::string()};
}

FrameCaptureResult captureSourceFrame(OFX::Clip &sourceClip, double time) {
  std::unique_ptr<OFX::Image> image;
  try {
    image.reset(sourceClip.fetchImage(time));
  } catch (const std::exception &error) {
    return {std::nullopt, std::string("clipGetImage failed: ") + error.what()};
  }
  if (image == nullptr) return {std::nullopt, "clipGetImage returned no source image"};

  HostImageData data;
  data.pixels = image->getPixelData();
  data.bounds = image->getBounds();
  data.rowBytes = image->getRowBytes();
  data.depth = image->getPixelDepth();
  data.components = image->getPixelComponents();
  data.premultiplication = image->getPreMultiplication();
  data.pixelAspectRatio = image->getPixelAspectRatio();
  // `image` remains alive through the complete copy in captureHostImage and is released by
  // the unique_ptr before this function returns. No caller can retain host-owned pixels.
  return captureHostImage(data);
}

}  // namespace ofx
}  // namespace whitewater
