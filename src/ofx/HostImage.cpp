// Vendored from warp-drive 887a123, src/ofx/HostImage.cpp.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "ofx/HostImage.h"

#include <algorithm>
#include <sstream>

#include "ofx/PixelFormat.h"

namespace whitewater {
namespace ofx {

namespace {

const char *depthName(OFX::BitDepthEnum depth) {
  switch (depth) {
    case OFX::eBitDepthUByte: return "byte";
    case OFX::eBitDepthUShort: return "short";
    case OFX::eBitDepthHalf: return "half";
    case OFX::eBitDepthFloat: return "float";
    case OFX::eBitDepthCustom: return "a custom depth";
    default: return "no depth";
  }
}

const char *componentsName(OFX::PixelComponentEnum components) {
  switch (components) {
    case OFX::ePixelComponentRGBA: return "RGBA";
    case OFX::ePixelComponentRGB: return "RGB";
    case OFX::ePixelComponentAlpha: return "alpha";
    case OFX::ePixelComponentCustom: return "custom components";
    default: return "no components";
  }
}

std::string describeLayout(OFX::BitDepthEnum depth, OFX::PixelComponentEnum components) {
  std::ostringstream message;
  message << depthName(depth) << " " << componentsName(components);
  return message.str();
}

// A row's worth of host pixels, addressed from the pointer the host gave for its
// (bounds.x1, bounds.y1) pixel. Row bytes may be negative -- an image stored top down is
// legal -- so this is signed arithmetic throughout.
const void *hostRowAddress(const void *base, int rowBytes, int row) {
  return static_cast<const char *>(base) + static_cast<std::ptrdiff_t>(rowBytes) * row;
}

void *hostRowAddress(void *base, int rowBytes, int row) {
  return static_cast<char *>(base) + static_cast<std::ptrdiff_t>(rowBytes) * row;
}

}  // namespace

AlphaMode alphaModeFor(OFX::PreMultiplicationEnum premultiplication) {
  // Opaque data has alpha 1 everywhere, so premultiplied and unpremultiplied are the same
  // pixels -- and the premultiplied path is the cheap one, with no divide per sample.
  return premultiplication == OFX::eImagePreMultiplied ? AlphaMode::kPremultiplied
                                                       : AlphaMode::kUnpremultiplied;
}

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

bool HostSourceImage::attach(const OFX::Image &image, std::string *error) {
  HostImageData data;
  data.pixels = image.getPixelData();
  data.bounds = image.getBounds();
  data.rowBytes = image.getRowBytes();
  data.depth = image.getPixelDepth();
  data.components = image.getPixelComponents();
  data.premultiplication = image.getPreMultiplication();
  data.pixelAspectRatio = image.getPixelAspectRatio();
  return attach(data, error);
}

bool HostSourceImage::attach(const HostImageData &image, std::string *error) {
  storage_.clear();
  bounds_ = image.bounds;
  const int width = bounds_.x2 - bounds_.x1;
  const int height = bounds_.y2 - bounds_.y1;
  const OFX::BitDepthEnum depth = image.depth;
  const OFX::PixelComponentEnum components = image.components;
  const void *pixels = image.pixels;

  if (width <= 0 || height <= 0 || pixels == nullptr) {
    // Not an error: a clip can legitimately hand back an empty region, and the caller's
    // answer to that is a black frame rather than a failed render.
    view_ = ConstImageView();
    return true;
  }
  if (hostPixelBytes(depth, components) == 0) {
    if (error != nullptr) {
      *error = "cannot read a source image in " + describeLayout(depth, components);
    }
    return false;
  }

  const int rowBytes = image.rowBytes;
  if (isDirectlyUsableLayout(depth, components, rowBytes)) {
    view_ = ConstImageView(static_cast<const float *>(pixels), width, height,
                           rowBytes / static_cast<std::ptrdiff_t>(sizeof(float)));
    return true;
  }

  storage_.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                      kImageChannels,
                  0.0f);
  const std::ptrdiff_t stride = static_cast<std::ptrdiff_t>(width) * kImageChannels;
  for (int y = 0; y < height; ++y) {
    readHostRow(hostRowAddress(pixels, rowBytes, y), depth, components, width,
                storage_.data() + stride * y);
  }
  view_ = ConstImageView(storage_.data(), width, height, stride);
  return true;
}

Vec2 HostSourceImage::origin() const {
  return Vec2(static_cast<double>(bounds_.x1), static_cast<double>(bounds_.y1));
}

// ---------------------------------------------------------------------------
// Destination
// ---------------------------------------------------------------------------

bool HostDestinationImage::attach(OFX::Image &image, const OfxRectI &window,
                                  std::string *error) {
  const OfxRectI bounds = image.getBounds();

  // The host is allowed to hand over an image smaller than the window it asked for -- and
  // writing outside the buffer because we trusted the window would corrupt the host's
  // memory, not ours. Intersect and render what actually exists.
  window_.x1 = std::max(window.x1, bounds.x1);
  window_.y1 = std::max(window.y1, bounds.y1);
  window_.x2 = std::min(window.x2, bounds.x2);
  window_.y2 = std::min(window.y2, bounds.y2);

  const int width = window_.x2 - window_.x1;
  const int height = window_.y2 - window_.y1;
  depth_ = image.getPixelDepth();
  components_ = image.getPixelComponents();
  hostRowBytes_ = image.getRowBytes();

  if (width <= 0 || height <= 0) {
    view_ = ImageView();
    return true;
  }
  const int pixelBytes = hostPixelBytes(depth_, components_);
  if (pixelBytes == 0) {
    if (error != nullptr) {
      *error = "cannot write an output image in " + describeLayout(depth_, components_);
    }
    return false;
  }

  hostRowZero_ = image.getPixelAddress(window_.x1, window_.y1);
  if (hostRowZero_ == nullptr) {
    view_ = ImageView();
    return true;
  }

  if (isDirectlyUsableLayout(depth_, components_, hostRowBytes_)) {
    view_ = ImageView(static_cast<float *>(hostRowZero_), width, height,
                      hostRowBytes_ / static_cast<std::ptrdiff_t>(sizeof(float)));
    return true;
  }

  storage_.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                      kImageChannels,
                  0.0f);
  view_ = ImageView(storage_.data(), width, height,
                    static_cast<std::ptrdiff_t>(width) * kImageChannels);
  return true;
}

Vec2 HostDestinationImage::origin() const {
  return Vec2(static_cast<double>(window_.x1), static_cast<double>(window_.y1));
}

void HostDestinationImage::writeBackRows(int firstRow, int rowCount) const {
  if (storage_.empty() || hostRowZero_ == nullptr) {
    return;
  }
  const int first = std::max(0, firstRow);
  const int last = std::min(view_.height(), firstRow + rowCount);
  for (int y = first; y < last; ++y) {
    writeHostRow(view_.row(y), view_.width(), depth_, components_,
                 hostRowAddress(hostRowZero_, hostRowBytes_, y));
  }
}

}  // namespace ofx
}  // namespace whitewater
