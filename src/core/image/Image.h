// Vendored from warp-drive 887a123, src/core/image/Image.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// A float RGBA image buffer, and a non-owning view of one.
//
// This is the type the resampler works on and, eventually, the type the OFX render path
// hands it. It is deliberately dumb: a pointer, two extents, a row stride, and nothing
// else. No file I/O (that belongs to the tools), no colour management, no reference
// counting, no lazy anything. Everything clever in this project lives in the warp math;
// the pixels are just pixels.
//
// Two decisions worth stating:
//
//   * RGBA always, four floats interleaved. Interleaved because every filter tap reads all
//     four channels of a pixel together, so planar storage would multiply the number of
//     cache lines a 4x4 Catmull-Rom neighbourhood touches by four. Fixed at four channels
//     because that is what a host hands us and what the alpha-aware resampler needs; a
//     three-channel path would only mean two copies of the same loop.
//
//   * The view carries a *row stride*, separately from the width. A host's image buffer is
//     frequently a window into a larger allocation, and the OFX render path will hand us
//     exactly that. A view that assumed packed rows would force a copy on the way in and
//     another on the way out, per tile, per frame.
//
// Coordinates. Row 0 is the bottom row: OFX is bottom-left origin, Flame reports an empty
// NativeOrigin so the spec's default is what we assume, and having the buffer agree with
// the coordinate system the warp is solved in means no vertical flip anywhere in the render
// path. Pixel (x, y) covers the unit square [x, x+1) x [y, y+1), so its centre -- the point
// the resampler asks the warp about -- is (x + 0.5, y + 0.5).

#ifndef WHITEWATER_CORE_IMAGE_IMAGE_H
#define WHITEWATER_CORE_IMAGE_IMAGE_H

#include <cstddef>
#include <type_traits>
#include <vector>

namespace whitewater {

// Red, green, blue, alpha, in that order.
constexpr int kImageChannels = 4;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

// A rectangle of interleaved float RGBA owned by somebody else. Copying a view is free and
// copies nothing but the pointer, so views are passed by value or by const reference
// interchangeably; the constness of the *view* says nothing about the constness of the
// pixels, which is carried by T.
template <typename T>
class BasicImageView {
 public:
  BasicImageView() = default;

  // rowStride is measured in floats, not in pixels and not in bytes. Floats because that
  // is the unit every pointer arithmetic in the sampler works in, so any other choice
  // would put a multiply or a divide at each conversion.
  BasicImageView(T *pixels, int width, int height, std::ptrdiff_t rowStride)
      : pixels_(pixels), width_(width), height_(height), rowStride_(rowStride) {}

  BasicImageView(T *pixels, int width, int height)
      : BasicImageView(pixels, width, height,
                       static_cast<std::ptrdiff_t>(width) * kImageChannels) {}

  // A mutable view converts implicitly to a const one; the reverse does not compile. The
  // enable_if is what makes that true -- with T = float there is no U satisfying
  // T == const U, so this constructor is not a candidate and cannot shadow the copy
  // constructor.
  template <typename U,
            typename = typename std::enable_if<std::is_same<T, const U>::value>::type>
  BasicImageView(const BasicImageView<U> &other)  // NOLINT(runtime/explicit)
      : pixels_(other.data()),
        width_(other.width()),
        height_(other.height()),
        rowStride_(other.rowStride()) {}

  bool isEmpty() const { return pixels_ == nullptr || width_ <= 0 || height_ <= 0; }

  int width() const { return width_; }
  int height() const { return height_; }
  std::ptrdiff_t rowStride() const { return rowStride_; }

  T *data() const { return pixels_; }

  // No bounds check. The sampler resolves every index through an edge mode before it gets
  // here, and a branch in this function would be a branch in the innermost loop.
  T *row(int y) const { return pixels_ + static_cast<std::ptrdiff_t>(y) * rowStride_; }
  T *pixel(int x, int y) const {
    return row(y) + static_cast<std::ptrdiff_t>(x) * kImageChannels;
  }

  bool contains(int x, int y) const {
    return x >= 0 && y >= 0 && x < width_ && y < height_;
  }

 private:
  T *pixels_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  std::ptrdiff_t rowStride_ = 0;
};

using ImageView = BasicImageView<float>;
using ConstImageView = BasicImageView<const float>;

// ---------------------------------------------------------------------------
// An owned buffer
// ---------------------------------------------------------------------------

// Packed rows, zero filled on construction. Anything with padding, or borrowed from a
// host, is a BasicImageView over somebody else's memory instead -- this class exists for
// the offline tools and the tests, which are the only places that need to own pixels.
class Image {
 public:
  Image() = default;

  // Non-positive extents give an empty image rather than throwing: a tool that computed a
  // size from a malformed header should report that itself, with the context to say what
  // was wrong, rather than have an exception thrown from underneath it.
  Image(int width, int height);

  bool isEmpty() const { return width_ <= 0 || height_ <= 0; }
  int width() const { return width_; }
  int height() const { return height_; }
  std::ptrdiff_t rowStride() const {
    return static_cast<std::ptrdiff_t>(width_) * kImageChannels;
  }

  ImageView view() { return ImageView(pixels_.data(), width_, height_, rowStride()); }
  ConstImageView view() const {
    return ConstImageView(pixels_.data(), width_, height_, rowStride());
  }

  float *pixel(int x, int y) { return view().pixel(x, y); }
  const float *pixel(int x, int y) const { return view().pixel(x, y); }

  void fill(float r, float g, float b, float a);

 private:
  int width_ = 0;
  int height_ = 0;
  std::vector<float> pixels_;
};

}  // namespace whitewater

#endif  // WHITEWATER_CORE_IMAGE_IMAGE_H
