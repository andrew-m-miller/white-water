// A dense sampled field, and the geometry that says where its nodes sit.
//
// Everything the tracker computes is one of these: a two-channel optical flow, or a
// one-channel confidence. They share a lattice, a sampler and a storage layout, so they
// share a template rather than two copies of the same bilinear loop.
//
// ---------------------------------------------------------------------------
// The one decision that matters: units
// ---------------------------------------------------------------------------
//
// A field is stored on its own lattice -- which may be coarser than the image, because
// flow is estimated at a reduced analysis resolution -- but the *values* are always in
// full-resolution pixels. The lattice is described by FieldGeometry: node (0, 0) sits at
// `origin`, with adjacent nodes `spacingX`/`spacingY` full-resolution pixels apart.
//
// This is why there is no upscale pass anywhere in this project. A half-resolution RAFT
// result becomes a field with spacing 2 on both axes whose vectors were multiplied by 2 on
// the way in; an anisotropic result scales each component independently. From that moment
// every consumer -- compose, the ST map, the resampler -- works in one coordinate space and
// never has to know that an analysis scale exists. The
// alternative, resampling every field up to full resolution on arrival, would cost a copy
// per link and would filter the same data twice: once on the way up, once when it is
// sampled.
//
// ---------------------------------------------------------------------------
// Why float and not half
// ---------------------------------------------------------------------------
//
// Half floats would halve the cache footprint and are tempting for exactly that reason.
// They are wrong here. Half carries an 11-bit significand, so at a displacement of 1000
// pixels -- which an accumulated chain across a whipping camera move reaches easily -- the
// representable step is about half a pixel. That is a visible judder in a locked-off
// insert. Under about 64 pixels half would be fine, but a field type whose precision
// silently depends on how far the shot has travelled is a bug waiting for a specific
// shot. Float everywhere; the cache budget is the knob for memory, not the number format.
//
// Interleaved rather than planar, for the same reason the image buffer is: every consumer
// reads all channels of a node together, so splitting them would multiply the cache lines
// a bilinear neighbourhood touches by the channel count.
//
// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
//
// Bilinear, with positions outside the lattice clamped to it. Clamping rather than
// extrapolating is the only defensible answer when the field has no information out
// there: the result stays bounded and continuous, and an insert that drifts off the plate
// keeps the motion of the nearest edge instead of flying away on an extrapolated slope.
//
// A built field is immutable in practice and holds no lazy state, so render threads share
// one freely. Nothing here allocates or caches on the sampling path.

#ifndef WHITEWATER_CORE_FLOW_FIELD_H
#define WHITEWATER_CORE_FLOW_FIELD_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "core/geom/Vec2.h"

namespace whitewater {

// Where a field's nodes sit in full-resolution image pixel coordinates.
//
// `origin` is the position of node (0, 0) itself -- not the corner of a cell. For a
// full-resolution field over an image whose lower-left pixel is (x0, y0), that is
// (x0 + 0.5, y0 + 0.5), because pixel centres are what the resampler asks about.
//
// The two pitches are intentionally separate.  An anamorphic image, an odd-sized image
// reduced to a model lattice, or independently rounded analysis extents can have a
// different X and Y pitch.  They are checked here, before a sampler can divide by one of
// them.  Keeping the check in the value type is important: a malformed geometry cannot
// be made safe merely by remembering to validate it at each call site.
struct FieldGeometry {
  Vec2 origin;
  double spacingX = 1.0;
  double spacingY = 1.0;

  FieldGeometry() = default;
  FieldGeometry(Vec2 originValue, double spacingXValue, double spacingYValue)
      : origin(originValue), spacingX(spacingXValue), spacingY(spacingYValue) {
    validate();
  }

  // A scalar-pitch overload is retained as a source-compatible convenience for callers
  // that genuinely have square analysis pixels.  New code should pass both pitches when
  // the analysis geometry is derived from an image.
  FieldGeometry(Vec2 originValue, double spacingValue)
      : FieldGeometry(originValue, spacingValue, spacingValue) {}

  bool isValid() const {
    return std::isfinite(origin.x) && std::isfinite(origin.y) &&
           std::isfinite(spacingX) && std::isfinite(spacingY) && spacingX > 0.0 &&
           spacingY > 0.0;
  }

  void validate() const {
    if (!isValid()) {
      throw std::invalid_argument("FieldGeometry requires finite, positive spacing");
    }
  }

  // Convert a continuous lattice coordinate (where (0, 0) is the first node) into the
  // full-resolution image coordinate space used by field values and the resampler.
  Vec2 latticeToImage(Vec2 lattice) const {
    return Vec2(origin.x + lattice.x * spacingX, origin.y + lattice.y * spacingY);
  }

  // The inverse of latticeToImage.  Validation in the constructor makes both divisions
  // safe for every FieldGeometry value that can be constructed.
  Vec2 imageToLattice(Vec2 image) const {
    return Vec2((image.x - origin.x) / spacingX, (image.y - origin.y) / spacingY);
  }

  // The geometry of a full-resolution field covering an image whose lower-left pixel is
  // (x, y). Named rather than left to the caller because getting the half-pixel wrong is
  // the classic way to introduce a uniform half-pixel shift that survives every test that
  // uses a symmetric image.
  static FieldGeometry forPixels(int x, int y) {
    return FieldGeometry(Vec2(x + 0.5, y + 0.5), 1.0, 1.0);
  }

  // The geometry of a field estimated at 1/`scale` resolution over the same image. A
  // half-resolution node covers two full-resolution pixels, and its centre sits at the
  // centre of that pair.
  static FieldGeometry forPixels(int x, int y, double scale) {
    return FieldGeometry(Vec2(x + 0.5 * scale, y + 0.5 * scale), scale, scale);
  }

  // The anisotropic equivalent of the scale overload.  Each lattice node remains at the
  // centre of the corresponding analysis footprint independently in X and Y.
  static FieldGeometry forPixels(int x, int y, double scaleX, double scaleY) {
    return FieldGeometry(Vec2(x + 0.5 * scaleX, y + 0.5 * scaleY), scaleX, scaleY);
  }

  bool operator==(const FieldGeometry &other) const {
    return origin == other.origin && spacingX == other.spacingX && spacingY == other.spacingY;
  }
  bool operator!=(const FieldGeometry &other) const { return !(*this == other); }
};

inline Vec2 latticeToImage(const FieldGeometry &geometry, Vec2 lattice) {
  return geometry.latticeToImage(lattice);
}

inline Vec2 imageToLattice(const FieldGeometry &geometry, Vec2 image) {
  return geometry.imageToLattice(image);
}

template <int kChannels>
class Field {
  static_assert(kChannels >= 1 && kChannels <= 4, "unsupported channel count");

 public:
  Field() = default;

  // Zero filled. Non-positive extents give an empty field rather than throwing: a caller
  // that derived a size from a malformed model output should report that itself, with the
  // context to say what was wrong.
  Field(int columns, int rows, const FieldGeometry &geometry)
      : columns_(columns > 0 ? columns : 0),
        rows_(rows > 0 ? rows : 0),
        geometry_(geometry) {
    // FieldGeometry validates in its constructors, but its coordinates are intentionally
    // plain value members so callers can use normal aggregate-style updates.  Recheck at
    // the storage boundary as well; no Field with a zero/NaN pitch may reach sample().
    geometry_.validate();
    if (columns_ > 0 && rows_ > 0) {
      data_.assign(static_cast<std::size_t>(columns_) * rows_ * kChannels, 0.0f);
    } else {
      columns_ = 0;
      rows_ = 0;
    }
  }

  bool isEmpty() const { return columns_ == 0 || rows_ == 0; }
  int columns() const { return columns_; }
  int rows() const { return rows_; }
  const FieldGeometry &geometry() const { return geometry_; }

  std::size_t byteSize() const { return data_.size() * sizeof(float); }

  float *node(int column, int row) {
    return data_.data() + index(column, row);
  }
  const float *node(int column, int row) const {
    return data_.data() + index(column, row);
  }

  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }
  std::size_t size() const { return data_.size(); }

  // Full-resolution image position of a node. The inverse of the mapping `sample` uses,
  // exposed because the estimators and the tests both need to walk a lattice in image
  // space and neither should re-derive the half-node offset.
  Vec2 positionOf(int column, int row) const {
    return geometry_.latticeToImage(Vec2(static_cast<double>(column),
                                         static_cast<double>(row)));
  }

  // Bilinear sample at a full-resolution image position, clamped to the lattice. Writes
  // kChannels floats. An empty field samples as zeros, which is the correct answer for a
  // field that carries no information rather than a sentinel the caller must test for.
  void sample(Vec2 position, float *out) const {
    if (isEmpty() || !std::isfinite(position.x) || !std::isfinite(position.y)) {
      for (int c = 0; c < kChannels; ++c) out[c] = 0.0f;
      return;
    }

    const Vec2 lattice = geometry_.imageToLattice(position);
    const double gx = lattice.x;
    const double gy = lattice.y;

    // Clamp the *continuous* coordinate before splitting it, so a position far outside the
    // lattice lands exactly on the boundary node with zero fractional part rather than
    // interpolating between two copies of the same clamped index.
    const double cx = std::min(std::max(gx, 0.0), static_cast<double>(columns_ - 1));
    const double cy = std::min(std::max(gy, 0.0), static_cast<double>(rows_ - 1));

    const int x0 = static_cast<int>(std::floor(cx));
    const int y0 = static_cast<int>(std::floor(cy));
    const int x1 = std::min(x0 + 1, columns_ - 1);
    const int y1 = std::min(y0 + 1, rows_ - 1);

    const float fx = static_cast<float>(cx - x0);
    const float fy = static_cast<float>(cy - y0);

    const float *n00 = node(x0, y0);
    const float *n10 = node(x1, y0);
    const float *n01 = node(x0, y1);
    const float *n11 = node(x1, y1);

    for (int c = 0; c < kChannels; ++c) {
      const float bottom = n00[c] + (n10[c] - n00[c]) * fx;
      const float top = n01[c] + (n11[c] - n01[c]) * fx;
      out[c] = bottom + (top - bottom) * fy;
    }
  }

 private:
  std::size_t index(int column, int row) const {
    return (static_cast<std::size_t>(row) * columns_ + column) * kChannels;
  }

  int columns_ = 0;
  int rows_ = 0;
  FieldGeometry geometry_;
  std::vector<float> data_;
};

// A two-channel displacement field. The stored vector at a node is where the content at
// that position came *from*, in full-resolution pixels -- a backward map, matching the
// direction the resampler wants. See FlowChain.h for what "from" means across a chain.
using FlowField = Field<2>;

// A one-channel field in [0, 1]. Used for forward-backward consistency, where 1 is a
// round trip that returned home and 0 is one that did not.
using ScalarField = Field<1>;

inline Vec2 sampleFlow(const FlowField &field, Vec2 position) {
  float v[2];
  field.sample(position, v);
  return Vec2(v[0], v[1]);
}

inline void setFlow(FlowField *field, int column, int row, Vec2 value) {
  float *n = field->node(column, row);
  n[0] = static_cast<float>(value.x);
  n[1] = static_cast<float>(value.y);
}

inline Vec2 flowNode(const FlowField &field, int column, int row) {
  const float *n = field.node(column, row);
  return Vec2(n[0], n[1]);
}

inline float sampleScalar(const ScalarField &field, Vec2 position) {
  float v[1];
  field.sample(position, v);
  return v[0];
}

}  // namespace whitewater

#endif  // WHITEWATER_CORE_FLOW_FIELD_H
