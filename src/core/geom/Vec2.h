// Vendored from warp-drive 887a123, src/core/geom/Vec2.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// A two-dimensional point/vector in canonical image coordinates.
//
// Doubles rather than floats throughout the core: arc-length inversion and the linear
// solves that follow are sensitive to accumulated error, and nothing here is in the
// per-pixel inner loop where the width would matter. Conversion to float happens at the
// render boundary, not before.

#ifndef WHITEWATER_CORE_GEOM_VEC2_H
#define WHITEWATER_CORE_GEOM_VEC2_H

#include <cmath>

namespace whitewater {

struct Vec2 {
  double x = 0.0;
  double y = 0.0;

  constexpr Vec2() = default;
  constexpr Vec2(double xValue, double yValue) : x(xValue), y(yValue) {}
};

constexpr Vec2 operator+(Vec2 a, Vec2 b) { return Vec2(a.x + b.x, a.y + b.y); }
constexpr Vec2 operator-(Vec2 a, Vec2 b) { return Vec2(a.x - b.x, a.y - b.y); }
constexpr Vec2 operator-(Vec2 a) { return Vec2(-a.x, -a.y); }
constexpr Vec2 operator*(Vec2 a, double s) { return Vec2(a.x * s, a.y * s); }
constexpr Vec2 operator*(double s, Vec2 a) { return Vec2(a.x * s, a.y * s); }
constexpr Vec2 operator/(Vec2 a, double s) { return Vec2(a.x / s, a.y / s); }

inline Vec2 &operator+=(Vec2 &a, Vec2 b) { a.x += b.x; a.y += b.y; return a; }
inline Vec2 &operator-=(Vec2 &a, Vec2 b) { a.x -= b.x; a.y -= b.y; return a; }
inline Vec2 &operator*=(Vec2 &a, double s) { a.x *= s; a.y *= s; return a; }

// Exact comparison. Used for document equality, where the question is "is this the same
// stored number", not "is this geometrically close". Approximate comparison in tests is
// spelled out explicitly at the call site instead.
constexpr bool operator==(Vec2 a, Vec2 b) { return a.x == b.x && a.y == b.y; }
constexpr bool operator!=(Vec2 a, Vec2 b) { return !(a == b); }

constexpr double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }

// The z component of the 3D cross product; positive when b is counter-clockwise from a.
constexpr double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }

constexpr double lengthSquared(Vec2 a) { return a.x * a.x + a.y * a.y; }

inline double length(Vec2 a) { return std::sqrt(lengthSquared(a)); }

inline double distance(Vec2 a, Vec2 b) { return length(b - a); }

// Returns the zero vector for a zero-length input rather than NaN, so degenerate
// geometry (coincident control points) does not poison a whole evaluation.
inline Vec2 normalized(Vec2 a) {
  const double len = length(a);
  return len > 0.0 ? a / len : Vec2();
}

inline Vec2 lerp(Vec2 a, Vec2 b, double u) { return a + (b - a) * u; }

}  // namespace whitewater

#endif  // WHITEWATER_CORE_GEOM_VEC2_H
