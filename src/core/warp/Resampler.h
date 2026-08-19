// Vendored from warp-drive 887a123, src/core/warp/Resampler.h. MTI Film internal.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

// The backward warp: destination pixels, one at a time, pulled from the source.
//
// For each destination pixel the renderer asks the WarpMap where that pixel comes from and
// filters the source there. Because the whole pipeline is posed destination -> source there
// is no scatter, no inversion and no hole filling -- every destination pixel gets exactly
// one answer, which is the entire reason for the direction convention.
//
// Alpha is the part of this file that is easy to get wrong, so it is stated up front.
// Flame hands us UNPREMULTIPLIED RGBA. Filtering unpremultiplied colour is wrong wherever
// alpha varies: a pixel with alpha zero still carries some colour, and a weighted average
// of colours that ignores their coverage drags that colour into its neighbours. That is
// where matte halos come from. The fix is not subtle -- premultiply, filter, divide the
// filtered colour by the filtered alpha -- but it has to be applied deliberately, so
// AlphaMode has no default value anywhere in this file. A caller that has not thought about
// it does not compile.
//
// Threading. resampleRows renders a half-open range of destination rows and nothing else:
// it reads the source and the warp, writes only the rows it was given, and keeps no state
// between rows or between calls. Splitting an image across threads by row range is
// therefore safe and gives bit-identical results to rendering it in one call, because no
// pixel's value depends on any other pixel's. The one precondition is on the caller's side:
// the WarpMap must already be fully built. BezierSpline's arc-length cache is lazy and not
// thread safe, so anything that reaches it -- correspondence extraction -- has to have run
// before the threads start. Solving a TpsWarp or building a DisplacementField does that,
// and both results are immutable.
//
// Coordinates. Both images live in one coordinate space, which is the space the warp is
// expressed in. A geometry record says where each image's (0, 0) pixel sits in it, so a
// tiled render -- which is what Flame will give us, since tiles are on -- can render a
// window of the destination from a differently placed window of the source without any
// coordinate arithmetic leaking into the caller. Pixel (x, y) of an image whose origin is
// o has its centre at o + (x + 0.5, y + 0.5).

#ifndef WHITEWATER_CORE_WARP_RESAMPLER_H
#define WHITEWATER_CORE_WARP_RESAMPLER_H

#include <string>

#include "core/geom/Vec2.h"
#include "core/image/Image.h"
#include "core/warp/WarpMap.h"

namespace whitewater {

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

enum class ResampleFilter {
  kNearest,     // no interpolation; the pixel whose centre is closest
  kBilinear,    // 2x2, linear in each axis
  kCatmullRom,  // 4x4 cubic; interpolating, mildly sharpening, and it rings
};

// What happens to a tap that falls outside the source.
enum class EdgeMode {
  kBlack,   // transparent black: the tap contributes nothing at all, alpha included
  kClamp,   // the nearest edge pixel, repeated outwards
  kMirror,  // reflected about the edge pixel *without* repeating it, so the sequence
            // running off the left of a four-pixel row is 0 1 2 3 2 1 0 1 2 3. Repeating
            // the edge pixel instead would put a step in the first derivative right at the
            // boundary, which a cubic filter turns into a visible line.
};

// How the caller's pixels are stored -- on both sides, since the destination is written
// back in the same form the source was read in.
enum class AlphaMode {
  // Colour already carries its own coverage, so the four channels are linear and can be
  // filtered directly. This is the cheap path and the only correct one for premultiplied
  // data.
  kPremultiplied,

  // Colour is independent of coverage, which is what Flame gives us. Taps are premultiplied
  // before filtering and the result is divided by the filtered alpha on the way out, so a
  // transparent pixel's colour contributes in proportion to nothing, which is the point.
  kUnpremultiplied,
};

const char *resampleFilterName(ResampleFilter filter);
bool resampleFilterFromName(const std::string &name, ResampleFilter *filter);
const char *edgeModeName(EdgeMode mode);
bool edgeModeFromName(const std::string &name, EdgeMode *mode);
const char *alphaModeName(AlphaMode mode);
bool alphaModeFromName(const std::string &name, AlphaMode *mode);

struct ResampleOptions {
  // Alpha has to be passed. There is no default constructor and no default member value,
  // because either possible default is silently wrong for half the callers and the symptom
  // -- a faint fringe along matte edges -- is exactly the kind of thing that survives
  // review and gets found in a grade.
  explicit ResampleOptions(AlphaMode alphaMode) : alpha(alphaMode) {}
  ResampleOptions() = delete;

  AlphaMode alpha;
  ResampleFilter filter = ResampleFilter::kBilinear;
  EdgeMode edge = EdgeMode::kBlack;
};

// Where the two images sit in the warp's coordinate space: the position of the lower-left
// corner of each image's (0, 0) pixel. Both default to the origin, which is the whole-frame
// case.
//
// Render scale is deliberately absent. A scaled render is a different warp, not a different
// resample, and the displacement field is already keyed on render scale -- so the caller
// builds a map in pixel coordinates at the scale it is rendering, and this code never has
// to know that scaling exists.
struct ResampleGeometry {
  Vec2 destinationOrigin;
  Vec2 sourceOrigin;
};

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

// The tap layout of a filter, exposed so that partition of unity can be asserted directly
// rather than inferred from rendered pixels.
//
// Nearest is described here as a two-tap filter carrying all of its weight on one tap. That
// is not a fiction for the sake of a uniform interface -- it is how the sampler implements
// it, so that every filter shares one accumulation loop and one set of edge rules, and so
// that "the weights sum to one" is a statement about all three filters and not two.
int resampleFilterTapCount(ResampleFilter filter);

// Offset of tap 0 from the pixel index the sample falls in: 0 for the two-tap filters, -1
// for Catmull-Rom.
int resampleFilterFirstTap(ResampleFilter filter);

// Fills resampleFilterTapCount(filter) weights for a sample at the given fractional phase
// within a pixel, which must be in [0, 1).
void resampleFilterWeights(ResampleFilter filter, double phase, double *weights);

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// One filtered sample, in the source image's own pixel coordinates -- pixel (x, y)'s centre
// is at (x + 0.5, y + 0.5). Writes four floats. This is the whole of the per-pixel work the
// renderer does, factored out so the filter and edge-mode properties can be tested without
// a warp in the way.
//
// A sample landing exactly on a pixel centre returns that pixel verbatim, byte for byte,
// including its colour where alpha is zero. Every filter here reduces to a single unit tap
// at that position, so this is the arithmetic answer and not a shortcut -- but it is worth
// naming, because it is what makes an identity warp an exact copy even in the
// unpremultiplied mode, where multiplying by alpha and dividing it out again is otherwise
// not an identity in floating point.
void sampleImage(const ConstImageView &source, Vec2 position, const ResampleOptions &options,
                 float *rgba);

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Renders destination rows [firstRow, firstRow + rowCount). Rows outside the destination
// are ignored rather than being an error, so a caller splitting rows between threads can
// use a round number of rows per thread without a special case for the last one.
void resampleRows(const ConstImageView &source, const WarpMap &warp,
                  const ResampleOptions &options, const ResampleGeometry &geometry,
                  const ImageView &destination, int firstRow, int rowCount);

// Renders every destination row, splitting the row range across worker threads. The split is
// bit-identical to the serial render for every warp, filter, edge mode and alpha mode: the
// partition is fixed and contiguous (see resampleRows), so a pixel is computed by the same
// arithmetic in the same order whatever the worker count. Rows are independent, so the shared
// destination needs no locking.
//
// threadCount selects the parallelism. The precondition for any value above 1 is the same one
// stated at the top of this file: the WarpMap must already be fully built.
//   0  -- automatic: one worker per hardware thread, dropping to serial for a small frame or a
//         single core. This is the offline tool's and the editor preview's default.
//   1  -- serial, on the calling thread. What a caller that already parallelizes a level up
//         forces: the OFX plugin splits rows across the host's own thread suite and calls
//         resampleRows directly, so it never nests a pool here, and a determinism test pins
//         this against the automatic path.
//   n  -- at most n workers (clamped to the row count).
void resample(const ConstImageView &source, const WarpMap &warp,
              const ResampleOptions &options, const ResampleGeometry &geometry,
              const ImageView &destination, int threadCount = 0);

}  // namespace whitewater

#endif  // WHITEWATER_CORE_WARP_RESAMPLER_H
