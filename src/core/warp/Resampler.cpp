// Vendored from warp-drive 887a123, src/core/warp/Resampler.cpp.
//
// Copied rather than shared: the two plugins ship on independent cadences and a
// submodule would couple them. A fix on either side is ported by hand, deliberately.
// Only the namespace and include guards were changed.

#include "core/warp/Resampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <thread>
#include <vector>

namespace whitewater {

namespace {

// Sample positions are clamped to this before being turned into integer pixel indices. A
// warp that has gone numerically wild -- or simply a document authored for a much larger
// frame -- must not be able to overflow the conversion, and every edge mode gives the same
// answer at 1e9 pixels outside the image as it does at 1e18.
constexpr double kCoordinateLimit = 1.0e9;

inline double clampTo(double value, double low, double high) {
  return value < low ? low : (value > high ? high : value);
}

// Catmull-Rom, the same basis the displacement field interpolates with. Duplicated rather
// than shared: there it walks a grid of displacements and here it walks pixels, and a
// shared header for six lines of polynomial would tie two inner loops together for no gain.
inline void catmullRomWeights(double f, double *w) {
  const double f2 = f * f;
  const double f3 = f2 * f;
  w[0] = -0.5 * f3 + f2 - 0.5 * f;
  w[1] = 1.5 * f3 - 2.5 * f2 + 1.0;
  w[2] = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
  w[3] = 0.5 * f3 - 0.5 * f2;
}

// The index a tap resolves to, or -1 when it contributes nothing. Returning a sentinel
// rather than a colour is what keeps kBlack from needing its own accumulation loop: an
// absent tap is skipped, which is arithmetically identical to adding weight * 0.
inline int edgeIndex(int index, int size, EdgeMode mode) {
  if (index >= 0 && index < size) {
    return index;
  }
  switch (mode) {
    case EdgeMode::kBlack:
      return -1;
    case EdgeMode::kClamp:
      return index < 0 ? 0 : size - 1;
    case EdgeMode::kMirror: {
      // Reflect about the edge pixel without repeating it, so the period is 2 * (size - 1)
      // and a one-pixel image degenerates to its only pixel rather than to a division by
      // zero.
      if (size == 1) {
        return 0;
      }
      const int period = 2 * (size - 1);
      int folded = index % period;
      if (folded < 0) {
        folded += period;
      }
      return folded < size ? folded : period - folded;
    }
  }
  return -1;
}

}  // namespace

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------

const char *resampleFilterName(ResampleFilter filter) {
  switch (filter) {
    case ResampleFilter::kNearest: return "nearest";
    case ResampleFilter::kBilinear: return "bilinear";
    case ResampleFilter::kCatmullRom: return "catmull-rom";
  }
  return "bilinear";
}

bool resampleFilterFromName(const std::string &name, ResampleFilter *filter) {
  ResampleFilter parsed = ResampleFilter::kBilinear;
  if (name == "nearest") {
    parsed = ResampleFilter::kNearest;
  } else if (name == "bilinear") {
    parsed = ResampleFilter::kBilinear;
  } else if (name == "catmull-rom" || name == "bicubic") {
    parsed = ResampleFilter::kCatmullRom;
  } else {
    return false;
  }
  if (filter != nullptr) {
    *filter = parsed;
  }
  return true;
}

const char *edgeModeName(EdgeMode mode) {
  switch (mode) {
    case EdgeMode::kBlack: return "black";
    case EdgeMode::kClamp: return "clamp";
    case EdgeMode::kMirror: return "mirror";
  }
  return "black";
}

bool edgeModeFromName(const std::string &name, EdgeMode *mode) {
  EdgeMode parsed = EdgeMode::kBlack;
  if (name == "black") {
    parsed = EdgeMode::kBlack;
  } else if (name == "clamp") {
    parsed = EdgeMode::kClamp;
  } else if (name == "mirror") {
    parsed = EdgeMode::kMirror;
  } else {
    return false;
  }
  if (mode != nullptr) {
    *mode = parsed;
  }
  return true;
}

const char *alphaModeName(AlphaMode mode) {
  switch (mode) {
    case AlphaMode::kPremultiplied: return "premultiplied";
    case AlphaMode::kUnpremultiplied: return "unpremultiplied";
  }
  return "unpremultiplied";
}

bool alphaModeFromName(const std::string &name, AlphaMode *mode) {
  AlphaMode parsed = AlphaMode::kUnpremultiplied;
  if (name == "premultiplied" || name == "premult") {
    parsed = AlphaMode::kPremultiplied;
  } else if (name == "unpremultiplied" || name == "unpremult" || name == "straight") {
    parsed = AlphaMode::kUnpremultiplied;
  } else {
    return false;
  }
  if (mode != nullptr) {
    *mode = parsed;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

int resampleFilterTapCount(ResampleFilter filter) {
  return filter == ResampleFilter::kCatmullRom ? 4 : 2;
}

int resampleFilterFirstTap(ResampleFilter filter) {
  return filter == ResampleFilter::kCatmullRom ? -1 : 0;
}

void resampleFilterWeights(ResampleFilter filter, double phase, double *weights) {
  switch (filter) {
    case ResampleFilter::kNearest:
      // Half-way rounds up, matching std::floor(t + 0.5) rather than a banker's rule --
      // consistency with the other filters matters more than the tie direction, since at
      // phase exactly 0.5 the two answers are equally defensible.
      weights[0] = phase < 0.5 ? 1.0 : 0.0;
      weights[1] = phase < 0.5 ? 0.0 : 1.0;
      return;
    case ResampleFilter::kBilinear:
      weights[0] = 1.0 - phase;
      weights[1] = phase;
      return;
    case ResampleFilter::kCatmullRom:
      catmullRomWeights(phase, weights);
      return;
  }
  weights[0] = 1.0;
  weights[1] = 0.0;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

void sampleImage(const ConstImageView &source, Vec2 position, const ResampleOptions &options,
                 float *rgba) {
  rgba[0] = 0.0f;
  rgba[1] = 0.0f;
  rgba[2] = 0.0f;
  rgba[3] = 0.0f;

  if (source.isEmpty() || !std::isfinite(position.x) || !std::isfinite(position.y)) {
    return;
  }

  // Into pixel-index space: a pixel's centre is half a pixel past its corner, so the whole
  // of the sampling offset convention is this one subtraction. Getting it wrong by that
  // half pixel is invisible in a still frame and shows up as softness and drift in a shot,
  // which is why the identity test in the suite demands bit equality rather than closeness.
  const double tx = clampTo(position.x - 0.5, -kCoordinateLimit, kCoordinateLimit);
  const double ty = clampTo(position.y - 0.5, -kCoordinateLimit, kCoordinateLimit);
  const double baseX = std::floor(tx);
  const double baseY = std::floor(ty);
  const double phaseX = tx - baseX;
  const double phaseY = ty - baseY;
  const int pixelX = static_cast<int>(baseX);
  const int pixelY = static_cast<int>(baseY);

  // Exactly on a pixel centre: every filter here collapses to a unit weight on that one
  // pixel, so copy it through. Doing the general accumulation would give the same colour
  // for premultiplied data, but in the unpremultiplied mode it would multiply by alpha and
  // divide it back out -- which is not an identity in floating point, and would leave an
  // identity warp very slightly not equal to its input.
  if (phaseX == 0.0 && phaseY == 0.0) {
    const int x = edgeIndex(pixelX, source.width(), options.edge);
    const int y = edgeIndex(pixelY, source.height(), options.edge);
    if (x < 0 || y < 0) {
      return;
    }
    std::memcpy(rgba, source.pixel(x, y), sizeof(float) * kImageChannels);
    return;
  }

  const int tapCount = resampleFilterTapCount(options.filter);
  const int firstTap = resampleFilterFirstTap(options.filter);

  double weightX[4];
  double weightY[4];
  resampleFilterWeights(options.filter, phaseX, weightX);
  resampleFilterWeights(options.filter, phaseY, weightY);

  int columns[4];
  int rows[4];
  for (int k = 0; k < tapCount; ++k) {
    columns[k] = edgeIndex(pixelX + firstTap + k, source.width(), options.edge);
    rows[k] = edgeIndex(pixelY + firstTap + k, source.height(), options.edge);
  }

  const bool premultiply = options.alpha == AlphaMode::kUnpremultiplied;

  // Accumulated in double. The taps are floats and there are at most sixteen of them, so
  // this costs nothing measurable and removes cancellation as a source of difference
  // between the filters when they are compared against each other.
  double accumulated[kImageChannels] = {0.0, 0.0, 0.0, 0.0};
  for (int j = 0; j < tapCount; ++j) {
    if (rows[j] < 0) {
      continue;
    }
    const float *line = source.row(rows[j]);
    double rowAccumulated[kImageChannels] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < tapCount; ++i) {
      if (columns[i] < 0) {
        continue;
      }
      const float *tap = line + static_cast<std::ptrdiff_t>(columns[i]) * kImageChannels;
      const double weight = weightX[i];
      if (premultiply) {
        const double coverage = static_cast<double>(tap[3]);
        const double scaled = weight * coverage;
        rowAccumulated[0] += scaled * static_cast<double>(tap[0]);
        rowAccumulated[1] += scaled * static_cast<double>(tap[1]);
        rowAccumulated[2] += scaled * static_cast<double>(tap[2]);
        rowAccumulated[3] += scaled;
      } else {
        rowAccumulated[0] += weight * static_cast<double>(tap[0]);
        rowAccumulated[1] += weight * static_cast<double>(tap[1]);
        rowAccumulated[2] += weight * static_cast<double>(tap[2]);
        rowAccumulated[3] += weight * static_cast<double>(tap[3]);
      }
    }
    const double weight = weightY[j];
    for (int c = 0; c < kImageChannels; ++c) {
      accumulated[c] += weight * rowAccumulated[c];
    }
  }

  if (premultiply) {
    const double coverage = accumulated[3];
    // Catmull-Rom rings, so a filtered alpha can land slightly below zero next to a hard
    // matte edge. Dividing by it would produce a wildly wrong, sign-flipped colour in the
    // one place a user is most likely to be looking. Where there is no coverage there is
    // no colour information to recover either, so zero is both the safe answer and the
    // correct one.
    if (coverage > 0.0) {
      const double inverse = 1.0 / coverage;
      for (int c = 0; c < 3; ++c) {
        // Unpremultiplying by a very small coverage amplifies, and with a cubic's negative
        // lobes the coverage can very nearly cancel while the colour does not. The result
        // is still the honest answer -- multiplied back by that coverage it is the
        // correctly filtered premultiplied value, which is why it is not clamped -- but it
        // can in principle overflow a float, and an infinity in an image poisons every
        // operation downstream of it. Where there is no representable colour there is no
        // usable colour either.
        const float converted = static_cast<float>(accumulated[c] * inverse);
        rgba[c] = std::isfinite(converted) ? converted : 0.0f;
      }
    }
    rgba[3] = static_cast<float>(coverage);
    return;
  }

  for (int c = 0; c < kImageChannels; ++c) {
    rgba[c] = static_cast<float>(accumulated[c]);
  }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void resampleRows(const ConstImageView &source, const WarpMap &warp,
                  const ResampleOptions &options, const ResampleGeometry &geometry,
                  const ImageView &destination, int firstRow, int rowCount) {
  if (destination.isEmpty() || rowCount <= 0) {
    return;
  }
  const int begin = std::max(firstRow, 0);
  const int end = std::min(firstRow + rowCount, destination.height());

  const double centreX = geometry.destinationOrigin.x + 0.5;
  const double centreY = geometry.destinationOrigin.y + 0.5;
  const int width = destination.width();

  for (int y = begin; y < end; ++y) {
    const double destinationY = centreY + static_cast<double>(y);
    float *out = destination.row(y);
    for (int x = 0; x < width; ++x, out += kImageChannels) {
      const Vec2 sourcePoint =
          warp.mapToSource(Vec2(centreX + static_cast<double>(x), destinationY));
      sampleImage(source, sourcePoint - geometry.sourceOrigin, options, out);
    }
  }
}

namespace {

// Rendering below this many destination pixels stays on the calling thread: for a frame this
// small the thread launch and join cost more than the work they save, and the editor's draft
// pass and thumbnail renders land here. Only consulted for the automatic count -- a caller
// that names a worker count is honoured.
constexpr long long kSerialPixelThreshold = 64LL * 1024LL;

// Joining is a scope obligation, not only the happy-path epilogue. In particular,
// std::thread's destructor terminates the process while it is still joinable, so a caller-slice
// exception or a failure to launch a later worker must join every worker that did start before
// it can propagate. Calling join() explicitly on success also lets the caller inspect worker
// exceptions only after every per-worker slot is complete.
class ThreadJoinGuard {
 public:
  explicit ThreadJoinGuard(std::vector<std::thread> &threads) : threads_(threads) {}
  ~ThreadJoinGuard() { join(); }

  void join() {
    for (std::thread &thread : threads_) {
      if (thread.joinable()) thread.join();
    }
  }

 private:
  std::vector<std::thread> &threads_;
};

// How many workers resample() actually launches for a given request. See resample()'s header
// for the threadCount contract; the height clamp keeps a worker from being handed no rows.
int resolveWorkerCount(int threadCount, int width, int height) {
  if (height <= 1 || threadCount == 1) {
    return 1;
  }
  int workers;
  if (threadCount > 1) {
    // A caller-forced count. Honoured as given, only clamped so a worker always has a row.
    workers = threadCount;
  } else {
    // Automatic: one worker per hardware thread, but never split a frame so small the launch
    // cost dominates the render.
    if (static_cast<long long>(width) * static_cast<long long>(height) < kSerialPixelThreshold) {
      return 1;
    }
    const unsigned int hardware = std::thread::hardware_concurrency();
    workers = hardware == 0 ? 1 : static_cast<int>(hardware);
  }
  return std::max(1, std::min(workers, height));
}

}  // namespace

void resample(const ConstImageView &source, const WarpMap &warp,
              const ResampleOptions &options, const ResampleGeometry &geometry,
              const ImageView &destination, int threadCount) {
  if (destination.isEmpty()) {
    return;
  }
  const int height = destination.height();
  const int workers = resolveWorkerCount(threadCount, destination.width(), height);
  if (workers <= 1) {
    resampleRows(source, warp, options, geometry, destination, 0, height);
    return;
  }

  // A fixed, contiguous row partition -- deliberately not work-stealing over pixels. Each
  // destination row is rendered by exactly one worker, and every pixel runs the identical
  // per-pixel arithmetic in the identical order it would on the serial path, so the result
  // is bit-identical whatever the worker count. Rows are independent -- resampleRows writes
  // only the rows it is given and reads nothing it writes -- so the shared destination needs
  // no locking. The one precondition is on the caller: the WarpMap must be fully built, since
  // a lazily populated map (BezierSpline's arc-length cache) is not thread safe. Solving a
  // warp or building a displacement field satisfies that, and both results are immutable.
  const int rowsPerWorker = (height + workers - 1) / workers;
  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(workers - 1));
  // Each worker owns one exception slot, so recording failures needs no mutex. The slots are
  // inspected only after the join establishes that every write has completed.
  std::vector<std::exception_ptr> workerErrors(static_cast<std::size_t>(workers - 1));
  // Declared after workerErrors so it joins before those captured exception slots are destroyed
  // during unwinding.
  ThreadJoinGuard joinGuard(pool);
  // Every worker but the last runs on its own thread; the last slice runs here on the calling
  // thread, so the common single-core case never spawns and one core does no idle launching.
  for (int w = 0; w + 1 < workers; ++w) {
    const int firstRow = w * rowsPerWorker;
    pool.emplace_back([&source, &warp, &options, &geometry, &destination, &workerErrors, w,
                       firstRow, rowsPerWorker]() {
      try {
        resampleRows(source, warp, options, geometry, destination, firstRow, rowsPerWorker);
      } catch (...) {
        workerErrors[static_cast<std::size_t>(w)] = std::current_exception();
      }
    });
  }
  resampleRows(source, warp, options, geometry, destination,
               (workers - 1) * rowsPerWorker, rowsPerWorker);
  joinGuard.join();
  // Report the first failed row range deterministically. By this point every worker is joined,
  // so rethrowing cannot unwind through a joinable std::thread.
  for (const std::exception_ptr &error : workerErrors) {
    if (error) std::rethrow_exception(error);
  }
}

}  // namespace whitewater
