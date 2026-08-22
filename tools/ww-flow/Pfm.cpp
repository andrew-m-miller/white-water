#include "Pfm.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <utility>

namespace whitewater::tool {
namespace {

bool hostIsLittleEndian() {
  const std::uint16_t value = 1;
  return *reinterpret_cast<const unsigned char *>(&value) == 1;
}

void swapFloatBytes(float *value) {
  unsigned char bytes[sizeof(float)];
  std::memcpy(bytes, value, sizeof(float));
  std::reverse(bytes, bytes + sizeof(float));
  std::memcpy(value, bytes, sizeof(float));
}

bool fail(const std::string &message, std::string *error) {
  if (error != nullptr) *error = message;
  return false;
}

bool checkedSampleCount(int width, int height, int channels, std::size_t *count) {
  if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) return false;
  const std::size_t w = static_cast<std::size_t>(width);
  const std::size_t h = static_cast<std::size_t>(height);
  const std::size_t c = static_cast<std::size_t>(channels);
  if (w > std::numeric_limits<std::size_t>::max() / h) return false;
  const std::size_t pixels = w * h;
  if (pixels > std::numeric_limits<std::size_t>::max() / c) return false;
  const std::size_t samples = pixels * c;
  const std::size_t maximumStreamBytes =
      static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
  if (samples > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      samples > maximumStreamBytes / sizeof(float))
    return false;
  *count = samples;
  return true;
}

}  // namespace

bool readPfm(const std::string &path, PfmImage *image, std::string *error) {
  if (image == nullptr) return fail("readPfm requires an output image", error);

  std::ifstream stream(path, std::ios::binary);
  if (!stream) return fail("cannot open PFM input: " + path, error);

  std::string magic;
  int width = 0;
  int height = 0;
  float scale = 0.0f;
  stream >> magic >> width >> height >> scale;
  if (!stream || (magic != "PF" && magic != "Pf"))
    return fail("invalid PFM header: " + path, error);
  if (!std::isfinite(scale) || scale == 0.0f)
    return fail("PFM scale must be finite and non-zero: " + path, error);

  const int channels = magic == "PF" ? 3 : 1;
  std::size_t sampleCount = 0;
  if (!checkedSampleCount(width, height, channels, &sampleCount))
    return fail("invalid or overflowing PFM dimensions: " + path, error);

  char separator = '\0';
  stream.get(separator);
  if (!stream || (separator != '\n' && separator != '\r' && separator != ' ' &&
                  separator != '\t'))
    return fail("PFM header is not followed by whitespace: " + path, error);
  if (separator == '\r' && stream.peek() == '\n') stream.get();

  PfmImage decoded;
  decoded.width = width;
  decoded.height = height;
  decoded.channels = channels;
  decoded.pixels.resize(sampleCount);
  stream.read(reinterpret_cast<char *>(decoded.pixels.data()),
              static_cast<std::streamsize>(sampleCount * sizeof(float)));
  if (!stream) return fail("truncated PFM pixels: " + path, error);

  const bool fileIsLittleEndian = scale < 0.0f;
  if (fileIsLittleEndian != hostIsLittleEndian()) {
    for (float &value : decoded.pixels) swapFloatBytes(&value);
  }
  const float magnitude = std::fabs(scale);
  if (magnitude != 1.0f) {
    for (float &value : decoded.pixels) value *= magnitude;
  }

  *image = std::move(decoded);
  if (error != nullptr) error->clear();
  return true;
}

bool writePfm(const std::string &path, const PfmImage &image, std::string *error) {
  std::size_t sampleCount = 0;
  if (!checkedSampleCount(image.width, image.height, image.channels, &sampleCount) ||
      image.pixels.size() != sampleCount)
    return fail("PFM output has inconsistent dimensions or storage", error);

  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) return fail("cannot open PFM output: " + path, error);

  stream << (image.channels == 3 ? "PF\n" : "Pf\n") << image.width << ' ' << image.height
         << "\n-1.0\n";
  if (hostIsLittleEndian()) {
    stream.write(reinterpret_cast<const char *>(image.pixels.data()),
                 static_cast<std::streamsize>(sampleCount * sizeof(float)));
  } else {
    std::vector<float> littleEndian = image.pixels;
    for (float &value : littleEndian) swapFloatBytes(&value);
    stream.write(reinterpret_cast<const char *>(littleEndian.data()),
                 static_cast<std::streamsize>(sampleCount * sizeof(float)));
  }
  if (!stream) return fail("failed while writing PFM output: " + path, error);
  if (error != nullptr) error->clear();
  return true;
}

}  // namespace whitewater::tool
