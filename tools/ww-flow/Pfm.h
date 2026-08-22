#ifndef WHITEWATER_TOOLS_WW_FLOW_PFM_H
#define WHITEWATER_TOOLS_WW_FLOW_PFM_H

#include <string>
#include <vector>

namespace whitewater::tool {

struct PfmImage {
  int width = 0;
  int height = 0;
  int channels = 0;
  std::vector<float> pixels;

  bool isEmpty() const {
    return width <= 0 || height <= 0 || (channels != 1 && channels != 3) || pixels.empty();
  }
};

bool readPfm(const std::string &path, PfmImage *image, std::string *error);
bool writePfm(const std::string &path, const PfmImage &image, std::string *error);

}  // namespace whitewater::tool

#endif  // WHITEWATER_TOOLS_WW_FLOW_PFM_H
