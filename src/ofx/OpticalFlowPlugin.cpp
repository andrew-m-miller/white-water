#include "ofx/OpticalFlowPlugin.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>

#include "ofx/HostImage.h"
#include "ofx/HostQuirks.h"
#include "ofxGPURender.h"
#include "ofxsLog.h"
#include "ofxsMultiThread.h"

namespace whitewater {
namespace ofx {
namespace {

constexpr int kRowsPerAbortCheck = 16;

void logLine(const char *message) {
  std::fprintf(stderr, "White Water: %s\n", message);
}

void logFormatFailure(const char *operation, const std::string &error) {
  std::fprintf(stderr, "White Water: %s fallback (%s)\n", operation, error.c_str());
}

class IdentityMap final : public WarpMap {
 public:
  Vec2 mapToSource(Vec2 destination) const override { return destination; }
};

// Writes either an identity copy or transparent black in row ranges.  The source and map
// are immutable before the host thread suite starts, so each worker only touches its own
// destination rows.  The explicit 16-row abort cadence is part of the Phase 1 host contract.
class FallbackRenderer final : public OFX::MultiThread::Processor {
 public:
  FallbackRenderer(OFX::ImageEffect &effect, const HostDestinationImage &destination,
                   const HostSourceImage *source, AlphaMode alphaMode)
      : effect_(effect),
        destination_(destination),
        source_(source),
        options_(alphaMode),
        geometry_(),
        copy_(source != nullptr && !source->view().isEmpty()) {
    if (copy_) {
      geometry_.destinationOrigin = destination_.origin();
      geometry_.sourceOrigin = source_->origin();
    }
  }

  void multiThreadFunction(unsigned int threadId, unsigned int threadCount) override {
    const int height = destination_.view().height();
    const int workers = std::max(1u, threadCount);
    const int rowsPerWorker = (height + static_cast<int>(workers) - 1) /
                              static_cast<int>(workers);
    const int first = static_cast<int>(threadId) * rowsPerWorker;
    const int last = std::min(height, first + rowsPerWorker);

    for (int row = first; row < last; row += kRowsPerAbortCheck) {
      if (effect_.abort()) return;
      const int count = std::min(kRowsPerAbortCheck, last - row);
      if (copy_) {
        resampleRows(source_->view(), identityMap_, options_, geometry_, destination_.view(), row,
                     count);
      } else {
        fillRows(destination_.view(), row, count);
      }
      destination_.writeBackRows(row, count);
    }
  }

 private:
  static void fillRows(const ImageView &view, int firstRow, int rowCount) {
    const int begin = std::max(0, firstRow);
    const int end = std::min(view.height(), firstRow + rowCount);
    for (int y = begin; y < end; ++y) {
      float *row = view.row(y);
      std::fill(row, row + static_cast<std::ptrdiff_t>(view.width()) * kImageChannels, 0.0f);
    }
  }

  OFX::ImageEffect &effect_;
  const HostDestinationImage &destination_;
  const HostSourceImage *source_;
  ResampleOptions options_;
  ResampleGeometry geometry_;
  bool copy_;
  IdentityMap identityMap_;
};

class StIdentityRenderer final : public OFX::MultiThread::Processor {
 public:
  StIdentityRenderer(OFX::ImageEffect &effect, const HostDestinationImage &destination,
                     int sourceX, int sourceY, int sourceWidth, int sourceHeight, int mode,
                     int origin)
      : effect_(effect),
        destination_(destination),
        sourceX_(sourceX),
        sourceY_(sourceY),
        sourceWidth_(sourceWidth),
        sourceHeight_(sourceHeight),
        relative_(mode == 1),
        topLeft_(origin == 1) {}

  void multiThreadFunction(unsigned int threadId, unsigned int threadCount) override {
    const ImageView view = destination_.view();
    const int height = view.height();
    const int workers = std::max(1u, threadCount);
    const int rowsPerWorker = (height + static_cast<int>(workers) - 1) /
                              static_cast<int>(workers);
    const int first = static_cast<int>(threadId) * rowsPerWorker;
    const int last = std::min(height, first + rowsPerWorker);

    for (int row = first; row < last; row += kRowsPerAbortCheck) {
      if (effect_.abort()) return;
      const int count = std::min(kRowsPerAbortCheck, last - row);
      writeRows(view, row, count);
      destination_.writeBackRows(row, count);
    }
  }

 private:
  void writeRows(const ImageView &view, int firstRow, int rowCount) const {
    const int begin = std::max(0, firstRow);
    const int end = std::min(view.height(), firstRow + rowCount);
    for (int y = begin; y < end; ++y) {
      float *out = view.row(y);
      const double globalY = destination_.origin().y + static_cast<double>(y);
      for (int x = 0; x < view.width(); ++x) {
        const double globalX = destination_.origin().x + static_cast<double>(x);
        float u = 0.0f;
        float v = 0.0f;
        if (!relative_) {
          const double localX = globalX - static_cast<double>(sourceX_);
          const double localY = globalY - static_cast<double>(sourceY_);
          u = static_cast<float>((localX + 0.5) / static_cast<double>(sourceWidth_));
          const double bottomLeftV =
              (localY + 0.5) / static_cast<double>(sourceHeight_);
          v = static_cast<float>(topLeft_ ? 1.0 - bottomLeftV : bottomLeftV);
        }

        out[0] = u;
        out[1] = v;
        // B is deliberately zero.  Confidence is a separate future output, not an implicit
        // channel, and alpha 1 makes the RG coordinate pair a normal opaque float image.
        out[2] = 0.0f;
        out[3] = 1.0f;
        out += kImageChannels;
      }
    }
  }

  OFX::ImageEffect &effect_;
  const HostDestinationImage &destination_;
  int sourceX_;
  int sourceY_;
  int sourceWidth_;
  int sourceHeight_;
  bool relative_;
  bool topLeft_;
};

template <typename Processor>
void runRows(Processor &processor, int height) {
  const unsigned int workers = std::max(
      1u, std::min(OFX::MultiThread::getNumCPUs(), static_cast<unsigned int>(height)));
  processor.multiThread(workers);
}

bool validBounds(const OfxRectI &bounds) {
  return bounds.x2 > bounds.x1 && bounds.y2 > bounds.y1;
}

bool isConnectedSafely(const OFX::Clip *clip) {
  if (clip == nullptr) return false;
  try {
    return clip->isConnected();
  } catch (...) {
    return false;
  }
}

void describeEffect(OFX::ImageEffectDescriptor &descriptor, DescriptorKind kind) {
  if (kind == DescriptorKind::kTrack) {
    descriptor.setLabels("White Water", "White Water", "White Water optical flow tracker");
    descriptor.setPluginDescription(
        "Learned optical-flow tracking for Flame. Phase 1 renders the deterministic fallback "
        "contract while the inference pipeline is built.");
    descriptor.addSupportedBitDepth(OFX::eBitDepthUByte);
    descriptor.addSupportedBitDepth(OFX::eBitDepthUShort);
    descriptor.addSupportedBitDepth(OFX::eBitDepthHalf);
    descriptor.addSupportedBitDepth(OFX::eBitDepthFloat);
  } else {
    descriptor.setLabels("WW ST Map", "WW ST Map", "White Water ST map");
    descriptor.setPluginDescription(
        "Float ST-map output from learned optical-flow tracking. Phase 1 emits an exact "
        "identity map while the inference pipeline is built.");
    descriptor.addSupportedBitDepth(OFX::eBitDepthFloat);
  }

  descriptor.addSupportedContext(OFX::eContextGeneral);
  descriptor.setSingleInstance(false);
  descriptor.setHostFrameThreading(false);
  descriptor.setSupportsMultiResolution(false);
  descriptor.setSupportsTiles(false);
  descriptor.setTemporalClipAccess(true);
  descriptor.setRenderTwiceAlways(false);
  descriptor.setSupportsMultipleClipDepths(false);
  descriptor.setSupportsMultipleClipPARs(false);
  descriptor.setRenderThreadSafety(OFX::eRenderInstanceSafe);

  // Flame's OpenGL capability property is a string and is optional across OFX hosts.  The
  // support-library setter throws for an unknown property, which would make this plugin
  // disappear from Flame, so write it directly with throwOnFailure=false.
  descriptor.getPropertySet().propSetString(kOfxImageEffectPropOpenGLRenderSupported, "false", 0,
                                             false);
}

void describeClips(OFX::ImageEffectDescriptor &descriptor, DescriptorKind kind) {
  OFX::ClipDescriptor *source =
      descriptor.defineClip(kOfxImageEffectSimpleSourceClipName);
  source->setLabels("Source", "Source", "Source");
  source->addSupportedComponent(OFX::ePixelComponentRGBA);
  if (kind == DescriptorKind::kTrack) {
    source->addSupportedComponent(OFX::ePixelComponentRGB);
    source->addSupportedComponent(OFX::ePixelComponentAlpha);
  }
  source->setTemporalClipAccess(true);
  source->setSupportsTiles(false);
  source->setIsMask(false);

  OFX::ClipDescriptor *output = descriptor.defineClip(kOfxImageEffectOutputClipName);
  output->setLabels("Output", "Output", "Output");
  output->addSupportedComponent(OFX::ePixelComponentRGBA);
  if (kind == DescriptorKind::kTrack) {
    // SupportsMultipleClipDepths is false, so the Track effect must preserve the host's
    // common clip layout. Advertising RGB and Alpha only on the inputs would make the
    // fallback copy silently change components at the output and break identity.
    output->addSupportedComponent(OFX::ePixelComponentRGB);
    output->addSupportedComponent(OFX::ePixelComponentAlpha);
  }
  output->setSupportsTiles(false);

  if (kind == DescriptorKind::kTrack) {
    OFX::ClipDescriptor *insert = descriptor.defineClip(kInsertClipName);
    insert->setLabels("Insert", "Insert", "Insert");
    insert->addSupportedComponent(OFX::ePixelComponentRGBA);
    insert->addSupportedComponent(OFX::ePixelComponentRGB);
    insert->addSupportedComponent(OFX::ePixelComponentAlpha);
    insert->setTemporalClipAccess(true);
    insert->setSupportsTiles(false);
    insert->setOptional(true);
    insert->setIsMask(false);
  }
}

}  // namespace

OpticalFlowPlugin::OpticalFlowPlugin(OfxImageEffectHandle handle, DescriptorKind kind)
    : OFX::ImageEffect(handle), kind_(kind), parameters_(*this, kind) {
  sourceClip_ = fetchClip(kOfxImageEffectSimpleSourceClipName);
  outputClip_ = fetchClip(kOfxImageEffectOutputClipName);
  if (kind_ == DescriptorKind::kTrack) {
    insertClip_ = fetchClip(kInsertClipName);
  }
}

bool OpticalFlowPlugin::isIdentity(const OFX::IsIdentityArguments &args,
                                   OFX::Clip *&identityClip, double &identityTime) {
  identityClip = nullptr;
  identityTime = args.time;
  if (kind_ == DescriptorKind::kStMap) return false;

  try {
    const FlowParameterValues values = parameters_.routingValuesAt(args.time);
    if (values.output == 0) {
      if (isConnectedSafely(sourceClip_)) {
        identityClip = sourceClip_;
        identityTime = args.time;
        return true;
      }
      return false;
    }

    if (values.output == 1 && isConnectedSafely(insertClip_)) {
      identityClip = insertClip_;
      identityTime = values.insertTime == 1 ? static_cast<double>(values.referenceFrame)
                                            : args.time;
      return true;
    }
  } catch (...) {
    // isIdentity is a query action.  A malformed host property must not make a descriptor
    // disappear, and the conservative answer is to render rather than claim a shortcut.
  }
  return false;
}

void OpticalFlowPlugin::getRegionsOfInterest(const OFX::RegionsOfInterestArguments &args,
                                             OFX::RegionOfInterestSetter &rois) {
  const FlowParameterValues values = parameters_.routingValuesAt(args.time);
  auto setRoi = [&](OFX::Clip *clip, double time) {
    if (clip == nullptr) return;
    if (!clip->isConnected()) return;
    try {
      rois.setRegionOfInterest(*clip, clip->getRegionOfDefinition(time));
    } catch (const std::exception &error) {
      std::fprintf(stderr, "White Water: ROI query used requested window (%s)\n", error.what());
      rois.setRegionOfInterest(*clip, args.regionOfInterest);
    } catch (...) {
      logLine("ROI query used requested window (unknown host failure)");
      rois.setRegionOfInterest(*clip, args.regionOfInterest);
    }
  };

  try {
    setRoi(sourceClip_, args.time);
    if (kind_ == DescriptorKind::kTrack) {
      const double insertTime = values.insertTime == 1
                                    ? static_cast<double>(values.referenceFrame)
                                    : args.time;
      setRoi(insertClip_, insertTime);
    }
  } catch (const std::exception &error) {
    // Query actions must never make a plugin disappear because a host is rebuilding a clip.
    std::fprintf(stderr, "White Water: ROI query fallback (%s)\n", error.what());
  } catch (...) {
    logLine("ROI query fallback (unknown host failure)");
  }
}

void OpticalFlowPlugin::getFramesNeeded(const OFX::FramesNeededArguments &args,
                                        OFX::FramesNeededSetter &frames) {
  try {
    const OfxRangeD current = {args.time, args.time};
    if (sourceClip_ != nullptr) frames.setFramesNeeded(*sourceClip_, current);
    if (kind_ != DescriptorKind::kTrack || insertClip_ == nullptr) return;

    const FlowParameterValues values = parameters_.routingValuesAt(args.time);
    const double insertTime = values.insertTime == 1
                                  ? static_cast<double>(values.referenceFrame)
                                  : args.time;
    frames.setFramesNeeded(*insertClip_, {insertTime, insertTime});
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: frame-needs query fallback (%s)\n", error.what());
  } catch (...) {
    logLine("frame-needs query fallback (unknown host failure)");
  }
}

void OpticalFlowPlugin::getClipPreferences(OFX::ClipPreferencesSetter &clipPreferences) {
  try {
    // Flame's source/output depth is shared (SupportsMultipleClipDepths = 0), so this is an
    // unconditional output preference rather than an attempt to negotiate float ST data from
    // a byte source.  The ST descriptor is float-only and is the separate honest contract.
    clipPreferences.setOutputPremultiplication(OFX::eImageUnPreMultiplied);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: clip-preferences fallback (%s)\n", error.what());
  } catch (...) {
    logLine("clip-preferences fallback (unknown host failure)");
  }
}

void OpticalFlowPlugin::changedParam(const OFX::InstanceChangedArgs &args,
                                     const std::string &paramName) {
  if (paramName != kParamSetReference) return;
  try {
    parameters_.setReferenceFrame(args.time);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: Set Ref could not update refFrame (%s)\n", error.what());
  } catch (...) {
    logLine("Set Ref could not update refFrame (unknown host failure)");
  }
}

void OpticalFlowPlugin::render(const OFX::RenderArguments &args) {
  if (outputClip_ == nullptr) {
    logLine("render fallback (Output clip is unavailable)");
    return;
  }

  std::unique_ptr<OFX::Image> destination;
  try {
    destination.reset(outputClip_->fetchImage(args.time));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: render fallback (Output fetch failed: %s)\n",
                 error.what());
    return;
  } catch (...) {
    logLine("render fallback (Output fetch failed: unknown host failure)");
    return;
  }
  if (destination == nullptr) {
    logLine("render fallback (Output fetch returned no image)");
    return;
  }

  std::string destinationError;
  HostDestinationImage destinationImage;
  try {
    if (!destinationImage.attach(*destination, args.renderWindow, &destinationError)) {
      logFormatFailure("render", destinationError);
      return;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: render fallback (Output attach failed: %s)\n",
                 error.what());
    return;
  } catch (...) {
    logLine("render fallback (Output attach failed: unknown host failure)");
    return;
  }
  if (destinationImage.view().isEmpty()) return;

  if (kind_ == DescriptorKind::kStMap) {
    renderStMap(args, *destination, destinationImage);
    return;
  }

  const FlowParameterValues values = parameters_.routingValuesAt(args.time);
  const bool composite = values.output == 0;
  OFX::Clip *inputClip = composite ? sourceClip_ : insertClip_;
  const double inputTime =
      composite || values.insertTime == 0 ? args.time : static_cast<double>(values.referenceFrame);

  if (inputClip == nullptr || !isConnectedSafely(inputClip)) {
    if (composite) {
      logLine("Composite fallback (Source is disconnected; transparent black rendered)");
    } else {
      logLine("Warped Insert fallback (Insert is disconnected; transparent black rendered)");
    }
    renderBlack(args, destinationImage);
    return;
  }

  std::unique_ptr<OFX::Image> source;
  try {
    source.reset(inputClip->fetchImage(inputTime));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: %s fallback (input fetch failed at %.3f: %s)\n",
                 composite ? "Composite" : "Warped Insert", inputTime, error.what());
    renderBlack(args, destinationImage);
    return;
  } catch (...) {
    std::fprintf(stderr, "White Water: %s fallback (input fetch failed at %.3f)\n",
                 composite ? "Composite" : "Warped Insert", inputTime);
    renderBlack(args, destinationImage);
    return;
  }
  if (source == nullptr) {
    std::fprintf(stderr, "White Water: %s fallback (input fetch returned no image)\n",
                 composite ? "Composite" : "Warped Insert");
    renderBlack(args, destinationImage);
    return;
  }

  HostSourceImage sourceImage;
  std::string sourceError;
  if (!sourceImage.attach(*source, &sourceError) || sourceImage.view().isEmpty()) {
    if (sourceError.empty()) sourceError = "input image has empty bounds";
    std::fprintf(stderr, "White Water: %s fallback (%s)\n",
                 composite ? "Composite" : "Warped Insert", sourceError.c_str());
    renderBlack(args, destinationImage);
    return;
  }

  try {
    std::fprintf(stderr, "White Water: %s Phase 1 fallback at %.3f\n",
                 composite ? "Composite" : "Warped Insert", inputTime);
    FallbackRenderer renderer(*this, destinationImage, &sourceImage,
                              alphaModeFor(source->getPreMultiplication()));
    runRows(renderer, destinationImage.view().height());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: %s fallback (copy failed: %s)\n",
                 composite ? "Composite" : "Warped Insert", error.what());
    renderBlack(args, destinationImage);
  } catch (...) {
    std::fprintf(stderr, "White Water: %s fallback (copy failed: unknown host failure)\n",
                 composite ? "Composite" : "Warped Insert");
    renderBlack(args, destinationImage);
  }
}

void OpticalFlowPlugin::renderBlack(const OFX::RenderArguments &args,
                                    const HostDestinationImage &destination) {
  try {
    FallbackRenderer renderer(*this, destination, nullptr, AlphaMode::kUnpremultiplied);
    runRows(renderer, destination.view().height());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: transparent-black fallback failed (%s)\n", error.what());
  } catch (...) {
    logLine("transparent-black fallback failed (unknown host failure)");
  }
  (void)args;
}

void OpticalFlowPlugin::renderStMap(const OFX::RenderArguments &args,
                                    OFX::Image &destination,
                                    const HostDestinationImage &destinationImage) {
  const OfxRectI destinationBounds = destination.getBounds();
  int sourceX = destinationBounds.x1;
  int sourceY = destinationBounds.y1;
  int sourceWidth = destinationBounds.x2 - destinationBounds.x1;
  int sourceHeight = destinationBounds.y2 - destinationBounds.y1;

  std::unique_ptr<OFX::Image> source;
  if (isConnectedSafely(sourceClip_)) {
    try {
      source.reset(sourceClip_->fetchImage(args.time));
    } catch (const std::exception &error) {
      std::fprintf(stderr,
                   "White Water: ST identity fallback (Source fetch failed; using Output "
                   "geometry: %s)\n",
                   error.what());
    } catch (...) {
      logLine("ST identity fallback (Source fetch failed; using Output geometry)");
    }
    if (source != nullptr && validBounds(source->getBounds())) {
      const OfxRectI bounds = source->getBounds();
      sourceX = bounds.x1;
      sourceY = bounds.y1;
      sourceWidth = bounds.x2 - bounds.x1;
      sourceHeight = bounds.y2 - bounds.y1;
    } else if (source == nullptr) {
      logLine("ST identity fallback (Source image unavailable; using Output geometry)");
    }
  } else {
    logLine("ST identity fallback (Source is disconnected; using Output geometry)");
  }

  if (sourceWidth <= 0 || sourceHeight <= 0) {
    logLine("ST identity fallback (empty geometry)");
    return;
  }

  const FlowParameterValues values = parameters_.routingValuesAt(args.time);
  try {
    std::fprintf(stderr, "White Water: ST identity Phase 1 fallback at %.3f\n", args.time);
    StIdentityRenderer renderer(*this, destinationImage, sourceX, sourceY, sourceWidth,
                                sourceHeight, values.stMode, values.stOrigin);
    try {
      runRows(renderer, destinationImage.view().height());
    } catch (...) {
      // If the host thread suite itself fails after some workers ran, finish the documented
      // identity output serially so the remaining rows are not left undefined.
      renderer.multiThreadFunction(0, 1);
      throw;
    }
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: ST identity fallback failed (%s)\n", error.what());
  } catch (...) {
    logLine("ST identity fallback failed (unknown host failure)");
  }
}

void OpticalFlowPluginFactory::load() {
  resolveHostQuirks();
  reportHostQuirks();
}

void OpticalFlowPluginFactory::unload() {}

void OpticalFlowPluginFactory::describe(OFX::ImageEffectDescriptor &descriptor) {
  try {
    describeEffect(descriptor, DescriptorKind::kTrack);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: Track describe recovered from host failure (%s)\n",
                 error.what());
  } catch (...) {
    logLine("Track describe recovered from unknown host failure");
  }
}

void OpticalFlowPluginFactory::describeInContext(OFX::ImageEffectDescriptor &descriptor,
                                                 OFX::ContextEnum context) {
  if (context != OFX::eContextGeneral) return;
  try {
    describeClips(descriptor, DescriptorKind::kTrack);
    defineFlowParameters(descriptor, DescriptorKind::kTrack);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: Track describeInContext recovered from host failure (%s)\n",
                 error.what());
  } catch (...) {
    logLine("Track describeInContext recovered from unknown host failure");
  }
}

OFX::ImageEffect *OpticalFlowPluginFactory::createInstance(OfxImageEffectHandle handle,
                                                           OFX::ContextEnum) {
  return new OpticalFlowPlugin(handle, DescriptorKind::kTrack);
}

void StMapPluginFactory::load() {
  resolveHostQuirks();
  reportHostQuirks();
}

void StMapPluginFactory::unload() {}

void StMapPluginFactory::describe(OFX::ImageEffectDescriptor &descriptor) {
  try {
    describeEffect(descriptor, DescriptorKind::kStMap);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: ST describe recovered from host failure (%s)\n",
                 error.what());
  } catch (...) {
    logLine("ST describe recovered from unknown host failure");
  }
}

void StMapPluginFactory::describeInContext(OFX::ImageEffectDescriptor &descriptor,
                                           OFX::ContextEnum context) {
  if (context != OFX::eContextGeneral) return;
  try {
    describeClips(descriptor, DescriptorKind::kStMap);
    defineFlowParameters(descriptor, DescriptorKind::kStMap);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "White Water: ST describeInContext recovered from host failure (%s)\n",
                 error.what());
  } catch (...) {
    logLine("ST describeInContext recovered from unknown host failure");
  }
}

OFX::ImageEffect *StMapPluginFactory::createInstance(OfxImageEffectHandle handle,
                                                     OFX::ContextEnum) {
  return new OpticalFlowPlugin(handle, DescriptorKind::kStMap);
}

}  // namespace ofx
}  // namespace whitewater
