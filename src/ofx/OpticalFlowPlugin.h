// The Phase 1 OFX boundary.
//
// This class owns no flow state yet.  It freezes the host-facing contract and renders the
// documented deterministic fallbacks until the host-free flow algebra and inference layers
// arrive in later phases.

#ifndef WHITEWATER_OFX_OPTICALFLOWPLUGIN_H
#define WHITEWATER_OFX_OPTICALFLOWPLUGIN_H

#include <string>

#include "ofx/FlowParameters.h"
#include "ofx/HostImage.h"
#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

inline constexpr const char *kOpticalFlowPluginIdentifier =
    "com.mtifilm.whitewater.opticalflow";
inline constexpr const char *kStMapPluginIdentifier = "com.mtifilm.whitewater.stmap";

class OpticalFlowPlugin : public OFX::ImageEffect {
 public:
  OpticalFlowPlugin(OfxImageEffectHandle handle, DescriptorKind kind);

  void render(const OFX::RenderArguments &args) override;
  bool isIdentity(const OFX::IsIdentityArguments &args, OFX::Clip *&identityClip,
                  double &identityTime) override;
  void getRegionsOfInterest(const OFX::RegionsOfInterestArguments &args,
                            OFX::RegionOfInterestSetter &rois) override;
  void getFramesNeeded(const OFX::FramesNeededArguments &args,
                       OFX::FramesNeededSetter &frames) override;
  void getClipPreferences(OFX::ClipPreferencesSetter &clipPreferences) override;
  void changedParam(const OFX::InstanceChangedArgs &args,
                    const std::string &paramName) override;

 private:
  void renderBlack(const OFX::RenderArguments &args,
                   const HostDestinationImage &destination);
  void renderStMap(const OFX::RenderArguments &args, OFX::Image &destination,
                   const HostDestinationImage &destinationImage);

  DescriptorKind kind_;
  FlowParameters parameters_;
  OFX::Clip *sourceClip_ = nullptr;
  OFX::Clip *insertClip_ = nullptr;
  OFX::Clip *outputClip_ = nullptr;
};

// The two factories intentionally remain separate.  Their identifiers are permanent, while
// the implementation and parameter reader are shared.
mDeclarePluginFactory(OpticalFlowPluginFactory, ;, ;);
mDeclarePluginFactory(StMapPluginFactory, ;, ;);

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_OPTICALFLOWPLUGIN_H
