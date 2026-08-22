// The parameter contract shared by the Track/Insert and ST Map descriptors.
//
// Phase 1 deliberately defines the controls before it defines any inference.  Keeping the
// names, order and typed reads in one place is important here: OFX setup files store choice
// indices, and a later implementation must not accidentally give two descriptors different
// parameter layouts.

#ifndef WHITEWATER_OFX_FLOWPARAMETERS_H
#define WHITEWATER_OFX_FLOWPARAMETERS_H

#include <string>

#include "ofxsImageEffect.h"

namespace whitewater {
namespace ofx {

// Clip names are part of the OFX contract and are kept explicit rather than being inferred
// from a descriptor kind in the render path.
inline constexpr const char *kInsertClipName = "Insert";

// Stable script names.  Do not rename these after the first artist build: Flame persists
// values by script name and saved Choice values by index.
inline constexpr const char *kParamModel = "model";
inline constexpr const char *kParamReferenceFrame = "refFrame";
inline constexpr const char *kParamSetReference = "setRef";
inline constexpr const char *kParamOutput = "output";
inline constexpr const char *kParamInsertTime = "insertTime";
inline constexpr const char *kParamMatte = "matte";
inline constexpr const char *kParamInputCurve = "inputCurve";
inline constexpr const char *kParamAnalysisScale = "analysisScale";
inline constexpr const char *kParamIterations = "iterations";
inline constexpr const char *kParamSmooth = "smooth";
inline constexpr const char *kParamForwardBackwardCheck = "fbCheck";
inline constexpr const char *kParamForwardBackwardTolerance = "fbTolerance";
inline constexpr const char *kParamFilter = "filter";
inline constexpr const char *kParamEdges = "edges";
inline constexpr const char *kParamDevice = "device";
inline constexpr const char *kParamThreads = "threads";
inline constexpr const char *kParamCacheMegabytes = "cacheMB";
inline constexpr const char *kParamPrecacheRange = "precacheRange";
inline constexpr const char *kParamPrecacheStart = "precacheStart";
inline constexpr const char *kParamPrecacheEnd = "precacheEnd";
inline constexpr const char *kParamPrecache = "precache";
inline constexpr const char *kParamClearCache = "clearCache";
inline constexpr const char *kParamModelDirectory = "modelDir";
inline constexpr const char *kParamStMode = "stMode";
inline constexpr const char *kParamStOrigin = "stOrigin";

enum class DescriptorKind {
  kTrack,
  kStMap,
};

// The values read by an action.  Phase 1 only consumes output, insertTime, stMode and
// stOrigin; the remaining values are still read as their declared types so later inference
// phases have one seam to extend and query actions remain deterministic.
struct FlowParameterValues {
  int model = 0;
  int referenceFrame = 0;
  int output = 0;
  int insertTime = 0;
  int matte = 0;
  int inputCurve = 0;
  int analysisScale = 0;
  int iterations = 0;
  double smooth = 0.0;
  bool forwardBackwardCheck = false;
  double forwardBackwardTolerance = 1.0;
  int filter = 1;
  int edges = 0;
  int device = 0;
  int threads = 0;
  int cacheMegabytes = 256;
  int precacheRange = 0;
  int precacheStart = 0;
  int precacheEnd = 0;
  std::string modelDirectory;
  int stMode = 0;
  int stOrigin = 0;
};

// Defines the common parameters in the approved order.  `kind` controls the two Track-only
// controls and the two ST-only controls; no parameter is hidden with setEnabled().
void defineFlowParameters(OFX::ImageEffectDescriptor &descriptor, DescriptorKind kind);

// Typed parameter access for an instance.  The support library owns the fetched parameter
// objects; this class only keeps non-owning typed pointers to them.
class FlowParameters {
 public:
  FlowParameters(const OFX::ParamSet &paramSet, DescriptorKind kind);

  // The small subset needed by frequent query actions and Phase 1 fallback routing. This
  // avoids reading unrelated controls (especially the Model Dir string) on every identity,
  // ROI, and frame-needs query.
  FlowParameterValues routingValuesAt(double time) const;
  FlowParameterValues valuesAt(double time) const;

  // Used only by the Set Ref push button.  The host supplies OFX time as a frame number for
  // this effect; rounding makes the conversion explicit while avoiding an overflowing cast
  // for a malformed host argument.
  void setReferenceFrame(double time) const;

 private:
  DescriptorKind kind_;

  OFX::ChoiceParam *model_ = nullptr;
  OFX::IntParam *referenceFrame_ = nullptr;
  OFX::ChoiceParam *output_ = nullptr;
  OFX::ChoiceParam *insertTime_ = nullptr;
  OFX::ChoiceParam *matte_ = nullptr;
  OFX::ChoiceParam *inputCurve_ = nullptr;
  OFX::ChoiceParam *analysisScale_ = nullptr;
  OFX::IntParam *iterations_ = nullptr;
  OFX::DoubleParam *smooth_ = nullptr;
  OFX::BooleanParam *forwardBackwardCheck_ = nullptr;
  OFX::DoubleParam *forwardBackwardTolerance_ = nullptr;
  OFX::ChoiceParam *filter_ = nullptr;
  OFX::ChoiceParam *edges_ = nullptr;
  OFX::ChoiceParam *device_ = nullptr;
  OFX::IntParam *threads_ = nullptr;
  OFX::IntParam *cacheMegabytes_ = nullptr;
  OFX::ChoiceParam *precacheRange_ = nullptr;
  OFX::IntParam *precacheStart_ = nullptr;
  OFX::IntParam *precacheEnd_ = nullptr;
  OFX::StringParam *modelDirectory_ = nullptr;
  OFX::ChoiceParam *stMode_ = nullptr;
  OFX::ChoiceParam *stOrigin_ = nullptr;
};

}  // namespace ofx
}  // namespace whitewater

#endif  // WHITEWATER_OFX_FLOWPARAMETERS_H
