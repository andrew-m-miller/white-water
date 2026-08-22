#include "ofx/FlowParameters.h"

#include <cmath>
#include <initializer_list>
#include <limits>

#include "ofxParam.h"

namespace whitewater {
namespace ofx {
namespace {

void label(OFX::ParamDescriptor *parameter, const char *text) {
  parameter->setLabel(text);
}

void defineChoice(OFX::ImageEffectDescriptor &descriptor, const char *name,
                  const char *labelText,
                  std::initializer_list<const char *> options, int defaultValue) {
  OFX::ChoiceParamDescriptor *parameter = descriptor.defineChoiceParam(name);
  label(parameter, labelText);
  for (const char *option : options) {
    parameter->appendOption(option, option);
  }
  parameter->setDefault(defaultValue);
}

template <typename Value, typename Getter>
Value readOr(Getter getter, Value fallback) {
  try {
    return getter();
  } catch (...) {
    return fallback;
  }
}

int choiceAt(OFX::ChoiceParam *parameter, double time, int fallback) {
  if (parameter == nullptr) return fallback;
  return readOr(
      [&] {
        int value = fallback;
        parameter->getValueAtTime(time, value);
        return value;
      },
      fallback);
}

int intAt(OFX::IntParam *parameter, double time, int fallback) {
  return readOr(
      [&] {
        int value = fallback;
        parameter->getValueAtTime(time, value);
        return value;
      },
      fallback);
}

double doubleAt(OFX::DoubleParam *parameter, double time, double fallback) {
  return readOr(
      [&] {
        double value = fallback;
        parameter->getValueAtTime(time, value);
        return value;
      },
      fallback);
}

bool boolAt(OFX::BooleanParam *parameter, double time, bool fallback) {
  return readOr(
      [&] {
        bool value = fallback;
        parameter->getValueAtTime(time, value);
        return value;
      },
      fallback);
}

std::string stringAt(OFX::StringParam *parameter, double time,
                     const std::string &fallback) {
  return readOr(
      [&] {
        std::string value = fallback;
        parameter->getValueAtTime(time, value);
        return value;
      },
      fallback);
}

int timeAsFrame(double time) {
  if (!std::isfinite(time)) return 0;
  const double rounded = std::round(time);
  const double minimum = static_cast<double>(std::numeric_limits<int>::min());
  const double maximum = static_cast<double>(std::numeric_limits<int>::max());
  if (rounded <= minimum) return std::numeric_limits<int>::min();
  if (rounded >= maximum) return std::numeric_limits<int>::max();
  return static_cast<int>(rounded);
}

}  // namespace

void defineFlowParameters(OFX::ImageEffectDescriptor &descriptor, DescriptorKind kind) {
  // Keep this sequence identical to the table in docs/plan.md.  The panel is flat in Flame,
  // so order is the only grouping/layout mechanism available to the plugin. The three
  // bake-off-owned choices are omitted until they have real options: a zero-option numeric
  // Choice has no valid value under OFX, while a placeholder would permanently consume index
  // zero in saved setups.

  OFX::IntParamDescriptor *referenceFrame =
      descriptor.defineIntParam(kParamReferenceFrame);
  label(referenceFrame, "Ref Frame");
  referenceFrame->setDefault(0);
  // Ref Frame is one persistent scalar, not a curve. Set Ref replaces it regardless of
  // timeline position. Use the non-throwing property path so an older host cannot make the
  // descriptor disappear while rejecting an optional property write.
  referenceFrame->getPropertySet().propSetInt(kOfxParamPropAnimates, 0, 0, false);

  OFX::PushButtonParamDescriptor *setReference =
      descriptor.definePushButtonParam(kParamSetReference);
  label(setReference, "Set Ref");

  if (kind == DescriptorKind::kTrack) {
    defineChoice(descriptor, kParamOutput, "Output", {"Composite", "Warped Insert"}, 0);
    defineChoice(descriptor, kParamInsertTime, "Insert At", {"Current", "Reference"}, 0);
  }

  defineChoice(descriptor, kParamMatte, "Matte", {"Premultiply", "Full Frame"}, 0);

  OFX::IntParamDescriptor *iterations = descriptor.defineIntParam(kParamIterations);
  label(iterations, "Iters");
  iterations->setDefault(0);
  iterations->setRange(0, 100);

  OFX::DoubleParamDescriptor *smooth = descriptor.defineDoubleParam(kParamSmooth);
  label(smooth, "Smooth");
  smooth->setDefault(0.0);
  smooth->setRange(0.0, 100.0);

  OFX::BooleanParamDescriptor *forwardBackwardCheck =
      descriptor.defineBooleanParam(kParamForwardBackwardCheck);
  label(forwardBackwardCheck, "FB Check");
  forwardBackwardCheck->setDefault(false);

  OFX::DoubleParamDescriptor *forwardBackwardTolerance =
      descriptor.defineDoubleParam(kParamForwardBackwardTolerance);
  label(forwardBackwardTolerance, "FB Tol");
  forwardBackwardTolerance->setDefault(1.0);
  forwardBackwardTolerance->setRange(0.0, 10000.0);

  defineChoice(descriptor, kParamFilter, "Filter",
               {"Nearest", "Bilinear", "Catmull-Rom"}, 1);
  defineChoice(descriptor, kParamEdges, "Edges", {"Black", "Clamp", "Mirror"}, 0);
  defineChoice(descriptor, kParamDevice, "Device", {"Auto", "GPU", "CPU"}, 0);

  OFX::IntParamDescriptor *threads = descriptor.defineIntParam(kParamThreads);
  label(threads, "Threads");
  threads->setDefault(0);
  threads->setRange(0, 1024);

  OFX::IntParamDescriptor *cacheMegabytes =
      descriptor.defineIntParam(kParamCacheMegabytes);
  label(cacheMegabytes, "Cache MB");
  cacheMegabytes->setDefault(256);
  cacheMegabytes->setRange(0, 1024 * 1024);

  defineChoice(descriptor, kParamPrecacheRange, "Pre Range",
               {"Current-to-Ref", "Work Range", "Custom"}, 0);

  OFX::IntParamDescriptor *precacheStart = descriptor.defineIntParam(kParamPrecacheStart);
  label(precacheStart, "Pre Start");
  precacheStart->setDefault(0);

  OFX::IntParamDescriptor *precacheEnd = descriptor.defineIntParam(kParamPrecacheEnd);
  label(precacheEnd, "Pre End");
  precacheEnd->setDefault(0);

  OFX::PushButtonParamDescriptor *precache = descriptor.definePushButtonParam(kParamPrecache);
  label(precache, "Precache");

  OFX::PushButtonParamDescriptor *clearCache =
      descriptor.definePushButtonParam(kParamClearCache);
  label(clearCache, "Clear");

  OFX::StringParamDescriptor *modelDirectory =
      descriptor.defineStringParam(kParamModelDirectory);
  label(modelDirectory, "Model Dir");
  modelDirectory->setDefault("");
  // Flame accepts the property, but its support library's throwing setter makes this
  // optional mode a load-time failure on older hosts.  Write it directly and non-throwing,
  // as required by the host notes; do not call setStringType().
  modelDirectory->getPropertySet().propSetString(kOfxParamPropStringMode,
                                                  kOfxParamStringIsFilePath, 0, false);

  if (kind == DescriptorKind::kStMap) {
    defineChoice(descriptor, kParamStMode, "ST Mode", {"Absolute UV", "Relative Pixels"}, 0);
    defineChoice(descriptor, kParamStOrigin, "ST Origin", {"Bottom Left", "Top Left"}, 0);
  }
}

FlowParameters::FlowParameters(const OFX::ParamSet &paramSet, DescriptorKind kind)
    : kind_(kind),
      referenceFrame_(paramSet.fetchIntParam(kParamReferenceFrame)),
      matte_(paramSet.fetchChoiceParam(kParamMatte)),
      iterations_(paramSet.fetchIntParam(kParamIterations)),
      smooth_(paramSet.fetchDoubleParam(kParamSmooth)),
      forwardBackwardCheck_(paramSet.fetchBooleanParam(kParamForwardBackwardCheck)),
      forwardBackwardTolerance_(paramSet.fetchDoubleParam(kParamForwardBackwardTolerance)),
      filter_(paramSet.fetchChoiceParam(kParamFilter)),
      edges_(paramSet.fetchChoiceParam(kParamEdges)),
      device_(paramSet.fetchChoiceParam(kParamDevice)),
      threads_(paramSet.fetchIntParam(kParamThreads)),
      cacheMegabytes_(paramSet.fetchIntParam(kParamCacheMegabytes)),
      precacheRange_(paramSet.fetchChoiceParam(kParamPrecacheRange)),
      precacheStart_(paramSet.fetchIntParam(kParamPrecacheStart)),
      precacheEnd_(paramSet.fetchIntParam(kParamPrecacheEnd)),
      modelDirectory_(paramSet.fetchStringParam(kParamModelDirectory)) {
  if (kind_ == DescriptorKind::kTrack) {
    output_ = paramSet.fetchChoiceParam(kParamOutput);
    insertTime_ = paramSet.fetchChoiceParam(kParamInsertTime);
  }
  if (kind_ == DescriptorKind::kStMap) {
    stMode_ = paramSet.fetchChoiceParam(kParamStMode);
    stOrigin_ = paramSet.fetchChoiceParam(kParamStOrigin);
  }
}

FlowParameterValues FlowParameters::routingValuesAt(double time) const {
  FlowParameterValues values;
  values.referenceFrame = intAt(referenceFrame_, time, values.referenceFrame);
  if (kind_ == DescriptorKind::kTrack) {
    values.output = choiceAt(output_, time, values.output);
    values.insertTime = choiceAt(insertTime_, time, values.insertTime);
  }
  if (kind_ == DescriptorKind::kStMap) {
    values.stMode = choiceAt(stMode_, time, values.stMode);
    values.stOrigin = choiceAt(stOrigin_, time, values.stOrigin);
  }
  return values;
}

FlowParameterValues FlowParameters::valuesAt(double time) const {
  FlowParameterValues values = routingValuesAt(time);
  values.model = choiceAt(model_, time, values.model);
  values.matte = choiceAt(matte_, time, values.matte);
  values.inputCurve = choiceAt(inputCurve_, time, values.inputCurve);
  values.analysisScale = choiceAt(analysisScale_, time, values.analysisScale);
  values.iterations = intAt(iterations_, time, values.iterations);
  values.smooth = doubleAt(smooth_, time, values.smooth);
  values.forwardBackwardCheck =
      boolAt(forwardBackwardCheck_, time, values.forwardBackwardCheck);
  values.forwardBackwardTolerance =
      doubleAt(forwardBackwardTolerance_, time, values.forwardBackwardTolerance);
  values.filter = choiceAt(filter_, time, values.filter);
  values.edges = choiceAt(edges_, time, values.edges);
  values.device = choiceAt(device_, time, values.device);
  values.threads = intAt(threads_, time, values.threads);
  values.cacheMegabytes = intAt(cacheMegabytes_, time, values.cacheMegabytes);
  values.precacheRange = choiceAt(precacheRange_, time, values.precacheRange);
  values.precacheStart = intAt(precacheStart_, time, values.precacheStart);
  values.precacheEnd = intAt(precacheEnd_, time, values.precacheEnd);
  values.modelDirectory = stringAt(modelDirectory_, time, values.modelDirectory);
  return values;
}

void FlowParameters::setReferenceFrame(double time) const {
  if (referenceFrame_ == nullptr) return;
  referenceFrame_->setValue(timeAsFrame(time));
}

}  // namespace ofx
}  // namespace whitewater
