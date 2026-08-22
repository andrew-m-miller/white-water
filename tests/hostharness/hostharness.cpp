// Adapted from warp-drive 887a123, tests/hostharness/hostharness.cpp.
// White Water Phase 1 host contract harness.
//
// This is intentionally a small, dependency-free OFX host.  It uses the raw OFX C headers
// rather than the C++ support library so a passing result says something about the plugin's
// actual host boundary.  The harness loads a bundle, enumerates both permanent descriptors,
// drives describe/create/render/destroy, and serves generated Source/Insert images at the
// exact times requested by clipGetImage.  Every generated frame carries its clip and time in
// its pixels; a temporal contract therefore cannot accidentally pass by returning the wrong
// image and a query action cannot quietly pull a frame without being counted.

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

#include "ofxCore.h"
#include "ofxGPURender.h"
#include "ofxImageEffect.h"
#include "ofxInteract.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParam.h"
#include "ofxParametricParam.h"
#include "ofxProgress.h"
#include "ofxProperty.h"
#include "ofxTimeLine.h"

namespace {

int gFailures = 0;

void fail(const std::string &message) {
  std::fprintf(stderr, "FAIL: %s\n", message.c_str());
  ++gFailures;
}

void info(const std::string &message) { std::fprintf(stdout, "  %s\n", message.c_str()); }

void check(bool condition, const std::string &message) {
  if (condition)
    info(message);
  else
    fail(message);
}

// ---------------------------------------------------------------------------
// Raw host property sets
// ---------------------------------------------------------------------------

struct Property {
  enum class Kind { kNone, kString, kInt, kDouble, kPointer };
  Kind kind = Kind::kNone;
  std::vector<std::string> strings;
  std::vector<int> ints;
  std::vector<double> doubles;
  std::vector<void *> pointers;

  int dimension() const {
    switch (kind) {
      case Kind::kString: return static_cast<int>(strings.size());
      case Kind::kInt: return static_cast<int>(ints.size());
      case Kind::kDouble: return static_cast<int>(doubles.size());
      case Kind::kPointer: return static_cast<int>(pointers.size());
      case Kind::kNone: return 0;
    }
    return 0;
  }

  void select(Kind desired) {
    if (kind == desired) return;
    kind = desired;
    strings.clear();
    ints.clear();
    doubles.clear();
    pointers.clear();
  }
};

struct PropertySet {
  std::map<std::string, Property> properties;
};

PropertySet *asPropertySet(OfxPropertySetHandle handle) {
  return reinterpret_cast<PropertySet *>(handle);
}

template <typename T>
void assignAt(std::vector<T> &values, int index, const T &value) {
  if (index < 0) return;
  if (static_cast<int>(values.size()) <= index) values.resize(static_cast<std::size_t>(index + 1));
  values[static_cast<std::size_t>(index)] = value;
}

bool refusedEnabledProperty(const char *name) {
  // Flame's parameter property implementation does not make kOfxParamPropEnabled a safe
  // capability probe.  The production plugin must not call setEnabled(); returning the
  // unsupported status here turns that forbidden assumption into a visible harness failure.
  return name != nullptr && std::strcmp(name, kOfxParamPropEnabled) == 0;
}

bool gRefuseStringModes = false;

OfxStatus setPointer(OfxPropertySetHandle handle, const char *name, int index, void *value) {
  if (refusedEnabledProperty(name)) return kOfxStatErrUnsupported;
  Property &property = asPropertySet(handle)->properties[name];
  property.select(Property::Kind::kPointer);
  assignAt(property.pointers, index, value);
  return kOfxStatOK;
}

OfxStatus setString(OfxPropertySetHandle handle, const char *name, int index, const char *value) {
  if (refusedEnabledProperty(name)) return kOfxStatErrUnsupported;
  if (gRefuseStringModes && name != nullptr &&
      std::strcmp(name, kOfxParamPropStringMode) == 0 &&
      (value == nullptr || std::strcmp(value, kOfxParamStringIsSingleLine) != 0)) {
    return kOfxStatErrValue;
  }
  Property &property = asPropertySet(handle)->properties[name];
  property.select(Property::Kind::kString);
  assignAt(property.strings, index, std::string(value == nullptr ? "" : value));
  return kOfxStatOK;
}

OfxStatus setDouble(OfxPropertySetHandle handle, const char *name, int index, double value) {
  if (refusedEnabledProperty(name)) return kOfxStatErrUnsupported;
  Property &property = asPropertySet(handle)->properties[name];
  property.select(Property::Kind::kDouble);
  assignAt(property.doubles, index, value);
  return kOfxStatOK;
}

OfxStatus setInt(OfxPropertySetHandle handle, const char *name, int index, int value) {
  if (refusedEnabledProperty(name)) return kOfxStatErrUnsupported;
  Property &property = asPropertySet(handle)->properties[name];
  property.select(Property::Kind::kInt);
  assignAt(property.ints, index, value);
  return kOfxStatOK;
}

OfxStatus setPointerN(OfxPropertySetHandle handle, const char *name, int count,
                     void *const *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = setPointer(handle, name, i, values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus setStringN(OfxPropertySetHandle handle, const char *name, int count,
                    const char *const *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = setString(handle, name, i, values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus setDoubleN(OfxPropertySetHandle handle, const char *name, int count,
                    const double *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = setDouble(handle, name, i, values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus setIntN(OfxPropertySetHandle handle, const char *name, int count, const int *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = setInt(handle, name, i, values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

const Property *findProperty(OfxPropertySetHandle handle, const char *name) {
  if (handle == nullptr || name == nullptr) return nullptr;
  const auto &properties = asPropertySet(handle)->properties;
  const auto it = properties.find(name);
  return it == properties.end() ? nullptr : &it->second;
}

OfxStatus getPointer(OfxPropertySetHandle handle, const char *name, int index, void **value) {
  const Property *property = findProperty(handle, name);
  if (property == nullptr || property->kind != Property::Kind::kPointer)
    return kOfxStatErrUnknown;
  if (index < 0 || index >= static_cast<int>(property->pointers.size())) return kOfxStatErrValue;
  *value = property->pointers[static_cast<std::size_t>(index)];
  return kOfxStatOK;
}

OfxStatus getString(OfxPropertySetHandle handle, const char *name, int index, char **value) {
  const Property *property = findProperty(handle, name);
  if (property == nullptr || property->kind != Property::Kind::kString)
    return kOfxStatErrUnknown;
  if (index < 0 || index >= static_cast<int>(property->strings.size())) return kOfxStatErrValue;
  *value = const_cast<char *>(property->strings[static_cast<std::size_t>(index)].c_str());
  return kOfxStatOK;
}

OfxStatus getDouble(OfxPropertySetHandle handle, const char *name, int index, double *value) {
  const Property *property = findProperty(handle, name);
  if (property == nullptr || property->kind != Property::Kind::kDouble)
    return kOfxStatErrUnknown;
  if (index < 0 || index >= static_cast<int>(property->doubles.size())) return kOfxStatErrValue;
  *value = property->doubles[static_cast<std::size_t>(index)];
  return kOfxStatOK;
}

OfxStatus getInt(OfxPropertySetHandle handle, const char *name, int index, int *value) {
  const Property *property = findProperty(handle, name);
  if (property == nullptr || property->kind != Property::Kind::kInt)
    return kOfxStatErrUnknown;
  if (index < 0 || index >= static_cast<int>(property->ints.size())) return kOfxStatErrValue;
  *value = property->ints[static_cast<std::size_t>(index)];
  return kOfxStatOK;
}

OfxStatus getPointerN(OfxPropertySetHandle handle, const char *name, int count, void **values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = getPointer(handle, name, i, &values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus getStringN(OfxPropertySetHandle handle, const char *name, int count, char **values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = getString(handle, name, i, &values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus getDoubleN(OfxPropertySetHandle handle, const char *name, int count, double *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = getDouble(handle, name, i, &values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus getIntN(OfxPropertySetHandle handle, const char *name, int count, int *values) {
  for (int i = 0; i < count; ++i) {
    const OfxStatus status = getInt(handle, name, i, &values[i]);
    if (status != kOfxStatOK) return status;
  }
  return kOfxStatOK;
}

OfxStatus resetProperty(OfxPropertySetHandle handle, const char *name) {
  if (handle == nullptr || name == nullptr) return kOfxStatErrBadHandle;
  asPropertySet(handle)->properties.erase(name);
  return kOfxStatOK;
}

OfxStatus getDimension(OfxPropertySetHandle handle, const char *name, int *dimension) {
  if (dimension == nullptr) return kOfxStatErrBadHandle;
  const Property *property = findProperty(handle, name);
  *dimension = property == nullptr ? 0 : property->dimension();
  return kOfxStatOK;
}

OfxPropertySuiteV1 gPropertySuite = {
    setPointer, setString, setDouble, setInt, setPointerN, setStringN, setDoubleN, setIntN,
    getPointer, getString, getDouble, getInt, getPointerN, getStringN, getDoubleN, getIntN,
    resetProperty, getDimension};

// Convenience writers/readers used by the host itself.  These bypass deliberate host
// refusals because they model properties the host would have created before the plugin ran.
void hostSetString(PropertySet *set, const char *name, const std::string &value) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kString);
  property.strings = {value};
}

void hostSetString(PropertySet *set, const std::string &name, const std::string &value) {
  hostSetString(set, name.c_str(), value);
}

void hostSetStrings(PropertySet *set, const char *name, std::vector<std::string> values) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kString);
  property.strings = std::move(values);
}

void hostSetInt(PropertySet *set, const char *name, int value) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kInt);
  property.ints = {value};
}

void hostSetInts(PropertySet *set, const char *name, std::vector<int> values) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kInt);
  property.ints = std::move(values);
}

void hostSetDouble(PropertySet *set, const char *name, double value) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kDouble);
  property.doubles = {value};
}

void hostSetDouble(PropertySet *set, const std::string &name, double value) {
  hostSetDouble(set, name.c_str(), value);
}

void hostSetDoubles(PropertySet *set, const char *name, std::vector<double> values) {
  Property &property = set->properties[name];
  property.select(Property::Kind::kDouble);
  property.doubles = std::move(values);
}

void hostSetDoubles(PropertySet *set, const std::string &name, std::vector<double> values) {
  hostSetDoubles(set, name.c_str(), std::move(values));
}

bool readString(OfxPropertySetHandle handle, const std::string &name, std::string *value,
                int index = 0) {
  char *raw = nullptr;
  if (getString(handle, name.c_str(), index, &raw) != kOfxStatOK || raw == nullptr) return false;
  *value = raw;
  return true;
}

bool readInt(OfxPropertySetHandle handle, const std::string &name, int *value, int index = 0) {
  return getInt(handle, name.c_str(), index, value) == kOfxStatOK;
}

bool readDouble(OfxPropertySetHandle handle, const std::string &name, double *value,
               int index = 0) {
  return getDouble(handle, name.c_str(), index, value) == kOfxStatOK;
}

std::vector<std::string> stringsFrom(OfxPropertySetHandle handle, const std::string &name) {
  std::vector<std::string> result;
  const Property *property = findProperty(handle, name.c_str());
  if (property == nullptr || property->kind != Property::Kind::kString) return result;
  return property->strings;
}

std::vector<double> doublesFrom(OfxPropertySetHandle handle, const std::string &name) {
  const Property *property = findProperty(handle, name.c_str());
  if (property == nullptr || property->kind != Property::Kind::kDouble) return {};
  return property->doubles;
}

// ---------------------------------------------------------------------------
// Raw host parameter suite
// ---------------------------------------------------------------------------

struct Parameter {
  std::string name;
  std::string type;
  PropertySet props;
  std::string stringValue;
  int intValue = 0;
  double doubleValue = 0.0;
  std::vector<double> doubleValues;
  unsigned int valueSetCount = 0;
  unsigned int valueSetAtTimeCount = 0;
};

struct ParamSet {
  std::vector<std::unique_ptr<Parameter>> params;

  Parameter *find(const std::string &name) const {
    for (const auto &param : params)
      if (param->name == name) return param.get();
    return nullptr;
  }
};

ParamSet *asParamSet(OfxParamSetHandle handle) {
  return reinterpret_cast<ParamSet *>(handle);
}

Parameter *asParameter(OfxParamHandle handle) {
  return reinterpret_cast<Parameter *>(handle);
}

void applyDefault(Parameter *parameter) {
  const Property *property = findProperty(
      reinterpret_cast<OfxPropertySetHandle>(&parameter->props), kOfxParamPropDefault);
  if (property == nullptr) return;
  if (property->kind == Property::Kind::kString && !property->strings.empty())
    parameter->stringValue = property->strings.front();
  if (property->kind == Property::Kind::kInt && !property->ints.empty())
    parameter->intValue = property->ints.front();
  if (property->kind == Property::Kind::kDouble && !property->doubles.empty()) {
    parameter->doubleValue = property->doubles.front();
    parameter->doubleValues = property->doubles;
  }
}

OfxStatus defineParameter(OfxParamSetHandle paramSet, const char *paramType, const char *name,
                         OfxPropertySetHandle *propertySet) {
  // This is an intentional host capability boundary.  White Water Phase 1 does not use
  // custom or parametric parameters; refusing them catches a plugin that assumes a feature
  // Flame does not provide rather than allowing the descriptor to disappear silently.
  if (paramType == nullptr || name == nullptr) return kOfxStatErrBadHandle;
  if (std::strcmp(paramType, kOfxParamTypeCustom) == 0 ||
      std::strcmp(paramType, kOfxParamTypeParametric) == 0)
    return kOfxStatErrUnsupported;

  ParamSet *set = asParamSet(paramSet);
  auto parameter = std::make_unique<Parameter>();
  parameter->name = name;
  parameter->type = paramType;
  hostSetString(&parameter->props, kOfxPropName, name);
  hostSetString(&parameter->props, kOfxParamPropType, paramType);
  // A real host materialises the standard descriptor properties before the plugin writes
  // them.  In particular, OFX's C++ support library reads an empty Hint while appending the
  // first Choice option; treating an as-yet-unwritten property as unknown would abort
  // describeInContext halfway through the parameter list and make the plugin appear to have
  // only the first few controls.
  hostSetString(&parameter->props, kOfxParamPropHint, "");
  hostSetString(&parameter->props, kOfxPropLabel, "");
  hostSetString(&parameter->props, kOfxPropShortLabel, "");
  hostSetString(&parameter->props, kOfxPropLongLabel, "");
  if (propertySet != nullptr)
    *propertySet = reinterpret_cast<OfxPropertySetHandle>(&parameter->props);
  set->params.push_back(std::move(parameter));
  return kOfxStatOK;
}

OfxStatus getParameterHandle(OfxParamSetHandle paramSet, const char *name, OfxParamHandle *param,
                             OfxPropertySetHandle *propertySet) {
  if (paramSet == nullptr || name == nullptr || param == nullptr) return kOfxStatErrBadHandle;
  Parameter *found = asParamSet(paramSet)->find(name);
  if (found == nullptr) return kOfxStatErrUnknown;
  *param = reinterpret_cast<OfxParamHandle>(found);
  if (propertySet != nullptr)
    *propertySet = reinterpret_cast<OfxPropertySetHandle>(&found->props);
  return kOfxStatOK;
}

OfxStatus getParamSetProperty(OfxParamSetHandle, OfxPropertySetHandle *propertySet) {
  // A real host creates the parameter-set properties even though Phase 1 does not consume
  // any of them.  Returning a stable empty set is enough for support code that asks.
  static PropertySet empty;
  if (propertySet == nullptr) return kOfxStatErrBadHandle;
  *propertySet = reinterpret_cast<OfxPropertySetHandle>(&empty);
  return kOfxStatOK;
}

OfxStatus getParameterProperty(OfxParamHandle param, OfxPropertySetHandle *propertySet) {
  if (param == nullptr || propertySet == nullptr) return kOfxStatErrBadHandle;
  *propertySet = reinterpret_cast<OfxPropertySetHandle>(&asParameter(param)->props);
  return kOfxStatOK;
}

OfxStatus readParameterValue(Parameter *parameter, va_list args) {
  if (parameter->type == kOfxParamTypeString) {
    char **value = va_arg(args, char **);
    *value = const_cast<char *>(parameter->stringValue.c_str());
    return kOfxStatOK;
  }
  if (parameter->type == kOfxParamTypeDouble || parameter->type == kOfxParamTypeDouble2D ||
      parameter->type == kOfxParamTypeDouble3D) {
    double *value = va_arg(args, double *);
    *value = parameter->doubleValue;
    if (parameter->type == kOfxParamTypeDouble2D || parameter->type == kOfxParamTypeDouble3D) {
      const int dimensions = parameter->type == kOfxParamTypeDouble2D ? 2 : 3;
      for (int i = 1; i < dimensions; ++i) value[i] = i < static_cast<int>(parameter->doubleValues.size())
                                                            ? parameter->doubleValues[i]
                                                            : 0.0;
    }
    return kOfxStatOK;
  }
  if (parameter->type == kOfxParamTypeInteger || parameter->type == kOfxParamTypeBoolean ||
      parameter->type == kOfxParamTypeChoice) {
    *va_arg(args, int *) = parameter->intValue;
    return kOfxStatOK;
  }
  return kOfxStatErrUnsupported;
}

OfxStatus getParameterValue(OfxParamHandle param, ...) {
  if (param == nullptr) return kOfxStatErrBadHandle;
  va_list args;
  va_start(args, param);
  const OfxStatus status = readParameterValue(asParameter(param), args);
  va_end(args);
  return status;
}

OfxStatus getParameterValueAtTime(OfxParamHandle param, OfxTime time, ...) {
  if (param == nullptr) return kOfxStatErrBadHandle;
  (void)time;
  va_list args;
  va_start(args, time);
  const OfxStatus status = readParameterValue(asParameter(param), args);
  va_end(args);
  return status;
}

OfxStatus setParameterValue(OfxParamHandle param, ...) {
  if (param == nullptr) return kOfxStatErrBadHandle;
  Parameter *parameter = asParameter(param);
  ++parameter->valueSetCount;
  va_list args;
  va_start(args, param);
  OfxStatus status = kOfxStatOK;
  if (parameter->type == kOfxParamTypeString) {
    const char *value = va_arg(args, const char *);
    parameter->stringValue = value == nullptr ? "" : value;
  } else if (parameter->type == kOfxParamTypeDouble ||
             parameter->type == kOfxParamTypeDouble2D ||
             parameter->type == kOfxParamTypeDouble3D) {
    parameter->doubleValue = va_arg(args, double);
    parameter->doubleValues = {parameter->doubleValue};
    const int dimensions = parameter->type == kOfxParamTypeDouble ? 1
                           : parameter->type == kOfxParamTypeDouble2D ? 2
                                                                        : 3;
    for (int i = 1; i < dimensions; ++i) parameter->doubleValues.push_back(va_arg(args, double));
  } else if (parameter->type == kOfxParamTypeInteger ||
             parameter->type == kOfxParamTypeBoolean || parameter->type == kOfxParamTypeChoice) {
    parameter->intValue = va_arg(args, int);
  } else if (parameter->type != kOfxParamTypePushButton) {
    status = kOfxStatErrUnsupported;
  }
  va_end(args);
  return status;
}

OfxStatus setParameterValueAtTime(OfxParamHandle param, OfxTime time, ...) {
  if (param == nullptr) return kOfxStatErrBadHandle;
  Parameter *parameter = asParameter(param);
  ++parameter->valueSetAtTimeCount;
  va_list args;
  va_start(args, time);
  OfxStatus status = kOfxStatOK;
  if (parameter->type == kOfxParamTypeString) {
    const char *value = va_arg(args, const char *);
    parameter->stringValue = value == nullptr ? "" : value;
  } else if (parameter->type == kOfxParamTypeDouble) {
    parameter->doubleValue = va_arg(args, double);
    parameter->doubleValues = {parameter->doubleValue};
  } else if (parameter->type == kOfxParamTypeInteger ||
             parameter->type == kOfxParamTypeBoolean ||
             parameter->type == kOfxParamTypeChoice) {
    parameter->intValue = va_arg(args, int);
  } else {
    status = kOfxStatErrUnsupported;
  }
  va_end(args);
  return status;
}

OfxStatus getParameterDerivative(OfxParamHandle, OfxTime, ...) { return kOfxStatErrUnsupported; }
OfxStatus getParameterIntegral(OfxParamHandle, OfxTime, OfxTime, ...) {
  return kOfxStatErrUnsupported;
}
OfxStatus getNumKeys(OfxParamHandle, unsigned int *count) {
  if (count == nullptr) return kOfxStatErrBadHandle;
  *count = 0;
  return kOfxStatOK;
}
OfxStatus getKeyTime(OfxParamHandle, unsigned int, OfxTime *) { return kOfxStatErrBadIndex; }
OfxStatus getKeyIndex(OfxParamHandle, OfxTime, int, int *) { return kOfxStatErrBadIndex; }
OfxStatus deleteKey(OfxParamHandle, OfxTime) { return kOfxStatOK; }
OfxStatus deleteAllKeys(OfxParamHandle) { return kOfxStatOK; }
OfxStatus copyParameter(OfxParamHandle, OfxParamHandle, OfxTime, const OfxRangeD *) {
  return kOfxStatOK;
}
OfxStatus editBegin(OfxParamSetHandle, const char *) { return kOfxStatOK; }
OfxStatus editEnd(OfxParamSetHandle) { return kOfxStatOK; }

OfxParameterSuiteV1 gParameterSuite = {
    defineParameter,
    getParameterHandle,
    getParamSetProperty,
    getParameterProperty,
    getParameterValue,
    getParameterValueAtTime,
    getParameterDerivative,
    getParameterIntegral,
    setParameterValue,
    setParameterValueAtTime,
    getNumKeys,
    getKeyTime,
    getKeyIndex,
    deleteKey,
    deleteAllKeys,
    copyParameter,
    editBegin,
    editEnd};

// ---------------------------------------------------------------------------
// Image storage and effect handles
// ---------------------------------------------------------------------------

struct Plane {
  std::vector<unsigned char> bytes;
  OfxRectI bounds = {0, 0, 0, 0};
  int rowBytes = 0;
  int pixelBytes = 0;
  std::string depth;
  std::string components;

  int width() const { return bounds.x2 - bounds.x1; }
  int height() const { return bounds.y2 - bounds.y1; }

  unsigned char *pixel(int x, int y) {
    return bytes.data() + static_cast<std::size_t>(y - bounds.y1) * rowBytes +
           static_cast<std::size_t>(x - bounds.x1) * pixelBytes;
  }
  const unsigned char *pixel(int x, int y) const {
    return bytes.data() + static_cast<std::size_t>(y - bounds.y1) * rowBytes +
           static_cast<std::size_t>(x - bounds.x1) * pixelBytes;
  }
};

int componentCount(const std::string &components) {
  if (components == kOfxImageComponentRGBA) return 4;
  if (components == kOfxImageComponentRGB) return 3;
  if (components == kOfxImageComponentAlpha) return 1;
  return 0;
}

int depthBytes(const std::string &depth) {
  if (depth == kOfxBitDepthByte) return 1;
  if (depth == kOfxBitDepthShort || depth == kOfxBitDepthHalf) return 2;
  if (depth == kOfxBitDepthFloat) return 4;
  return 0;
}

void allocatePlane(Plane *plane, int width, int height, const std::string &depth,
                   const std::string &components, int originX = 0, int originY = 0,
                   int rowPadding = 0) {
  plane->bounds = {originX, originY, originX + width, originY + height};
  plane->depth = depth;
  plane->components = components;
  plane->pixelBytes = depthBytes(depth) * componentCount(components);
  plane->rowBytes = plane->pixelBytes * width + rowPadding;
  plane->bytes.assign(static_cast<std::size_t>(plane->rowBytes) * height, 0);
}

// A deterministic per-clip/per-time frame.  The values are intentionally not a smooth image:
// every temporal pull and channel remains distinguishable after a plugin copies, composites,
// or converts it.  The same function creates the expected result, so the harness does not
// make assumptions about a model's inference output.
void fillSentinel(Plane *plane, int clipTag, double time) {
  const int channels = componentCount(plane->components);
  const int timeTag = static_cast<int>(std::llround(time));
  for (int y = plane->bounds.y1; y < plane->bounds.y2; ++y) {
    for (int x = plane->bounds.x1; x < plane->bounds.x2; ++x) {
      unsigned char *pixel = plane->pixel(x, y);
      for (int c = 0; c < channels; ++c) {
        const std::uint32_t mix = static_cast<std::uint32_t>(
            clipTag * 0x1f123u + (timeTag + 37) * 0x2b1u + (x + 4096) * 37u +
            (y + 4096) * 101u + static_cast<unsigned int>(c * 7919));
        if (plane->depth == kOfxBitDepthByte) {
          pixel[c] = static_cast<unsigned char>((mix % 251u) + 1u);
        } else if (plane->depth == kOfxBitDepthShort) {
          const std::uint16_t value = static_cast<std::uint16_t>((mix * 251u) & 0xfff0u);
          std::memcpy(pixel + c * 2, &value, sizeof(value));
        } else if (plane->depth == kOfxBitDepthHalf) {
          // Finite, positive half values.  Keeping the exponent fixed makes expected
          // source/insert sentinels stable through a well-behaved half conversion path.
          const std::uint16_t value = static_cast<std::uint16_t>(0x3000u | (mix & 0x03f0u));
          std::memcpy(pixel + c * 2, &value, sizeof(value));
        } else if (plane->depth == kOfxBitDepthFloat) {
          const float value = static_cast<float>((mix % 100000u) + 1u) / 100001.0f;
          std::memcpy(pixel + c * 4, &value, sizeof(value));
        }
      }
    }
  }
}

bool planesEqual(const Plane &a, const Plane &b) {
  return a.bounds.x1 == b.bounds.x1 && a.bounds.y1 == b.bounds.y1 &&
         a.bounds.x2 == b.bounds.x2 && a.bounds.y2 == b.bounds.y2 &&
         a.depth == b.depth && a.components == b.components && a.rowBytes == b.rowBytes &&
         a.bytes == b.bytes;
}

enum class ClipRole { kSource, kInsert, kOutput, kOther };

struct ImageAllocation {
  PropertySet props;
  Plane plane;
};

struct Clip {
  std::string name;
  ClipRole role = ClipRole::kOther;
  PropertySet props;
  bool connected = false;
  std::string depth = kOfxBitDepthFloat;
  std::string components = kOfxImageComponentRGBA;
  double pixelAspectRatio = 1.0;
};

struct Effect {
  PropertySet props;
  ParamSet params;
  std::vector<std::unique_ptr<Clip>> clips;
  std::vector<std::unique_ptr<ImageAllocation>> liveImages;
  std::unique_ptr<ImageAllocation> outputImage;

  Clip *find(const std::string &name) const {
    for (const auto &clip : clips)
      if (clip->name == name) return clip.get();
    return nullptr;
  }
};

struct Pull {
  std::string clip;
  double time = 0.0;
  bool hadRegion = false;
};

struct RenderState {
  int width = 12;
  int height = 8;
  std::string depth = kOfxBitDepthFloat;
  std::string components = kOfxImageComponentRGBA;
  double pixelAspectRatio = 1.0;
  double renderTime = 0.0;
  OfxRectI renderWindow = {0, 0, 12, 8};
  int imageOriginX = 0;
  int imageOriginY = 0;
  int rowPadding = 0;
  std::vector<Pull> pulls;
  std::vector<Pull> rodQueries;
};

Effect *gActiveEffect = nullptr;
RenderState *gRenderState = nullptr;

OfxStatus effectGetPropertySet(OfxImageEffectHandle effect, OfxPropertySetHandle *propertySet) {
  if (effect == nullptr || propertySet == nullptr) return kOfxStatErrBadHandle;
  *propertySet = reinterpret_cast<OfxPropertySetHandle>(&reinterpret_cast<Effect *>(effect)->props);
  return kOfxStatOK;
}

OfxStatus effectGetParamSet(OfxImageEffectHandle effect, OfxParamSetHandle *paramSet) {
  if (effect == nullptr || paramSet == nullptr) return kOfxStatErrBadHandle;
  *paramSet = reinterpret_cast<OfxParamSetHandle>(&reinterpret_cast<Effect *>(effect)->params);
  return kOfxStatOK;
}

ClipRole roleForName(const char *name) {
  if (name == nullptr) return ClipRole::kOther;
  if (std::strcmp(name, kOfxImageEffectSimpleSourceClipName) == 0 ||
      std::strcmp(name, "Source") == 0)
    return ClipRole::kSource;
  if (std::strcmp(name, "Insert") == 0) return ClipRole::kInsert;
  if (std::strcmp(name, kOfxImageEffectOutputClipName) == 0) return ClipRole::kOutput;
  return ClipRole::kOther;
}

OfxStatus effectClipDefine(OfxImageEffectHandle effect, const char *name,
                           OfxPropertySetHandle *propertySet) {
  if (effect == nullptr || name == nullptr) return kOfxStatErrBadHandle;
  Effect *hostEffect = reinterpret_cast<Effect *>(effect);
  auto clip = std::make_unique<Clip>();
  clip->name = name;
  clip->role = roleForName(name);
  hostSetString(&clip->props, kOfxPropName, name);
  // Optional and Connected are host-owned instance/descriptor properties.  Their standard
  // defaults exist before a plugin calls setOptional; initializing the descriptor default is
  // what lets a required Source/Output clip be distinguished from an omitted property.
  hostSetInt(&clip->props, kOfxImageClipPropOptional, 0);
  hostSetInt(&clip->props, kOfxImageClipPropConnected, 0);
  if (propertySet != nullptr)
    *propertySet = reinterpret_cast<OfxPropertySetHandle>(&clip->props);
  hostEffect->clips.push_back(std::move(clip));
  return kOfxStatOK;
}

OfxStatus effectClipGetHandle(OfxImageEffectHandle effect, const char *name,
                              OfxImageClipHandle *clip, OfxPropertySetHandle *propertySet) {
  if (effect == nullptr || name == nullptr || clip == nullptr) return kOfxStatErrBadHandle;
  Clip *found = reinterpret_cast<Effect *>(effect)->find(name);
  if (found == nullptr) return kOfxStatErrUnknown;
  *clip = reinterpret_cast<OfxImageClipHandle>(found);
  if (propertySet != nullptr)
    *propertySet = reinterpret_cast<OfxPropertySetHandle>(&found->props);
  return kOfxStatOK;
}

OfxStatus effectClipGetPropertySet(OfxImageClipHandle clip, OfxPropertySetHandle *propertySet) {
  if (clip == nullptr || propertySet == nullptr) return kOfxStatErrBadHandle;
  *propertySet = reinterpret_cast<OfxPropertySetHandle>(&reinterpret_cast<Clip *>(clip)->props);
  return kOfxStatOK;
}

PropertySet *imageProps(ImageAllocation *allocation) { return &allocation->props; }

void fillImageProperties(ImageAllocation *allocation, Clip *clip) {
  Plane &plane = allocation->plane;
  PropertySet *props = imageProps(allocation);
  hostSetString(props, kOfxPropType, kOfxTypeImage);
  props->properties[kOfxImagePropData].select(Property::Kind::kPointer);
  props->properties[kOfxImagePropData].pointers = {plane.bytes.data()};
  hostSetInts(props, kOfxImagePropBounds,
              {plane.bounds.x1, plane.bounds.y1, plane.bounds.x2, plane.bounds.y2});
  hostSetInts(props, kOfxImagePropRegionOfDefinition,
              {plane.bounds.x1, plane.bounds.y1, plane.bounds.x2, plane.bounds.y2});
  hostSetInt(props, kOfxImagePropRowBytes, plane.rowBytes);
  hostSetDouble(props, kOfxImagePropPixelAspectRatio, clip->pixelAspectRatio);
  hostSetString(props, kOfxImageEffectPropComponents, plane.components);
  hostSetString(props, kOfxImageEffectPropPixelDepth, plane.depth);
  hostSetString(props, kOfxImageEffectPropPreMultiplication, kOfxImageUnPreMultiplied);
  hostSetDoubles(props, kOfxImageEffectPropRenderScale, {1.0, 1.0});
  hostSetString(props, kOfxImagePropField, kOfxImageFieldNone);
  hostSetString(props, kOfxImagePropUniqueIdentifier,
                clip->name + "@" + std::to_string(gRenderState == nullptr ? 0.0
                                                                         : gRenderState->renderTime));
}

OfxStatus effectClipGetImage(OfxImageClipHandle clipHandle, OfxTime time, const OfxRectD *region,
                             OfxPropertySetHandle *image) {
  if (image != nullptr) *image = nullptr;
  if (clipHandle == nullptr || gActiveEffect == nullptr || gRenderState == nullptr)
    return kOfxStatFailed;
  Clip *clip = reinterpret_cast<Clip *>(clipHandle);
  if (gRenderState != nullptr)
    gRenderState->pulls.push_back({clip->name, time, region != nullptr});

  if (clip->role == ClipRole::kOutput) {
    if (gActiveEffect->outputImage == nullptr) return kOfxStatFailed;
    if (image != nullptr)
      *image = reinterpret_cast<OfxPropertySetHandle>(&gActiveEffect->outputImage->props);
    return kOfxStatOK;
  }
  if (!clip->connected) return kOfxStatFailed;

  auto allocation = std::make_unique<ImageAllocation>();
  const int originX = gRenderState->imageOriginX;
  const int originY = gRenderState->imageOriginY;
  allocatePlane(&allocation->plane, gRenderState->width, gRenderState->height, clip->depth,
                clip->components, originX, originY, gRenderState->rowPadding);
  fillSentinel(&allocation->plane, clip->role == ClipRole::kSource ? 17 : 83, time);
  fillImageProperties(allocation.get(), clip);
  if (image != nullptr)
    *image = reinterpret_cast<OfxPropertySetHandle>(&allocation->props);
  gActiveEffect->liveImages.push_back(std::move(allocation));
  return kOfxStatOK;
}

OfxStatus effectClipReleaseImage(OfxPropertySetHandle image) {
  if (gActiveEffect == nullptr || image == nullptr) return kOfxStatOK;
  for (auto it = gActiveEffect->liveImages.begin(); it != gActiveEffect->liveImages.end(); ++it) {
    if (reinterpret_cast<OfxPropertySetHandle>(&(*it)->props) == image) {
      gActiveEffect->liveImages.erase(it);
      return kOfxStatOK;
    }
  }
  if (gActiveEffect->outputImage != nullptr &&
      reinterpret_cast<OfxPropertySetHandle>(&gActiveEffect->outputImage->props) == image)
    return kOfxStatOK;
  return kOfxStatErrBadHandle;
}

OfxStatus effectClipGetRoD(OfxImageClipHandle clipHandle, OfxTime time, OfxRectD *bounds) {
  if (clipHandle == nullptr || bounds == nullptr || gRenderState == nullptr) return kOfxStatFailed;
  Clip *clip = reinterpret_cast<Clip *>(clipHandle);
  gRenderState->rodQueries.push_back({clip->name, time, false});
  if (!clip->connected && clip->role != ClipRole::kOutput) return kOfxStatFailed;
  const double par = clip->pixelAspectRatio;
  bounds->x1 = gRenderState->imageOriginX * par;
  bounds->y1 = gRenderState->imageOriginY;
  bounds->x2 = (gRenderState->imageOriginX + gRenderState->width) * par;
  bounds->y2 = gRenderState->imageOriginY + gRenderState->height;
  return kOfxStatOK;
}

int effectAbort(OfxImageEffectHandle) { return 0; }
OfxStatus effectMemoryAlloc(OfxImageEffectHandle, std::size_t, OfxImageMemoryHandle *) {
  return kOfxStatErrUnsupported;
}
OfxStatus effectMemoryFree(OfxImageMemoryHandle) { return kOfxStatErrUnsupported; }
OfxStatus effectMemoryLock(OfxImageMemoryHandle, void **) { return kOfxStatErrUnsupported; }
OfxStatus effectMemoryUnlock(OfxImageMemoryHandle) { return kOfxStatErrUnsupported; }

OfxImageEffectSuiteV1 gImageEffectSuite = {
    effectGetPropertySet, effectGetParamSet, effectClipDefine, effectClipGetHandle,
    effectClipGetPropertySet, effectClipGetImage, effectClipReleaseImage, effectClipGetRoD,
    effectAbort, effectMemoryAlloc, effectMemoryFree, effectMemoryLock, effectMemoryUnlock};

// ---------------------------------------------------------------------------
// Message, memory, progress, timeline and threading suites
// ---------------------------------------------------------------------------

struct HostMessage {
  std::string type;
  std::string text;
};
std::vector<HostMessage> gMessages;

OfxStatus hostMessage(void *, const char *messageType, const char *, const char *format, ...) {
  char buffer[2048];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  gMessages.push_back({messageType == nullptr ? "" : messageType, buffer});
  std::fprintf(stdout, "  [plugin message: %s] %s\n",
               messageType == nullptr ? "" : messageType, buffer);
  return kOfxStatOK;
}

OfxMessageSuiteV1 gMessageSuite = {hostMessage};

OfxStatus hostSetPersistentMessage(void *handle, const char *messageType, const char *,
                                   const char *format, ...) {
  if (handle == nullptr) return kOfxStatErrBadHandle;
  char buffer[2048];
  va_list args;
  va_start(args, format);
  std::vsnprintf(buffer, sizeof(buffer), format, args);
  va_end(args);
  return hostMessage(handle, messageType, nullptr, "%s", buffer);
}

OfxStatus hostClearPersistentMessage(void *handle) {
  return handle == nullptr ? kOfxStatErrBadHandle : kOfxStatOK;
}

OfxMessageSuiteV2 gMessageSuiteV2 = {hostMessage, hostSetPersistentMessage,
                                     hostClearPersistentMessage};

OfxStatus memoryAlloc(void *, std::size_t bytes, void **allocated) {
  if (allocated == nullptr) return kOfxStatErrBadHandle;
  *allocated = std::malloc(bytes);
  return *allocated == nullptr ? kOfxStatErrMemory : kOfxStatOK;
}

OfxStatus memoryFree(void *allocated) {
  std::free(allocated);
  return kOfxStatOK;
}

OfxMemorySuiteV1 gMemorySuite = {memoryAlloc, memoryFree};

unsigned int gReportedCPUs = 4;
thread_local unsigned int gThreadIndex = 0;
thread_local bool gSpawnedThread = false;

OfxStatus hostMultiThread(OfxThreadFunctionV1 function, unsigned int threadCount, void *customArg) {
  if (function == nullptr) return kOfxStatErrBadHandle;
  if (threadCount <= 1) {
    function(0, 1, customArg);
    return kOfxStatOK;
  }
  std::vector<std::thread> threads;
  threads.reserve(threadCount - 1);
  for (unsigned int index = 1; index < threadCount; ++index) {
    threads.emplace_back([function, index, threadCount, customArg]() {
      gThreadIndex = index;
      gSpawnedThread = true;
      function(index, threadCount, customArg);
    });
  }
  function(0, threadCount, customArg);
  for (std::thread &thread : threads) thread.join();
  return kOfxStatOK;
}

OfxStatus hostMultiThreadNumCPUs(unsigned int *count) {
  if (count == nullptr) return kOfxStatErrBadHandle;
  *count = gReportedCPUs;
  return kOfxStatOK;
}

OfxStatus hostMultiThreadIndex(unsigned int *index) {
  if (index == nullptr) return kOfxStatErrBadHandle;
  *index = gThreadIndex;
  return kOfxStatOK;
}

int hostMultiThreadIsSpawnedThread(void) { return gSpawnedThread ? 1 : 0; }
OfxStatus hostMutexCreate(OfxMutexHandle *, int) { return kOfxStatErrUnsupported; }
OfxStatus hostMutexDestroy(const OfxMutexHandle) { return kOfxStatErrUnsupported; }
OfxStatus hostMutexLock(const OfxMutexHandle) { return kOfxStatErrUnsupported; }
OfxStatus hostMutexUnlock(const OfxMutexHandle) { return kOfxStatErrUnsupported; }
OfxStatus hostMutexTryLock(const OfxMutexHandle) { return kOfxStatErrUnsupported; }

OfxMultiThreadSuiteV1 gMultiThreadSuite = {
    hostMultiThread, hostMultiThreadNumCPUs, hostMultiThreadIndex, hostMultiThreadIsSpawnedThread,
    hostMutexCreate, hostMutexDestroy, hostMutexLock, hostMutexUnlock, hostMutexTryLock};

OfxStatus progressStart(void *, const char *) { return kOfxStatOK; }
OfxStatus progressUpdate(void *, double) { return kOfxStatOK; }
OfxStatus progressEnd(void *) { return kOfxStatOK; }
OfxProgressSuiteV1 gProgressSuite = {progressStart, progressUpdate, progressEnd};

OfxStatus timelineGetTime(void *, OfxTime *time) {
  if (time == nullptr) return kOfxStatErrBadHandle;
  *time = gRenderState == nullptr ? 0.0 : gRenderState->renderTime;
  return kOfxStatOK;
}
OfxStatus timelineGotoTime(void *, OfxTime) { return kOfxStatOK; }
OfxStatus timelineGetTimeBounds(void *, OfxTime *start, OfxTime *end) {
  if (start == nullptr || end == nullptr) return kOfxStatErrBadHandle;
  *start = 0.0;
  *end = 3.0;
  return kOfxStatOK;
}
OfxTimeLineSuiteV1 gTimelineSuite = {timelineGetTime, timelineGotoTime, timelineGetTimeBounds};

OfxStatus interactSwapBuffers(OfxInteractHandle) { return kOfxStatOK; }
OfxStatus interactRedraw(OfxInteractHandle) { return kOfxStatOK; }
OfxStatus interactGetPropertySet(OfxInteractHandle interact, OfxPropertySetHandle *propertySet) {
  if (propertySet == nullptr) return kOfxStatErrBadHandle;
  *propertySet = reinterpret_cast<OfxPropertySetHandle>(interact);
  return kOfxStatOK;
}
OfxInteractSuiteV1 gInteractSuite = {interactSwapBuffers, interactRedraw, interactGetPropertySet};

// ---------------------------------------------------------------------------
// Host description and suite dispatch
// ---------------------------------------------------------------------------

PropertySet gHostProperties;
OfxHost gHost;

const void *fetchSuite(OfxPropertySetHandle, const char *suiteName, int suiteVersion) {
  if (suiteName == nullptr) return nullptr;
  if (std::strcmp(suiteName, kOfxPropertySuite) == 0 && suiteVersion == 1) return &gPropertySuite;
  if (std::strcmp(suiteName, kOfxParameterSuite) == 0 && suiteVersion == 1)
    return &gParameterSuite;
  if (std::strcmp(suiteName, kOfxImageEffectSuite) == 0 && suiteVersion == 1)
    return &gImageEffectSuite;
  if (std::strcmp(suiteName, kOfxMessageSuite) == 0 && suiteVersion == 1)
    return &gMessageSuite;
  if (std::strcmp(suiteName, kOfxMessageSuite) == 0 && suiteVersion == 2)
    return &gMessageSuiteV2;
  if (std::strcmp(suiteName, kOfxMemorySuite) == 0 && suiteVersion == 1) return &gMemorySuite;
  if (std::strcmp(suiteName, kOfxMultiThreadSuite) == 0 && suiteVersion == 1)
    return &gMultiThreadSuite;
  if (std::strcmp(suiteName, kOfxProgressSuite) == 0 && suiteVersion == 1)
    return &gProgressSuite;
  if (std::strcmp(suiteName, kOfxTimeLineSuite) == 0 && suiteVersion == 1)
    return &gTimelineSuite;
  if (std::strcmp(suiteName, kOfxInteractSuite) == 0 && suiteVersion == 1)
    return &gInteractSuite;
  return nullptr;
}

void setUpHost(const std::string &hostName) {
  gHostProperties = {};
  const bool flame = hostName == "com.autodesk.flame";
  hostSetString(&gHostProperties, kOfxPropName, hostName);
  hostSetString(&gHostProperties, kOfxPropLabel, "White Water Phase 1 Host Harness");
  hostSetInts(&gHostProperties, kOfxPropVersion, {0, 1, 0});
  hostSetString(&gHostProperties, kOfxPropVersionLabel, "0.1.0");
  hostSetInts(&gHostProperties, kOfxPropAPIVersion, {1, 4});
  hostSetInt(&gHostProperties, kOfxImageEffectHostPropIsBackground, 0);
  if (flame) {
    // Measured Flame shape: the standard property exists but has dimension zero.
    gHostProperties.properties[kOfxImageEffectHostPropNativeOrigin].select(
        Property::Kind::kString);
  } else {
    hostSetString(&gHostProperties, kOfxImageEffectHostPropNativeOrigin,
                  kOfxHostNativeOriginBottomLeft);
  }
  hostSetInt(&gHostProperties, kOfxImageEffectPropSupportsOverlays, 0);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSupportsMultiResolution, 0);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSupportsTiles, 1);
  hostSetInt(&gHostProperties, kOfxImageEffectPropTemporalClipAccess, 1);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSupportsMultipleClipDepths, 0);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSupportsMultipleClipPARs, 0);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSetableFrameRate, 0);
  hostSetInt(&gHostProperties, kOfxImageEffectPropSetableFielding, 0);
  hostSetStrings(&gHostProperties, kOfxImageEffectPropSupportedContexts,
                 {kOfxImageEffectContextGeneral});
  hostSetStrings(&gHostProperties, kOfxImageEffectPropSupportedPixelDepths,
                 {kOfxBitDepthByte, kOfxBitDepthShort, kOfxBitDepthHalf, kOfxBitDepthFloat});
  hostSetStrings(&gHostProperties, kOfxImageEffectPropSupportedComponents,
                 {kOfxImageComponentRGBA, kOfxImageComponentRGB, kOfxImageComponentAlpha});
  // Flame reports GPU booleans as strings.  A plugin must read these in a non-throwing way and
  // decline GPU render suites; inference is entirely outside OFX's render suites.
  hostSetString(&gHostProperties, kOfxImageEffectPropOpenGLRenderSupported,
                flame ? "true" : "false");
  hostSetString(&gHostProperties, kOfxImageEffectPropCudaRenderSupported, "false");
  hostSetString(&gHostProperties, kOfxImageEffectPropMetalRenderSupported, "false");
  hostSetString(&gHostProperties, kOfxImageEffectPropOpenCLRenderSupported, "false");
  hostSetInt(&gHostProperties, kOfxParamHostPropSupportsCustomInteract, 0);
  hostSetInt(&gHostProperties, kOfxParamHostPropSupportsStringAnimation, 0);
  hostSetInt(&gHostProperties, kOfxParamHostPropSupportsChoiceAnimation, flame ? 1 : 0);
  hostSetInt(&gHostProperties, kOfxParamHostPropSupportsBooleanAnimation, flame ? 1 : 0);
  hostSetInt(&gHostProperties, kOfxParamHostPropSupportsCustomAnimation, 0);
  hostSetInt(&gHostProperties, kOfxParamHostPropMaxParameters, -1);
  hostSetInt(&gHostProperties, kOfxParamHostPropMaxPages, 0);
  hostSetInts(&gHostProperties, kOfxParamHostPropPageRowColumnCount, {0, 0});

  gHost.host = reinterpret_cast<OfxPropertySetHandle>(&gHostProperties);
  gHost.fetchSuite = fetchSuite;
}

// ---------------------------------------------------------------------------
// Descriptor/instance driving helpers
// ---------------------------------------------------------------------------

struct DescriptorRun {
  OfxPlugin *plugin = nullptr;
  Effect descriptor;
  Effect context;
  std::string identifier;
};

struct LoadedBundle {
  void *handle = nullptr;
  std::vector<DescriptorRun> descriptors;
};

bool statusOK(OfxStatus status) {
  return status == kOfxStatOK || status == kOfxStatReplyDefault;
}

void expectStatus(const std::string &action, OfxStatus status) {
  if (statusOK(status))
    info(action + ": ok");
  else
    fail(action + " returned status " + std::to_string(status));
}

void expectRequiredStatus(const std::string &action, OfxStatus status) {
  if (status == kOfxStatOK)
    info(action + ": ok");
  else
    fail(action + " must return kOfxStatOK, got " + std::to_string(status));
}

void cloneParamSet(const ParamSet &source, ParamSet *destination) {
  for (const auto &sourceParam : source.params) {
    auto destinationParam = std::make_unique<Parameter>();
    destinationParam->name = sourceParam->name;
    destinationParam->type = sourceParam->type;
    destinationParam->props = sourceParam->props;
    destinationParam->stringValue = sourceParam->stringValue;
    destinationParam->intValue = sourceParam->intValue;
    destinationParam->doubleValue = sourceParam->doubleValue;
    destinationParam->doubleValues = sourceParam->doubleValues;
    destination->params.push_back(std::move(destinationParam));
  }
}

void cloneEffectDescriptor(const Effect &source, Effect *destination) {
  destination->props = source.props;
  cloneParamSet(source.params, &destination->params);
  for (const auto &sourceClip : source.clips) {
    auto destinationClip = std::make_unique<Clip>();
    destinationClip->name = sourceClip->name;
    destinationClip->role = sourceClip->role;
    destinationClip->props = sourceClip->props;
    destinationClip->depth = sourceClip->depth;
    destinationClip->components = sourceClip->components;
    destinationClip->pixelAspectRatio = sourceClip->pixelAspectRatio;
    destination->clips.push_back(std::move(destinationClip));
  }
}

PropertySet actionArgs(double time, const OfxRectI *window = nullptr) {
  PropertySet args;
  hostSetDouble(&args, kOfxPropTime, time);
  hostSetString(&args, kOfxImageEffectPropFieldToRender, kOfxImageFieldNone);
  hostSetDoubles(&args, kOfxImageEffectPropRenderScale, {1.0, 1.0});
  hostSetInt(&args, kOfxImageEffectPropSequentialRenderStatus, 0);
  hostSetInt(&args, kOfxImageEffectPropInteractiveRenderStatus, 0);
  hostSetInt(&args, kOfxImageEffectPropRenderQualityDraft, 0);
  hostSetInt(&args, kOfxImageEffectPropNoSpatialAwareness, 0);
  if (window != nullptr)
    hostSetInts(&args, kOfxImageEffectPropRenderWindow,
                {window->x1, window->y1, window->x2, window->y2});
  return args;
}

void configureClip(Clip *clip, const RenderState &state, bool connected) {
  if (clip == nullptr) return;
  clip->connected = connected;
  clip->depth = state.depth;
  clip->components = state.components;
  clip->pixelAspectRatio = state.pixelAspectRatio;
  hostSetInt(&clip->props, kOfxImageClipPropConnected, connected ? 1 : 0);
  hostSetInt(&clip->props, kOfxImageClipPropOptional,
             clip->role == ClipRole::kInsert ? 1 : 0);
  hostSetString(&clip->props, kOfxImageEffectPropPixelDepth,
                state.depth);
  hostSetString(&clip->props, kOfxImageEffectPropComponents,
                state.components);
  hostSetString(&clip->props, kOfxImageEffectPropPreMultiplication,
                kOfxImageUnPreMultiplied);
  hostSetDouble(&clip->props, kOfxImagePropPixelAspectRatio, state.pixelAspectRatio);
  hostSetDoubles(&clip->props, kOfxImageEffectPropFrameRange, {0.0, 3.0});
  hostSetDoubles(&clip->props, kOfxImageEffectPropUnmappedFrameRange, {0.0, 3.0});
  hostSetString(&clip->props, kOfxImageClipPropUnmappedPixelDepth,
                state.depth);
  hostSetString(&clip->props, kOfxImageClipPropUnmappedComponents,
                state.components);
}

void configureInstance(Effect *instance, const RenderState &state, bool insertConnected) {
  Clip *source = instance->find("Source");
  Clip *insert = instance->find("Insert");
  Clip *output = instance->find(kOfxImageEffectOutputClipName);
  configureClip(source, state, true);
  configureClip(insert, state, insertConnected);
  configureClip(output, state, true);
  if (output != nullptr) {
    output->depth = state.depth;
    const std::vector<std::string> supported = stringsFrom(
        reinterpret_cast<OfxPropertySetHandle>(&output->props),
        kOfxImageEffectPropSupportedComponents);
    output->components = state.components;
    if (!supported.empty() &&
        std::find(supported.begin(), supported.end(), output->components) == supported.end())
      output->components = supported.front();
  }
}

bool setIntParameter(Effect *instance, const std::string &name, int value) {
  Parameter *parameter = instance->params.find(name);
  if (parameter == nullptr) {
    fail("parameter is missing: " + name);
    return false;
  }
  parameter->intValue = value;
  return true;
}

bool setChoiceByLabel(Effect *instance, const std::string &name, const std::string &label) {
  Parameter *parameter = instance->params.find(name);
  if (parameter == nullptr) {
    fail("choice parameter is missing: " + name);
    return false;
  }
  const std::vector<std::string> options = stringsFrom(
      reinterpret_cast<OfxPropertySetHandle>(&parameter->props), kOfxParamPropChoiceOption);
  for (std::size_t index = 0; index < options.size(); ++index) {
    if (options[index] == label) {
      parameter->intValue = static_cast<int>(index);
      return true;
    }
  }
  fail("choice " + name + " has no option " + label);
  return false;
}

void initializeFrameRanges(const Effect &instance, double time, PropertySet *outArgs) {
  for (const auto &clip : instance.clips) {
    if (clip->role == ClipRole::kOutput) continue;
    hostSetDoubles(outArgs, "OfxImageClipPropFrameRange_" + clip->name, {time, time});
  }
}

bool noPullsSince(const RenderState &state, std::size_t before, const std::string &reason) {
  if (state.pulls.size() == before) return true;
  fail(reason + " caused " + std::to_string(state.pulls.size() - before) +
       " clipGetImage pull(s)");
  return false;
}

bool outputFormatFromPreferences(const PropertySet &outArgs, Effect *instance,
                                 const RenderState &state) {
  Clip *output = instance->find(kOfxImageEffectOutputClipName);
  if (output == nullptr) return false;
  std::string depth;
  std::string components;
  double par = state.pixelAspectRatio;
  const std::string depthKey = "OfxImageClipPropDepth_" + output->name;
  const std::string componentsKey = "OfxImageClipPropComponents_" + output->name;
  const std::string parKey = "OfxImageClipPropPAR_" + output->name;
  if (readString(reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&outArgs)),
                 depthKey, &depth))
    output->depth = depth;
  if (readString(reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&outArgs)),
                 componentsKey, &components))
    output->components = components;
  readDouble(reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&outArgs)), parKey,
             &par);
  output->pixelAspectRatio = par;
  return !output->depth.empty() && !output->components.empty();
}

bool createOutputImage(Effect *instance, const RenderState &state) {
  Clip *output = instance->find(kOfxImageEffectOutputClipName);
  if (output == nullptr) return false;
  instance->outputImage = std::make_unique<ImageAllocation>();
  const OfxRectI window = state.renderWindow;
  allocatePlane(&instance->outputImage->plane, window.x2 - window.x1, window.y2 - window.y1,
                output->depth, output->components, window.x1, window.y1, state.rowPadding);
  fillImageProperties(instance->outputImage.get(), output);
  return true;
}

bool invokeClipPreferences(OfxPlugin *plugin, Effect *instance, const RenderState &state) {
  PropertySet outArgs;
  // These are host defaults; a plugin may override them by setting the documented suffixed
  // properties.  This is also how the host keeps a byte/RGB input from being implicitly
  // remapped to float when SupportsMultipleClipDepths is false.
  for (const auto &clip : instance->clips) {
    if (clip->role == ClipRole::kOutput) continue;
    hostSetString(&outArgs, "OfxImageClipPropDepth_" + clip->name, state.depth);
    hostSetString(&outArgs, "OfxImageClipPropComponents_" + clip->name,
                  state.components);
    hostSetDouble(&outArgs, "OfxImageClipPropPAR_" + clip->name,
                  state.pixelAspectRatio);
  }
  hostSetString(&outArgs, "OfxImageClipPropDepth_Output", state.depth);
  const Clip *output = instance->find(kOfxImageEffectOutputClipName);
  hostSetString(&outArgs, "OfxImageClipPropComponents_Output",
                output == nullptr ? state.components : output->components);
  hostSetDouble(&outArgs, "OfxImageClipPropPAR_Output", state.pixelAspectRatio);
  hostSetString(&outArgs, kOfxImageEffectPropPreMultiplication, kOfxImageUnPreMultiplied);
  const OfxStatus status = plugin->mainEntry(kOfxImageEffectActionGetClipPreferences,
                                             reinterpret_cast<OfxImageEffectHandle>(instance),
                                             nullptr,
                                             reinterpret_cast<OfxPropertySetHandle>(&outArgs));
  if (!statusOK(status)) {
    fail("getClipPreferences returned status " + std::to_string(status));
    return false;
  }
  if (!outputFormatFromPreferences(outArgs, instance, state)) {
    fail("getClipPreferences did not leave a usable Output depth/components pair");
    return false;
  }
  std::string premultiplication;
  if (!readString(reinterpret_cast<OfxPropertySetHandle>(&outArgs),
                  kOfxImageEffectPropPreMultiplication, &premultiplication)) {
    fail("getClipPreferences omitted output premultiplication");
  } else {
    check(premultiplication == kOfxImageUnPreMultiplied,
          "output clip preference is unpremultiplied");
  }
  return true;
}

struct ExpectedParameter {
  const char *name;
  const char *type;
  const char *label;
};

const std::vector<ExpectedParameter> kTrackParameters = {
    {"refFrame", kOfxParamTypeInteger, "Ref Frame"},
    {"setRef", kOfxParamTypePushButton, "Set Ref"},
    {"output", kOfxParamTypeChoice, "Output"},
    {"insertTime", kOfxParamTypeChoice, "Insert At"},
    {"matte", kOfxParamTypeChoice, "Matte"},
    {"iterations", kOfxParamTypeInteger, "Iters"},
    {"smooth", kOfxParamTypeDouble, "Smooth"},
    {"fbCheck", kOfxParamTypeBoolean, "FB Check"},
    {"fbTolerance", kOfxParamTypeDouble, "FB Tol"},
    {"filter", kOfxParamTypeChoice, "Filter"},
    {"edges", kOfxParamTypeChoice, "Edges"},
    {"device", kOfxParamTypeChoice, "Device"},
    {"threads", kOfxParamTypeInteger, "Threads"},
    {"cacheMB", kOfxParamTypeInteger, "Cache MB"},
    {"precacheRange", kOfxParamTypeChoice, "Pre Range"},
    {"precacheStart", kOfxParamTypeInteger, "Pre Start"},
    {"precacheEnd", kOfxParamTypeInteger, "Pre End"},
    {"precache", kOfxParamTypePushButton, "Precache"},
    {"clearCache", kOfxParamTypePushButton, "Clear"},
    {"modelDir", kOfxParamTypeString, "Model Dir"}};

const std::vector<ExpectedParameter> kStParameters = {
    {"refFrame", kOfxParamTypeInteger, "Ref Frame"},
    {"setRef", kOfxParamTypePushButton, "Set Ref"},
    {"matte", kOfxParamTypeChoice, "Matte"},
    {"iterations", kOfxParamTypeInteger, "Iters"},
    {"smooth", kOfxParamTypeDouble, "Smooth"},
    {"fbCheck", kOfxParamTypeBoolean, "FB Check"},
    {"fbTolerance", kOfxParamTypeDouble, "FB Tol"},
    {"filter", kOfxParamTypeChoice, "Filter"},
    {"edges", kOfxParamTypeChoice, "Edges"},
    {"device", kOfxParamTypeChoice, "Device"},
    {"threads", kOfxParamTypeInteger, "Threads"},
    {"cacheMB", kOfxParamTypeInteger, "Cache MB"},
    {"precacheRange", kOfxParamTypeChoice, "Pre Range"},
    {"precacheStart", kOfxParamTypeInteger, "Pre Start"},
    {"precacheEnd", kOfxParamTypeInteger, "Pre End"},
    {"precache", kOfxParamTypePushButton, "Precache"},
    {"clearCache", kOfxParamTypePushButton, "Clear"},
    {"modelDir", kOfxParamTypeString, "Model Dir"},
    {"stMode", kOfxParamTypeChoice, "ST Mode"},
    {"stOrigin", kOfxParamTypeChoice, "ST Origin"}};

void validateParameterSet(const Effect &context, bool stDescriptor) {
  const std::vector<ExpectedParameter> &expected =
      stDescriptor ? kStParameters : kTrackParameters;
  check(context.params.params.size() == expected.size(),
        (stDescriptor ? "ST" : "Track") + std::string(" parameter count is ") +
            std::to_string(context.params.params.size()) + " (expected " +
            std::to_string(expected.size()) + ")");
  const std::size_t count = std::min(context.params.params.size(), expected.size());
  for (std::size_t i = 0; i < count; ++i) {
    const Parameter &parameter = *context.params.params[i];
    const ExpectedParameter &want = expected[i];
    check(parameter.name == want.name,
          "parameter " + std::to_string(i) + " is " + parameter.name + " (expected " +
              want.name + ")");
    check(parameter.type == want.type,
          "parameter " + parameter.name + " has documented type");
    std::string label;
    if (!readString(reinterpret_cast<OfxPropertySetHandle>(
                        const_cast<PropertySet *>(&parameter.props)),
                    kOfxPropLabel, &label)) {
      fail("parameter " + parameter.name + " has no label");
    } else {
      check(label == want.label,
            "parameter " + parameter.name + " label is \"" + label + "\"");
      check(label.size() <= 12, "parameter " + parameter.name + " label fits Flame's panel");
    }
    if (parameter.type == kOfxParamTypeCustom || parameter.type == kOfxParamTypeParametric)
      fail("unsupported custom/parametric parameter was defined: " + parameter.name);
    if (parameter.name == "refFrame") {
      int animates = 1;
      const bool hasAnimates = readInt(
          reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&parameter.props)),
          kOfxParamPropAnimates, &animates);
      check(hasAnimates && animates == 0, "refFrame is a non-animating scalar");
    }
    if (parameter.type == kOfxParamTypeChoice) {
      const std::vector<std::string> options = stringsFrom(
          reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&parameter.props)),
          kOfxParamPropChoiceOption);
      check(!options.empty(), "choice " + parameter.name + " has options");
    }
  }
  if (context.params.params.size() > expected.size()) {
    for (std::size_t i = expected.size(); i < context.params.params.size(); ++i)
      fail("unexpected extra parameter at index " + std::to_string(i) + ": " +
           context.params.params[i]->name);
  }

  for (const char *deferred : {"model", "inputCurve", "analysisScale"}) {
    check(context.params.find(deferred) == nullptr,
          std::string("deferred choice ") + deferred +
              " is absent until a measured option ordering exists");
  }

  // Option order is persisted in Flame setups.  These four pairs are the options whose
  // semantics Phase 1 exercises; bake-off-controlled choices are absent until Phase 2.5.
  const std::vector<std::pair<std::string, std::vector<std::string>>> fixedOptions = {
      {"output", {"Composite", "Warped Insert"}},
      {"insertTime", {"Current", "Reference"}},
      {"stMode", {"Absolute UV", "Relative Pixels"}},
      {"stOrigin", {"Bottom Left", "Top Left"}}};
  for (const auto &fixed : fixedOptions) {
    const Parameter *parameter = context.params.find(fixed.first);
    if (parameter == nullptr) continue;
    const std::vector<std::string> options = stringsFrom(
        reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&parameter->props)),
        kOfxParamPropChoiceOption);
    if (options.size() < fixed.second.size()) {
      fail("choice " + fixed.first + " has too few options");
      continue;
    }
    for (std::size_t i = 0; i < fixed.second.size(); ++i)
      check(options[i] == fixed.second[i],
            "choice " + fixed.first + " option " + std::to_string(i) + " is " + options[i]);
  }
}

void validateEffectProperties(const Effect &context, bool stDescriptor) {
  std::string grouping;
  const bool hasGrouping = readString(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPluginPropGrouping, &grouping);
  check(hasGrouping && grouping == "White Water",
        (stDescriptor ? "ST" : "Track") + std::string(" descriptor uses White Water menu"));

  const auto supportedDepths = stringsFrom(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropSupportedPixelDepths);
  if (stDescriptor) {
    check(std::find(supportedDepths.begin(), supportedDepths.end(), kOfxBitDepthFloat) !=
              supportedDepths.end(),
          "ST descriptor supports float depth");
    check(supportedDepths.size() == 1 && supportedDepths.front() == kOfxBitDepthFloat,
          "ST descriptor is float-only");
  } else {
    for (const char *depth : {kOfxBitDepthByte, kOfxBitDepthShort, kOfxBitDepthHalf,
                              kOfxBitDepthFloat})
      check(std::find(supportedDepths.begin(), supportedDepths.end(), depth) !=
                supportedDepths.end(),
            std::string("Track supports ") + depth);
  }
  // Components are clip properties, not effect-wide properties in OFX.  They are checked in
  // validateClips below; an empty effect-wide list is therefore not a contract failure.

  int tiles = 1;
  const bool hasTiles = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropSupportsTiles, &tiles);
  check(hasTiles && tiles == 0,
        "descriptor declares whole-frame flow (SupportsTiles=0)");
  int multires = 1;
  const bool hasMultires = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropSupportsMultiResolution, &multires);
  check(hasMultires && multires == 0, "descriptor declines multi-resolution");
  std::string threading;
  const bool hasThreading = readString(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPluginRenderThreadSafety, &threading);
  check(hasThreading && threading == kOfxImageEffectRenderInstanceSafe,
        "descriptor uses instance-safe render threading");
  int hostThreading = 1;
  const bool hasHostThreading = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPluginPropHostFrameThreading, &hostThreading);
  check(hasHostThreading && hostThreading == 0, "descriptor declines host frame threading");
  std::string openGL;
  const bool hasOpenGL = readString(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropOpenGLRenderSupported, &openGL);
  check(hasOpenGL && openGL == "false", "descriptor declines OpenGL render suite");

  int temporal = 0;
  const bool hasTemporal = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropTemporalClipAccess, &temporal);
  check(hasTemporal && temporal == 1, "descriptor declares temporal clip access");
  int multipleDepths = 1;
  const bool hasMultipleDepths = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropSupportsMultipleClipDepths, &multipleDepths);
  check(hasMultipleDepths && multipleDepths == 0,
        "descriptor respects Flame's common-depth contract");
  int multiplePars = 1;
  const bool hasMultiplePars = readInt(
      reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&context.props)),
      kOfxImageEffectPropSupportsMultipleClipPARs, &multiplePars);
  check(hasMultiplePars && multiplePars == 0,
        "descriptor respects Flame's common-PAR contract");
}

void validateClips(const Effect &context, bool stDescriptor) {
  std::set<std::string> names;
  for (const auto &clip : context.clips) names.insert(clip->name);
  const std::set<std::string> expected = stDescriptor
                                             ? std::set<std::string>{"Source", "Output"}
                                             : std::set<std::string>{"Source", "Insert", "Output"};
  check(names == expected, (stDescriptor ? "ST" : "Track") + std::string(" clip names match"));
  Clip *source = context.find("Source");
  Clip *insert = context.find("Insert");
  Clip *output = context.find("Output");
  check(source != nullptr && output != nullptr, "required Source and Output clips are defined");
  if (source != nullptr) {
    int optional = 1;
    readInt(reinterpret_cast<OfxPropertySetHandle>(&source->props), kOfxImageClipPropOptional,
            &optional);
    check(optional == 0, "Source clip is required");
    int temporal = 0;
    const bool hasTemporal = readInt(
        reinterpret_cast<OfxPropertySetHandle>(&source->props),
        kOfxImageEffectPropTemporalClipAccess, &temporal);
    check(hasTemporal && temporal == 1, "Source clip declares temporal access");
  }
  if (stDescriptor) {
    check(insert == nullptr, "ST descriptor does not define an Insert clip");
  } else if (insert != nullptr) {
    int optional = 0;
    readInt(reinterpret_cast<OfxPropertySetHandle>(&insert->props), kOfxImageClipPropOptional,
            &optional);
    check(optional == 1, "Track Insert clip is optional");
    int temporal = 0;
    const bool hasTemporal = readInt(
        reinterpret_cast<OfxPropertySetHandle>(&insert->props),
        kOfxImageEffectPropTemporalClipAccess, &temporal);
    check(hasTemporal && temporal == 1, "Track Insert clip declares temporal access");
  }
  if (output != nullptr) {
    int optional = 1;
    readInt(reinterpret_cast<OfxPropertySetHandle>(&output->props), kOfxImageClipPropOptional,
            &optional);
    check(optional == 0, "Output clip is not optional");
  }
  const auto checkWholeFrameClip = [](const Clip *clip, const std::string &label) {
    if (clip == nullptr) return;
    int tiles = 1;
    const bool present = readInt(
        reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&clip->props)),
        kOfxImageEffectPropSupportsTiles, &tiles);
    check(present && tiles == 0, label + " declines tiled images");
  };
  checkWholeFrameClip(source, stDescriptor ? "ST Source" : "Track Source");
  checkWholeFrameClip(insert, "Track Insert");
  checkWholeFrameClip(output, stDescriptor ? "ST Output" : "Track Output");
  const auto checkComponents = [](const Clip *clip, const std::vector<const char *> &expected,
                                  const std::string &label) {
    if (clip == nullptr) return;
    const std::vector<std::string> actual = stringsFrom(
        reinterpret_cast<OfxPropertySetHandle>(const_cast<PropertySet *>(&clip->props)),
        kOfxImageEffectPropSupportedComponents);
    for (const char *component : expected)
      check(std::find(actual.begin(), actual.end(), component) != actual.end(),
            label + " supports " + component);
  };
  const std::vector<const char *> trackInputComponents = {
      kOfxImageComponentRGBA, kOfxImageComponentRGB, kOfxImageComponentAlpha};
  checkComponents(source,
                  stDescriptor ? std::vector<const char *>{kOfxImageComponentRGBA}
                               : trackInputComponents,
                  stDescriptor ? "ST Source" : "Track Source");
  checkComponents(insert, trackInputComponents, "Track Insert");
  checkComponents(output,
                  stDescriptor ? std::vector<const char *>{kOfxImageComponentRGBA}
                               : trackInputComponents,
                  stDescriptor ? "ST Output" : "Track Output");
}

struct RenderResult {
  bool statusOK = false;
  Plane output;
  std::vector<Pull> pulls;
};

std::vector<Pull> inputPulls(const RenderState &state) {
  std::vector<Pull> pulls;
  for (const Pull &pull : state.pulls)
    if (pull.clip == "Source" || pull.clip == "Insert") pulls.push_back(pull);
  return pulls;
}

RenderResult renderOne(OfxPlugin *plugin, Effect *instance, RenderState state,
                       bool insertConnected, const std::string &outputChoice = "Composite",
                       const std::string &insertTimeChoice = "Current") {
  configureInstance(instance, state, insertConnected);
  if (instance->params.find("output") != nullptr)
    setChoiceByLabel(instance, "output", outputChoice);
  if (!insertTimeChoice.empty() && instance->params.find("insertTime") != nullptr)
    setChoiceByLabel(instance, "insertTime", insertTimeChoice);
  if (!invokeClipPreferences(plugin, instance, state)) return {};
  if (!createOutputImage(instance, state)) return {};

  gActiveEffect = instance;
  gRenderState = &state;
  PropertySet args = actionArgs(state.renderTime, &state.renderWindow);
  const OfxStatus status = plugin->mainEntry(
      kOfxImageEffectActionRender, reinterpret_cast<OfxImageEffectHandle>(instance),
      reinterpret_cast<OfxPropertySetHandle>(&args), nullptr);
  RenderResult result;
  result.statusOK = status == kOfxStatOK;
  result.pulls = inputPulls(state);
  if (!statusOK(status))
    fail("render at time " + std::to_string(state.renderTime) + " returned status " +
         std::to_string(status));
  if (instance->outputImage != nullptr) result.output = instance->outputImage->plane;
  if (!instance->liveImages.empty()) {
    fail("render leaked " + std::to_string(instance->liveImages.size()) + " input image(s)");
    instance->liveImages.clear();
  }
  instance->outputImage.reset();
  gRenderState = nullptr;
  gActiveEffect = nullptr;
  return result;
}

Plane expectedPlaneFor(const Plane &outputShape, int clipTag, double time) {
  Plane expected;
  const int width = outputShape.width();
  const int height = outputShape.height();
  const int rowPadding = outputShape.rowBytes - outputShape.pixelBytes * width;
  allocatePlane(&expected, width, height, outputShape.depth, outputShape.components,
                outputShape.bounds.x1, outputShape.bounds.y1, rowPadding);
  fillSentinel(&expected, clipTag, time);
  return expected;
}

float halfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  const std::uint32_t exponent = (value >> 10) & 0x1fu;
  const std::uint32_t mantissa = value & 0x3ffu;
  std::uint32_t bits = sign;
  if (exponent == 0) {
    if (mantissa == 0) {
      float result = 0.0f;
      std::memcpy(&result, &bits, sizeof(result));
      return result;
    }
    return static_cast<float>(mantissa) * 5.9604644775390625e-08f *
           (sign == 0 ? 1.0f : -1.0f);
  }
  if (exponent == 31) bits |= 0x7f800000u | (mantissa << 13);
  else bits |= ((exponent + 112u) << 23) | (mantissa << 13);
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa >> 13));
  const int halfExponent = static_cast<int>(exponent) - 127 + 15;
  if (halfExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (halfExponent <= 0) return sign;
  return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(halfExponent) << 10) |
                                    ((mantissa + 0x1000u) >> 13));
}

float readNormalized(const Plane &plane, int x, int y, int channel) {
  const unsigned char *pixel = plane.pixel(x, y);
  if (plane.depth == kOfxBitDepthByte) return static_cast<float>(pixel[channel]) / 255.0f;
  if (plane.depth == kOfxBitDepthShort) {
    std::uint16_t value = 0;
    std::memcpy(&value, pixel + channel * 2, sizeof(value));
    return static_cast<float>(value) / 65535.0f;
  }
  if (plane.depth == kOfxBitDepthHalf) {
    std::uint16_t value = 0;
    std::memcpy(&value, pixel + channel * 2, sizeof(value));
    return halfToFloat(value);
  }
  float value = 0.0f;
  std::memcpy(&value, pixel + channel * 4, sizeof(value));
  return value;
}

void writeNormalized(Plane *plane, int x, int y, int channel, float value) {
  unsigned char *pixel = plane->pixel(x, y);
  value = std::isfinite(value) ? std::max(0.0f, std::min(1.0f, value)) : 0.0f;
  if (plane->depth == kOfxBitDepthByte) {
    pixel[channel] = static_cast<unsigned char>(value * 255.0f + 0.5f);
  } else if (plane->depth == kOfxBitDepthShort) {
    const std::uint16_t converted = static_cast<std::uint16_t>(value * 65535.0f + 0.5f);
    std::memcpy(pixel + channel * 2, &converted, sizeof(converted));
  } else if (plane->depth == kOfxBitDepthHalf) {
    const std::uint16_t converted = floatToHalf(value);
    std::memcpy(pixel + channel * 2, &converted, sizeof(converted));
  } else {
    std::memcpy(pixel + channel * 4, &value, sizeof(value));
  }
}

Plane expectedFallbackCopy(const Plane &outputShape, const RenderState &state, int clipTag,
                          double time) {
  Plane input;
  allocatePlane(&input, state.width, state.height, state.depth, state.components,
                state.imageOriginX, state.imageOriginY, state.rowPadding);
  fillSentinel(&input, clipTag, time);
  Plane expected;
  const int rowPadding = outputShape.rowBytes - outputShape.pixelBytes * outputShape.width();
  allocatePlane(&expected, outputShape.width(), outputShape.height(), outputShape.depth,
                outputShape.components, outputShape.bounds.x1, outputShape.bounds.y1,
                rowPadding);
  const int outputChannels = componentCount(expected.components);
  for (int y = expected.bounds.y1; y < expected.bounds.y2; ++y) {
    for (int x = expected.bounds.x1; x < expected.bounds.x2; ++x) {
      const int sourceX = std::max(input.bounds.x1, std::min(input.bounds.x2 - 1, x));
      const int sourceY = std::max(input.bounds.y1, std::min(input.bounds.y2 - 1, y));
      float source[4] = {0.0f, 0.0f, 0.0f, 1.0f};
      if (input.components == kOfxImageComponentRGBA) {
        for (int c = 0; c < 4; ++c) source[c] = readNormalized(input, sourceX, sourceY, c);
      } else if (input.components == kOfxImageComponentRGB) {
        for (int c = 0; c < 3; ++c) source[c] = readNormalized(input, sourceX, sourceY, c);
      } else if (input.components == kOfxImageComponentAlpha) {
        source[3] = readNormalized(input, sourceX, sourceY, 0);
      }
      if (outputChannels == 4) {
        for (int c = 0; c < 4; ++c) writeNormalized(&expected, x, y, c, source[c]);
      } else if (outputChannels == 3) {
        for (int c = 0; c < 3; ++c) writeNormalized(&expected, x, y, c, source[c]);
      } else if (outputChannels == 1) {
        writeNormalized(&expected, x, y, 0, source[3]);
      }
    }
  }
  return expected;
}

bool isBlack(const Plane &plane) {
  return std::all_of(plane.bytes.begin(), plane.bytes.end(), [](unsigned char byte) {
    return byte == 0;
  });
}

void verifyCheapQueries(OfxPlugin *plugin, Effect *instance, double time) {
  RenderState queryState;
  queryState.renderTime = time;
  gActiveEffect = instance;
  gRenderState = &queryState;
  const std::size_t before = queryState.pulls.size();

  PropertySet frameArgs;
  hostSetDouble(&frameArgs, kOfxPropTime, time);
  PropertySet frameOut;
  initializeFrameRanges(*instance, time, &frameOut);
  expectStatus("getFramesNeeded", plugin->mainEntry(
                                  kOfxImageEffectActionGetFramesNeeded,
                                  reinterpret_cast<OfxImageEffectHandle>(instance),
                                  reinterpret_cast<OfxPropertySetHandle>(&frameArgs),
                                  reinterpret_cast<OfxPropertySetHandle>(&frameOut)));
  noPullsSince(queryState, before, "getFramesNeeded");

  PropertySet roiArgs;
  hostSetDouble(&roiArgs, kOfxPropTime, time);
  hostSetDoubles(&roiArgs, kOfxImageEffectPropRenderScale, {1.0, 1.0});
  hostSetDoubles(&roiArgs, kOfxImageEffectPropRegionOfInterest, {0.0, 0.0, 12.0, 8.0});
  PropertySet roiOut;
  const std::size_t beforeRoi = queryState.pulls.size();
  expectStatus("getRegionsOfInterest", plugin->mainEntry(
                                      kOfxImageEffectActionGetRegionsOfInterest,
                                      reinterpret_cast<OfxImageEffectHandle>(instance),
                                      reinterpret_cast<OfxPropertySetHandle>(&roiArgs),
                                      reinterpret_cast<OfxPropertySetHandle>(&roiOut)));
  noPullsSince(queryState, beforeRoi, "getRegionsOfInterest");
  const std::vector<double> sourceRoi = doublesFrom(
      reinterpret_cast<OfxPropertySetHandle>(&roiOut), "OfxImageClipPropRoI_Source");
  check(sourceRoi == std::vector<double>({0.0, 0.0, 12.0, 8.0}),
        "getRegionsOfInterest requests the complete connected Source RoD");

  const OfxRectI identityWindow = {0, 0, queryState.width, queryState.height};
  PropertySet identityArgs = actionArgs(time, &identityWindow);
  PropertySet identityOut;
  const std::size_t beforeIdentity = queryState.pulls.size();
  const OfxStatus identityStatus = plugin->mainEntry(
      kOfxImageEffectActionIsIdentity, reinterpret_cast<OfxImageEffectHandle>(instance),
      reinterpret_cast<OfxPropertySetHandle>(&identityArgs),
      reinterpret_cast<OfxPropertySetHandle>(&identityOut));
  expectStatus("isIdentity", identityStatus);
  noPullsSince(queryState, beforeIdentity, "isIdentity");
  if (instance->find("Insert") != nullptr) {
    std::string identityClip;
    double identityTime = -1.0;
    const bool hasClip = readString(reinterpret_cast<OfxPropertySetHandle>(&identityOut),
                                    kOfxPropName, &identityClip);
    const bool hasTime = readDouble(reinterpret_cast<OfxPropertySetHandle>(&identityOut),
                                    kOfxPropTime, &identityTime);
    check(identityStatus == kOfxStatOK && hasClip && identityClip == "Source" && hasTime &&
              identityTime == time,
          "Composite isIdentity returns Source at N without inference");
  } else {
    check(identityStatus == kOfxStatReplyDefault,
          "ST isIdentity conservatively requires its identity-grid render");
  }
  gRenderState = nullptr;
  gActiveEffect = nullptr;

  const std::vector<double> sourceRange = doublesFrom(
      reinterpret_cast<OfxPropertySetHandle>(&frameOut), "OfxImageClipPropFrameRange_Source");
  check(sourceRange.size() >= 2 && sourceRange[0] == time && sourceRange[1] == time,
        "getFramesNeeded declares only Source N");
  const std::vector<double> insertRange = doublesFrom(
      reinterpret_cast<OfxPropertySetHandle>(&frameOut), "OfxImageClipPropFrameRange_Insert");
  if (instance->find("Insert") != nullptr) {
    check(insertRange.size() >= 2 && insertRange[0] == time && insertRange[1] == time,
          "getFramesNeeded defaults Insert to N");
    setChoiceByLabel(instance, "insertTime", "Reference");
    PropertySet referenceOut;
    initializeFrameRanges(*instance, time, &referenceOut);
    expectStatus("getFramesNeeded (Reference)", plugin->mainEntry(
                                         kOfxImageEffectActionGetFramesNeeded,
                                         reinterpret_cast<OfxImageEffectHandle>(instance),
                                         reinterpret_cast<OfxPropertySetHandle>(&frameArgs),
                                         reinterpret_cast<OfxPropertySetHandle>(&referenceOut)));
    const std::vector<double> referenceRange = doublesFrom(
        reinterpret_cast<OfxPropertySetHandle>(&referenceOut), "OfxImageClipPropFrameRange_Insert");
    const Parameter *referenceFrame = instance->params.find("refFrame");
    const double referenceTime =
        referenceFrame == nullptr ? 0.0 : static_cast<double>(referenceFrame->intValue);
    check(referenceRange.size() >= 2 && referenceRange[0] == referenceTime &&
              referenceRange[1] == referenceTime,
          "getFramesNeeded declares Insert R for Reference mode");
    setChoiceByLabel(instance, "insertTime", "Current");
  }
}

void testTrackRenderContract(OfxPlugin *plugin, const Effect &descriptor) {
  const std::vector<std::string> depths = {kOfxBitDepthByte, kOfxBitDepthShort,
                                           kOfxBitDepthHalf, kOfxBitDepthFloat};
  const std::vector<std::string> components = {kOfxImageComponentRGBA,
                                                kOfxImageComponentRGB,
                                                kOfxImageComponentAlpha};

  Effect instance;
  cloneEffectDescriptor(descriptor, &instance);
  expectStatus("createInstance (Track contract)", plugin->mainEntry(
                                                kOfxActionCreateInstance,
                                                reinterpret_cast<OfxImageEffectHandle>(&instance),
                                                nullptr, nullptr));

  // `Set Ref` is the artist-facing bridge from Flame's visible timeline to its measured
  // 0-based OFX time. Exercise the real instance-changed action rather than only setting
  // refFrame directly in the harness.
  PropertySet changedArgs;
  hostSetString(&changedArgs, kOfxPropChangeReason, kOfxChangeUserEdited);
  hostSetDouble(&changedArgs, kOfxPropTime, 5.0);
  hostSetDoubles(&changedArgs, kOfxImageEffectPropRenderScale, {1.0, 1.0});
  hostSetString(&changedArgs, kOfxPropType, kOfxTypeParameter);
  hostSetString(&changedArgs, kOfxPropName, "setRef");
  expectStatus("Set Ref changedParam (first press)", plugin->mainEntry(
                                           kOfxActionInstanceChanged,
                                           reinterpret_cast<OfxImageEffectHandle>(&instance),
                                           reinterpret_cast<OfxPropertySetHandle>(&changedArgs),
                                           nullptr));
  hostSetDouble(&changedArgs, kOfxPropTime, 7.0);
  expectStatus("Set Ref changedParam (second press)", plugin->mainEntry(
                                           kOfxActionInstanceChanged,
                                           reinterpret_cast<OfxImageEffectHandle>(&instance),
                                           reinterpret_cast<OfxPropertySetHandle>(&changedArgs),
                                           nullptr));
  Parameter *referenceFrame = instance.params.find("refFrame");
  check(referenceFrame != nullptr && referenceFrame->intValue == 7,
        "Set Ref replaces the scalar with the current 0-based OFX time");
  check(referenceFrame != nullptr && referenceFrame->valueSetCount == 2 &&
            referenceFrame->valueSetAtTimeCount == 0,
        "Set Ref never creates timeline keys");

  // Each format uses the same host instance with a fresh clip configuration.  This is
  // intentionally a fallback case: the disconnected Insert must make Composite an exact
  // Source copy even before an estimator or model has been configured.
  for (const std::string &depth : depths) {
    for (const std::string &component : components) {
      RenderState state;
      state.width = 12;
      state.height = 8;
      state.depth = depth;
      state.components = component;
      state.renderTime = 2.0;
      state.renderWindow = {0, 0, state.width, state.height};
      configureInstance(&instance, state, false);
      verifyCheapQueries(plugin, &instance, state.renderTime);
      RenderResult result = renderOne(plugin, &instance, state, false, "Composite", "Current");
      check(result.statusOK, "Track Composite fallback render " + depth + "/" + component);
      check(result.output.depth == depth,
            "Track output preserves " + depth + " depth under the host's single-depth contract");
      check(result.output.components == component,
            "Track output preserves " + component +
                " components under the host's common-layout contract");
      const Plane expected = expectedFallbackCopy(result.output, state, 17, state.renderTime);
      check(planesEqual(result.output, expected),
            "Composite disconnected Insert is Source bit-for-bit for " + depth + "/" + component);

      RenderState partial = state;
      partial.renderWindow = {2, 1, 9, 6};
      RenderResult partialResult = renderOne(plugin, &instance, partial, false, "Composite", "Current");
      check(partialResult.statusOK, "Track partial-window fallback render " + depth + "/" + component);
      const Plane partialExpected =
          expectedFallbackCopy(partialResult.output, partial, 17, partial.renderTime);
      check(planesEqual(partialResult.output, partialExpected),
            "Composite fallback respects a partial render window for " + depth + "/" + component);
    }
  }

  // Connected Insert sentinels make Current and Reference observable with the identity
  // estimator.  The check is also useful when the model path is unavailable: the documented
  // Warped Insert fallback is still the unwarped selected Insert frame.
  RenderState state;
  state.width = 12;
  state.height = 8;
  state.depth = kOfxBitDepthFloat;
  state.components = kOfxImageComponentRGBA;
  state.renderTime = 2.0;
  state.renderWindow = {0, 0, state.width, state.height};
  configureInstance(&instance, state, true);
  check(setChoiceByLabel(&instance, "insertTime", "Reference"),
        "Insert Reference mode is selectable for ROI");
  gActiveEffect = &instance;
  gRenderState = &state;
  PropertySet referenceRoiArgs;
  hostSetDouble(&referenceRoiArgs, kOfxPropTime, state.renderTime);
  hostSetDoubles(&referenceRoiArgs, kOfxImageEffectPropRenderScale, {1.0, 1.0});
  hostSetDoubles(&referenceRoiArgs, kOfxImageEffectPropRegionOfInterest,
                 {0.0, 0.0, 12.0, 8.0});
  PropertySet referenceRoiOut;
  expectStatus("getRegionsOfInterest (Reference Insert)", plugin->mainEntry(
      kOfxImageEffectActionGetRegionsOfInterest,
      reinterpret_cast<OfxImageEffectHandle>(&instance),
      reinterpret_cast<OfxPropertySetHandle>(&referenceRoiArgs),
      reinterpret_cast<OfxPropertySetHandle>(&referenceRoiOut)));
  const bool queriedInsertAtReference = std::any_of(
      state.rodQueries.begin(), state.rodQueries.end(), [](const Pull &query) {
        return query.clip == "Insert" && query.time == 7.0;
      });
  check(queriedInsertAtReference, "Insert ROI is queried at selected Reference time R");
  gRenderState = nullptr;
  gActiveEffect = nullptr;
  setChoiceByLabel(&instance, "insertTime", "Current");
  RenderResult current = renderOne(plugin, &instance, state, true, "Warped Insert", "Current");
  RenderResult reference = renderOne(plugin, &instance, state, true, "Warped Insert", "Reference");
  check(current.statusOK && reference.statusOK, "Warped Insert connected renders complete");
  const Plane currentExpected = expectedPlaneFor(current.output, 83, 2.0);
  const Plane referenceExpected = expectedPlaneFor(reference.output, 83, 7.0);
  check(!planesEqual(currentExpected, referenceExpected),
        "Insert Current and Reference sentinels are distinct");
  check(planesEqual(current.output, currentExpected),
        "Warped Insert Current pulls Insert at N");
  check(planesEqual(reference.output, referenceExpected),
        "Warped Insert Reference pulls Insert at R");

  RenderResult disconnectedWarped = renderOne(plugin, &instance, state, false, "Warped Insert", "Current");
  check(disconnectedWarped.statusOK, "Warped Insert disconnected fallback render completes");
  check(isBlack(disconnectedWarped.output), "disconnected Insert is transparent black in Warped Insert");

  expectStatus("destroyInstance (Track contract)", plugin->mainEntry(
                                                kOfxActionDestroyInstance,
                                                reinterpret_cast<OfxImageEffectHandle>(&instance),
                                                nullptr, nullptr));
}

float readFloatChannel(const Plane &plane, int x, int y, int channel) {
  const unsigned char *pixel = plane.pixel(x, y);
  float value = 0.0f;
  std::memcpy(&value, pixel + channel * sizeof(float), sizeof(float));
  return value;
}

void testStRenderContract(OfxPlugin *plugin, const Effect &descriptor) {
  Effect instance;
  cloneEffectDescriptor(descriptor, &instance);
  expectStatus("createInstance (ST contract)", plugin->mainEntry(
                                             kOfxActionCreateInstance,
                                             reinterpret_cast<OfxImageEffectHandle>(&instance),
                                             nullptr, nullptr));
  check(setIntParameter(&instance, "refFrame", 2), "ST reference frame can be set");

  RenderState state;
  state.width = 9;
  state.height = 7;
  state.depth = kOfxBitDepthFloat;
  state.components = kOfxImageComponentRGBA;
  state.renderTime = 2.0;
  state.renderWindow = {0, 0, state.width, state.height};

  for (const std::string &origin : {"Bottom Left", "Top Left"}) {
    check(setChoiceByLabel(&instance, "stMode", "Absolute UV"),
          "ST Absolute UV option is selectable");
    check(setChoiceByLabel(&instance, "stOrigin", origin),
          "ST origin " + origin + " is selectable");
    RenderResult result = renderOne(plugin, &instance, state, false, "", "");
    check(result.statusOK, "ST identity render in " + origin + " convention completes");
    check(result.output.depth == kOfxBitDepthFloat,
          "ST identity output remains float in " + origin + " convention");
    const int channels = componentCount(result.output.components);
    check(channels >= 2, "ST output carries U and V channels");
    if (result.statusOK && channels >= 2) {
      bool exact = true;
      for (int y = result.output.bounds.y1; y < result.output.bounds.y2; ++y) {
        for (int x = result.output.bounds.x1; x < result.output.bounds.x2; ++x) {
          const float expectedU = static_cast<float>(
              (static_cast<double>(x - result.output.bounds.x1) + 0.5) /
              static_cast<double>(result.output.width()));
          const double bottomV =
              (static_cast<double>(y - result.output.bounds.y1) + 0.5) /
              static_cast<double>(result.output.height());
          const float expectedV = static_cast<float>(
              origin == "Bottom Left" ? bottomV : 1.0 - bottomV);
          const float actualU = readFloatChannel(result.output, x, y, 0);
          const float actualV = readFloatChannel(result.output, x, y, 1);
          if (actualU != expectedU || actualV != expectedV) exact = false;
        }
      }
      check(exact, "ST identity grid is exact at float for " + origin);
    }
  }

  for (const std::string &origin : {"Bottom Left", "Top Left"}) {
    check(setChoiceByLabel(&instance, "stMode", "Relative Pixels"),
          "ST Relative Pixels option is selectable");
    check(setChoiceByLabel(&instance, "stOrigin", origin),
          "ST origin " + origin + " is selectable for relative zero");
    RenderResult result = renderOne(plugin, &instance, state, false, "", "");
    check(result.statusOK, "ST relative-zero render in " + origin + " convention completes");
    const int channels = componentCount(result.output.components);
    if (result.statusOK && channels >= 2) {
      bool zero = true;
      for (int y = result.output.bounds.y1; y < result.output.bounds.y2; ++y)
        for (int x = result.output.bounds.x1; x < result.output.bounds.x2; ++x)
          zero = zero && readFloatChannel(result.output, x, y, 0) == 0.0f &&
                 readFloatChannel(result.output, x, y, 1) == 0.0f;
      if (!zero)
        info("ST relative first-pixel U/V=" +
             std::to_string(readFloatChannel(result.output, result.output.bounds.x1,
                                              result.output.bounds.y1, 0)) + "/" +
             std::to_string(readFloatChannel(result.output, result.output.bounds.x1,
                                              result.output.bounds.y1, 1)));
      check(zero, "ST relative identity is exact zero for " + origin);
    }
  }

  RenderState partial = state;
  partial.renderWindow = {2, 1, 8, 6};
  setChoiceByLabel(&instance, "stMode", "Absolute UV");
  setChoiceByLabel(&instance, "stOrigin", "Bottom Left");
  RenderResult partialResult = renderOne(plugin, &instance, partial, false, "", "");
  check(partialResult.statusOK, "ST identity render accepts a partial render window");
  bool partialExact = partialResult.statusOK;
  if (partialExact) {
    for (int y = partialResult.output.bounds.y1; y < partialResult.output.bounds.y2; ++y) {
      for (int x = partialResult.output.bounds.x1; x < partialResult.output.bounds.x2; ++x) {
        const float expectedU = (static_cast<float>(x) + 0.5f) /
                                static_cast<float>(state.width);
        const float expectedV = (static_cast<float>(y) + 0.5f) /
                                static_cast<float>(state.height);
        partialExact = partialExact &&
                       readFloatChannel(partialResult.output, x, y, 0) == expectedU &&
                       readFloatChannel(partialResult.output, x, y, 1) == expectedV;
      }
    }
  }
  check(partialExact,
        "partial ST identity uses complete Source geometry rather than window geometry");
  expectStatus("destroyInstance (ST contract)", plugin->mainEntry(
                                             kOfxActionDestroyInstance,
                                             reinterpret_cast<OfxImageEffectHandle>(&instance),
                                             nullptr, nullptr));
}

void validateAndRunDescriptor(DescriptorRun *run, bool descriptorOnly) {
  const bool isTrack = run->identifier == "com.mtifilm.whitewater.opticalflow";
  const bool isSt = run->identifier == "com.mtifilm.whitewater.stmap";
  if (!isTrack && !isSt) {
    fail("unexpected White Water plugin identifier: " + run->identifier);
    return;
  }
  const std::string label = isTrack ? "Track" : "ST Map";
  std::string descriptorLabel;
  check(readString(reinterpret_cast<OfxPropertySetHandle>(&run->descriptor.props),
                   kOfxPropLabel, &descriptorLabel) && !descriptorLabel.empty(),
        label + " descriptor has a label");
  std::string contextLabel;
  check(readString(reinterpret_cast<OfxPropertySetHandle>(&run->context.props), kOfxPropLabel,
                   &contextLabel) && !contextLabel.empty(),
        label + " General context has a label");
  validateClips(run->context, isSt);
  validateEffectProperties(run->context, isSt);
  validateParameterSet(run->context, isSt);

  if (descriptorOnly) return;

  if (isTrack) {
    testTrackRenderContract(run->plugin, run->context);
  } else {
    Effect instance;
    cloneEffectDescriptor(run->context, &instance);
    RenderState queryState;
    queryState.renderTime = 2.0;
    configureInstance(&instance, queryState, false);
    expectStatus("createInstance (ST query contract)", run->plugin->mainEntry(
                                                    kOfxActionCreateInstance,
                                                    reinterpret_cast<OfxImageEffectHandle>(&instance),
                                                    nullptr, nullptr));
    verifyCheapQueries(run->plugin, &instance, queryState.renderTime);
    expectStatus("destroyInstance (ST query contract)", run->plugin->mainEntry(
                                                    kOfxActionDestroyInstance,
                                                    reinterpret_cast<OfxImageEffectHandle>(&instance),
                                                    nullptr, nullptr));
    testStRenderContract(run->plugin, run->context);
  }
}

int runBundle(const std::string &path, const std::string &hostName, bool descriptorOnly) {
  std::filesystem::path requested(path);
  std::filesystem::path binary = requested;
  std::error_code filesystemError;
  if (std::filesystem::is_directory(requested, filesystemError)) {
    std::string stem = requested.filename().string();
    const std::string bundleSuffix = ".ofx.bundle";
    if (stem.size() > bundleSuffix.size() &&
        stem.compare(stem.size() - bundleSuffix.size(), bundleSuffix.size(), bundleSuffix) == 0)
      stem.erase(stem.size() - bundleSuffix.size());
    const std::vector<std::filesystem::path> candidates = {
        requested / "Contents" / "MacOS" / (stem + ".ofx"),
        requested / "Contents" / "Linux-x86-64" / (stem + ".ofx"),
        requested / "Contents" / "Linux-arm-64" / (stem + ".ofx")};
    binary.clear();
    for (const auto &candidate : candidates) {
      if (std::filesystem::is_regular_file(candidate, filesystemError)) {
        binary = candidate;
        break;
      }
    }
    if (binary.empty()) {
      for (std::filesystem::recursive_directory_iterator it(requested, filesystemError), end;
           !filesystemError && it != end; it.increment(filesystemError)) {
        if (it->is_regular_file(filesystemError) && it->path().extension() == ".ofx") {
          binary = it->path();
          break;
        }
      }
    }
    if (binary.empty()) {
      fail("bundle directory contains no loadable .ofx binary: " + requested.string());
      return 1;
    }
  }
  std::fprintf(stdout, "Loading White Water bundle %s\n", binary.string().c_str());
  void *library = dlopen(binary.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    fail(std::string("dlopen failed: ") + dlerror());
    return 1;
  }

  setUpHost(hostName);
  using SetHostFunction = OfxStatus (*)(const OfxHost *);
  using GetCountFunction = int (*)(void);
  using GetPluginFunction = OfxPlugin *(*)(int);
  auto setHost = reinterpret_cast<SetHostFunction>(dlsym(library, "OfxSetHost"));
  auto getCount = reinterpret_cast<GetCountFunction>(dlsym(library, "OfxGetNumberOfPlugins"));
  auto getPlugin = reinterpret_cast<GetPluginFunction>(dlsym(library, "OfxGetPlugin"));
  check(setHost != nullptr, "bundle exports OfxSetHost");
  check(getCount != nullptr, "bundle exports OfxGetNumberOfPlugins");
  check(getPlugin != nullptr, "bundle exports OfxGetPlugin");
  if (setHost == nullptr || getCount == nullptr || getPlugin == nullptr) {
    dlclose(library);
    return 1;
  }
  expectStatus("OfxSetHost", setHost(&gHost));

  const int count = getCount();
  check(count == 2, "bundle enumerates exactly the two permanent White Water descriptors");
  LoadedBundle loaded;
  loaded.handle = library;
  for (int index = 0; index < count; ++index) {
    OfxPlugin *plugin = getPlugin(index);
    if (plugin == nullptr) {
      fail("OfxGetPlugin returned null at index " + std::to_string(index));
      continue;
    }
    if (plugin->pluginApi == nullptr || std::strcmp(plugin->pluginApi, kOfxImageEffectPluginApi) != 0) {
      fail("plugin at index " + std::to_string(index) + " is not an image effect");
      continue;
    }
    check(plugin->pluginIdentifier != nullptr, "enumerated plugin has an identifier");
    if (plugin->pluginIdentifier == nullptr || plugin->mainEntry == nullptr ||
        plugin->setHost == nullptr)
      continue;
    plugin->setHost(&gHost);
    const std::string identifier = plugin->pluginIdentifier;
    std::fprintf(stdout, "\nPlugin %d: %s v%u.%u\n", index, identifier.c_str(),
                 plugin->pluginVersionMajor, plugin->pluginVersionMinor);
    expectStatus(identifier + " load", plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr));

    DescriptorRun run;
    run.plugin = plugin;
    run.identifier = identifier;
    expectRequiredStatus(
        identifier + " describe",
        plugin->mainEntry(kOfxActionDescribe,
                          reinterpret_cast<OfxImageEffectHandle>(&run.descriptor), nullptr,
                          nullptr));
    // Context descriptors inherit the effect-wide properties established by describe.  A
    // real host supplies a fresh handle with those standard properties already materialized;
    // copying them here prevents a missing label/depth list from being mistaken for a plugin
    // omission while keeping the context's clips and parameters independent.
    run.context.props = run.descriptor.props;
    PropertySet contextArgs;
    hostSetString(&contextArgs, kOfxImageEffectPropContext, kOfxImageEffectContextGeneral);
    hostSetString(&run.context.props, kOfxImageEffectPropContext, kOfxImageEffectContextGeneral);
    expectRequiredStatus(
        identifier + " describeInContext(General)",
        plugin->mainEntry(kOfxImageEffectActionDescribeInContext,
                          reinterpret_cast<OfxImageEffectHandle>(&run.context),
                          reinterpret_cast<OfxPropertySetHandle>(&contextArgs), nullptr));
    for (auto &param : run.context.params.params) applyDefault(param.get());
    loaded.descriptors.push_back(std::move(run));
  }

  std::set<std::string> identifiers;
  for (DescriptorRun &run : loaded.descriptors) {
    identifiers.insert(run.identifier);
    validateAndRunDescriptor(&run, descriptorOnly);
  }
  check(identifiers.count("com.mtifilm.whitewater.opticalflow") == 1,
        "Track descriptor identifier is permanent and enumerable");
  check(identifiers.count("com.mtifilm.whitewater.stmap") == 1,
        "ST Map descriptor identifier is permanent and enumerable");

  for (DescriptorRun &run : loaded.descriptors)
    expectStatus(run.identifier + " unload", run.plugin->mainEntry(kOfxActionUnload, nullptr,
                                                                     nullptr, nullptr));
  dlclose(library);
  return gFailures == 0 ? 0 : 1;
}

void usage(const char *program) {
  std::fprintf(stderr,
               "Usage: %s <WhiteWater.ofx> [--host-name NAME] [--descriptor-only]\n"
               "       [--refuse-string-modes] [--cpus N]\n",
               program);
}

}  // namespace

int main(int argc, char **argv) {
  std::string bundlePath;
  std::string hostName = "org.whitewater.testharness";
  bool descriptorOnly = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--host-name" && index + 1 < argc) {
      hostName = argv[++index];
    } else if (argument == "--descriptor-only") {
      descriptorOnly = true;
    } else if (argument == "--refuse-string-modes") {
      gRefuseStringModes = true;
    } else if (argument == "--cpus" && index + 1 < argc) {
      gReportedCPUs = static_cast<unsigned int>(std::max(1, std::atoi(argv[++index])));
    } else if (!argument.empty() && argument[0] != '-') {
      bundlePath = argument;
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if (bundlePath.empty()) {
    usage(argv[0]);
    return 2;
  }
  const int result = runBundle(bundlePath, hostName, descriptorOnly);
  std::fprintf(stdout, "\n%s (%d failure%s)\n", result == 0 ? "PASSED" : "FAILED", gFailures,
               gFailures == 1 ? "" : "s");
  return result;
}
