// A minimal OFX host that drives the ORT probe outside Flame.
//
// The probe's value is that it runs inside a host that already has ONNX Runtime loaded.
// This driver cannot reproduce that -- nothing here preloads a competing runtime -- so it
// does NOT answer the RTLD_DEEPBIND question. What it does answer, on a developer machine
// and in CI, is whether the probe itself works: that the OFX actions are wired correctly,
// that the bundled runtime is found by the relative path, that the embedded model parses,
// and that the whole ORT call sequence runs and produces the right numbers.
//
// Without this, the first time any of that code executed would be inside Flame on an
// airgapped box, where the only diagnostic is a log file. That is a bad place to discover a
// typo in a tensor shape.
//
//   ortprobe_driver <path to WhiteWaterOrtProbe.ofx>
//
// Exits non-zero if the plugin fails to load or an action returns an error. It deliberately
// does not assert on the probe's *verdict*: "no mode is usable" is a legitimate measurement,
// and a driver that failed on it would be asserting the answer rather than checking the
// instrument.

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <dlfcn.h>

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxMessage.h"
#include "ofxParam.h"
#include "ofxProperty.h"

namespace {

// A property set is a bag of typed values keyed by name and index. Nothing here needs to be
// efficient or complete -- it needs to be correct for the handful of properties the probe
// touches, and to fail loudly rather than silently for anything else.
struct PropertySet {
  std::map<std::string, std::vector<std::string>> strings;
  std::map<std::string, std::vector<int>> ints;
  std::map<std::string, std::vector<double>> doubles;
  std::map<std::string, std::vector<void *>> pointers;
};

std::vector<PropertySet *> gAllSets;

PropertySet *newSet() {
  PropertySet *set = new PropertySet();
  gAllSets.push_back(set);
  return set;
}

template <typename T>
void setAt(std::vector<T> &values, int index, const T &value) {
  if (index < 0)
    return;
  if ((int)values.size() <= index)
    values.resize(index + 1);
  values[index] = value;
}

OfxStatus propSetString(OfxPropertySetHandle h, const char *p, int i, const char *v) {
  setAt(((PropertySet *)h)->strings[p], i, std::string(v ? v : ""));
  return kOfxStatOK;
}
OfxStatus propGetString(OfxPropertySetHandle h, const char *p, int i, char **v) {
  PropertySet *s = (PropertySet *)h;
  auto it = s->strings.find(p);
  if (it == s->strings.end() || (int)it->second.size() <= i)
    return kOfxStatErrUnknown;
  *v = const_cast<char *>(it->second[i].c_str());
  return kOfxStatOK;
}
OfxStatus propSetInt(OfxPropertySetHandle h, const char *p, int i, int v) {
  setAt(((PropertySet *)h)->ints[p], i, v);
  return kOfxStatOK;
}
OfxStatus propGetInt(OfxPropertySetHandle h, const char *p, int i, int *v) {
  PropertySet *s = (PropertySet *)h;
  auto it = s->ints.find(p);
  if (it == s->ints.end() || (int)it->second.size() <= i)
    return kOfxStatErrUnknown;
  *v = it->second[i];
  return kOfxStatOK;
}
OfxStatus propSetDouble(OfxPropertySetHandle h, const char *p, int i, double v) {
  setAt(((PropertySet *)h)->doubles[p], i, v);
  return kOfxStatOK;
}
OfxStatus propGetDouble(OfxPropertySetHandle h, const char *p, int i, double *v) {
  PropertySet *s = (PropertySet *)h;
  auto it = s->doubles.find(p);
  if (it == s->doubles.end() || (int)it->second.size() <= i)
    return kOfxStatErrUnknown;
  *v = it->second[i];
  return kOfxStatOK;
}
OfxStatus propSetPointer(OfxPropertySetHandle h, const char *p, int i, void *v) {
  setAt(((PropertySet *)h)->pointers[p], i, v);
  return kOfxStatOK;
}
OfxStatus propGetPointer(OfxPropertySetHandle h, const char *p, int i, void **v) {
  PropertySet *s = (PropertySet *)h;
  auto it = s->pointers.find(p);
  if (it == s->pointers.end() || (int)it->second.size() <= i)
    return kOfxStatErrUnknown;
  *v = it->second[i];
  return kOfxStatOK;
}
OfxStatus propGetDimension(OfxPropertySetHandle h, const char *p, int *n) {
  PropertySet *s = (PropertySet *)h;
  if (s->strings.count(p)) { *n = (int)s->strings[p].size(); return kOfxStatOK; }
  if (s->ints.count(p)) { *n = (int)s->ints[p].size(); return kOfxStatOK; }
  if (s->doubles.count(p)) { *n = (int)s->doubles[p].size(); return kOfxStatOK; }
  if (s->pointers.count(p)) { *n = (int)s->pointers[p].size(); return kOfxStatOK; }
  return kOfxStatErrUnknown;
}
OfxStatus propReset(OfxPropertySetHandle, const char *) { return kOfxStatOK; }

OfxStatus propSetStringN(OfxPropertySetHandle h, const char *p, int n,
                         const char *const *v) {
  for (int i = 0; i < n; ++i) propSetString(h, p, i, v[i]);
  return kOfxStatOK;
}
OfxStatus propSetIntN(OfxPropertySetHandle h, const char *p, int n, const int *v) {
  for (int i = 0; i < n; ++i) propSetInt(h, p, i, v[i]);
  return kOfxStatOK;
}
OfxStatus propSetDoubleN(OfxPropertySetHandle h, const char *p, int n, const double *v) {
  for (int i = 0; i < n; ++i) propSetDouble(h, p, i, v[i]);
  return kOfxStatOK;
}
OfxStatus propSetPointerN(OfxPropertySetHandle h, const char *p, int n, void *const *v) {
  for (int i = 0; i < n; ++i) propSetPointer(h, p, i, v[i]);
  return kOfxStatOK;
}
OfxStatus propGetStringN(OfxPropertySetHandle h, const char *p, int n, char **v) {
  for (int i = 0; i < n; ++i)
    if (propGetString(h, p, i, &v[i]) != kOfxStatOK) return kOfxStatErrUnknown;
  return kOfxStatOK;
}
OfxStatus propGetIntN(OfxPropertySetHandle h, const char *p, int n, int *v) {
  for (int i = 0; i < n; ++i)
    if (propGetInt(h, p, i, &v[i]) != kOfxStatOK) return kOfxStatErrUnknown;
  return kOfxStatOK;
}
OfxStatus propGetDoubleN(OfxPropertySetHandle h, const char *p, int n, double *v) {
  for (int i = 0; i < n; ++i)
    if (propGetDouble(h, p, i, &v[i]) != kOfxStatOK) return kOfxStatErrUnknown;
  return kOfxStatOK;
}
OfxStatus propGetPointerN(OfxPropertySetHandle h, const char *p, int n, void **v) {
  for (int i = 0; i < n; ++i)
    if (propGetPointer(h, p, i, &v[i]) != kOfxStatOK) return kOfxStatErrUnknown;
  return kOfxStatOK;
}

OfxPropertySuiteV1 gPropSuite = {
    propSetPointer, propSetString, propSetDouble, propSetInt,
    propSetPointerN, propSetStringN, propSetDoubleN, propSetIntN,
    propGetPointer, propGetString, propGetDouble, propGetInt,
    propGetPointerN, propGetStringN, propGetDoubleN, propGetIntN,
    propReset, propGetDimension,
};

// --- image effect suite ----------------------------------------------------

PropertySet *gEffectProps = nullptr;
PropertySet *gParamSetProps = nullptr;

OfxStatus getPropertySet(OfxImageEffectHandle, OfxPropertySetHandle *out) {
  *out = (OfxPropertySetHandle)gEffectProps;
  return kOfxStatOK;
}
OfxStatus getParamSet(OfxImageEffectHandle, OfxParamSetHandle *out) {
  *out = (OfxParamSetHandle)gParamSetProps;
  return kOfxStatOK;
}
OfxStatus clipDefine(OfxImageEffectHandle, const char *, OfxPropertySetHandle *out) {
  *out = (OfxPropertySetHandle)newSet();
  return kOfxStatOK;
}
OfxStatus clipGetHandle(OfxImageEffectHandle, const char *, OfxImageClipHandle *out,
                        OfxPropertySetHandle *props) {
  if (out) *out = nullptr;
  if (props) *props = nullptr;
  return kOfxStatErrUnknown;  // no clips in this driver; the probe tolerates it
}
OfxStatus notImplemented() { return kOfxStatErrUnsupported; }

OfxImageEffectSuiteV1 gEffectSuite = {};

// --- parameter suite -------------------------------------------------------

OfxStatus paramDefine(OfxParamSetHandle, const char *, const char *,
                      OfxPropertySetHandle *out) {
  if (out) *out = (OfxPropertySetHandle)newSet();
  return kOfxStatOK;
}

OfxParameterSuiteV1 gParamSuite = {};

// --- message suite ---------------------------------------------------------

OfxStatus message(void *, const char *, const char *, const char *format, ...) {
  (void)format;
  return kOfxStatOK;
}

OfxMessageSuiteV1 gMessageSuite = {message};

// --- host ------------------------------------------------------------------

PropertySet *gHostProps = nullptr;

const void *fetchSuite(OfxPropertySetHandle, const char *name, int version) {
  if (std::strcmp(name, kOfxImageEffectSuite) == 0 && version == 1) return &gEffectSuite;
  if (std::strcmp(name, kOfxPropertySuite) == 0 && version == 1) return &gPropSuite;
  if (std::strcmp(name, kOfxParameterSuite) == 0 && version == 1) return &gParamSuite;
  if (std::strcmp(name, kOfxMessageSuite) == 0 && version == 1) return &gMessageSuite;
  return nullptr;
}

OfxHost gHost = {};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <path to .ofx>\n", argv[0]);
    return 2;
  }

  gEffectSuite.getPropertySet = getPropertySet;
  gEffectSuite.getParamSet = getParamSet;
  gEffectSuite.clipDefine = clipDefine;
  gEffectSuite.clipGetHandle = clipGetHandle;
  gParamSuite.paramDefine = paramDefine;

  gHostProps = newSet();
  gEffectProps = newSet();
  gParamSetProps = newSet();
  gHost.host = (OfxPropertySetHandle)gHostProps;
  gHost.fetchSuite = fetchSuite;

  dlerror();
  void *module = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!module) {
    std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }

  typedef int (*CountFn)(void);
  typedef OfxPlugin *(*GetFn)(int);
  typedef OfxStatus (*SetHostFn)(const OfxHost *);

  CountFn count = (CountFn)dlsym(module, "OfxGetNumberOfPlugins");
  GetFn get = (GetFn)dlsym(module, "OfxGetPlugin");
  SetHostFn setHost = (SetHostFn)dlsym(module, "OfxSetHost");
  if (!count || !get) {
    std::fprintf(stderr, "missing OFX entry points\n");
    return 1;
  }
  if (count() < 1) {
    std::fprintf(stderr, "plugin reports no plugins\n");
    return 1;
  }

  OfxPlugin *plugin = get(0);
  if (!plugin) {
    std::fprintf(stderr, "OfxGetPlugin(0) returned null\n");
    return 1;
  }
  std::printf("driver: plugin %s v%d.%d\n", plugin->pluginIdentifier,
              plugin->pluginVersionMajor, plugin->pluginVersionMinor);

  if (setHost)
    setHost(&gHost);
  plugin->setHost(&gHost);

  OfxStatus st = plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr);
  if (st != kOfxStatOK) {
    std::fprintf(stderr, "load failed: %d\n", st);
    return 1;
  }

  OfxImageEffectHandle effect = (OfxImageEffectHandle)1;
  st = plugin->mainEntry(kOfxActionDescribe, effect, nullptr, nullptr);
  if (st != kOfxStatOK) {
    std::fprintf(stderr, "describe failed: %d\n", st);
    return 1;
  }

  PropertySet *inArgs = newSet();
  propSetString((OfxPropertySetHandle)inArgs, kOfxImageEffectPropContext, 0,
                kOfxImageEffectContextFilter);
  st = plugin->mainEntry(kOfxImageEffectActionDescribeInContext, effect,
                         (OfxPropertySetHandle)inArgs, nullptr);
  if (st != kOfxStatOK) {
    std::fprintf(stderr, "describeInContext failed: %d\n", st);
    return 1;
  }

  // The whole point: fire the button that runs the probe.
  PropertySet *changed = newSet();
  propSetString((OfxPropertySetHandle)changed, kOfxPropName, 0, "runOrtProbe");
  propSetString((OfxPropertySetHandle)changed, kOfxPropChangeReason, 0,
                kOfxChangeUserEdited);
  propSetDouble((OfxPropertySetHandle)changed, kOfxPropTime, 0, 0.0);
  st = plugin->mainEntry(kOfxActionInstanceChanged, effect,
                         (OfxPropertySetHandle)changed, nullptr);
  if (st != kOfxStatOK) {
    std::fprintf(stderr, "instanceChanged failed: %d\n", st);
    return 1;
  }

  std::printf("driver: probe ran to completion\n");
  return 0;
}
