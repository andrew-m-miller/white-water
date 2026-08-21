# The ONNX Runtime isolation probe.
#
# A separate bundle from the capability probe on purpose: a runtime that refuses to
# initialise, or that takes the host down when it tries, must not be able to destroy the
# capability probe's ability to report anything.
#
# NOTE the LIBRARIES list: whitewater::ofx_headers and ${CMAKE_DL_LIBS}, and NOT ONNX
# Runtime. Linking ORT would record a DT_NEEDED that the loader resolves through the global
# scope, binding us to Flame's already-loaded copy before any of our code runs -- which is
# precisely the thing being measured. The runtime is reached only through dlopen/dlsym; the
# ORT include directory below supplies struct layouts and nothing else.

whitewater_add_ofx_bundle(
  NAME WhiteWaterOrtProbe
  SOURCES ortprobe.cpp
  LIBRARIES ${CMAKE_DL_LIBS})

target_include_directories(WhiteWaterOrtProbe SYSTEM PRIVATE "${WHITEWATER_ORT_INCLUDE_DIR}")
# probe_model.inc lives beside the source.
target_include_directories(WhiteWaterOrtProbe PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Stage the runtime into Contents/Libraries, which is where the probe looks for it --
# relative to its own binary, so it works from a bundle carried to an airgapped box with no
# environment set up. WHITEWATER_ORT_LIBRARY overrides the path at runtime for experiments.
set(ort_bundle_libraries
    "${CMAKE_BINARY_DIR}/bundles/WhiteWaterOrtProbe.ofx.bundle/Contents/Libraries")
get_filename_component(ort_library_name "${WHITEWATER_ORT_LIBRARY}" NAME)

if(APPLE)
  set(ort_staged_name "libonnxruntime.dylib")
else()
  set(ort_staged_name "libonnxruntime.so")
endif()

# Copied under a plain, version-free name. The probe opens it by explicit path rather than by
# soname: dlopen("libonnxruntime.so.1") would consult the search path and could hand back the
# host's already-loaded copy without ever touching disk, silently answering a different
# question.
add_custom_command(TARGET WhiteWaterOrtProbe POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${ort_bundle_libraries}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${WHITEWATER_ORT_LIBRARY}" "${ort_bundle_libraries}/${ort_staged_name}"
  COMMENT "Staging ${ort_library_name} into the ORT probe bundle as ${ort_staged_name}"
  VERBATIM)

# A SECOND, byte-identical copy under a different name, on Linux only.
#
# Two modes have to open the runtime under different dlopen flags, and dlopen returns the
# already-loaded library for a repeated path -- RTLD_DEEPBIND is honoured only at initial
# load. Measured 2026-08-20 in Flame: without this, the second mode silently received the
# first mode's library, both reported the identical OrtGetApiBase address, and the verdict
# for the mode still in question was an artifact.
#
# dlclose between modes is not an alternative: it is advisory, and unloading a runtime that
# has registered thread-local state inside a host process is its own hazard. Distinct files
# are unambiguous, and 28 MB of duplication in a diagnostic bundle is a cheap price for a
# measurement that means something.
if(NOT APPLE)
  add_custom_command(TARGET WhiteWaterOrtProbe POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${WHITEWATER_ORT_LIBRARY}" "${ort_bundle_libraries}/libonnxruntime-b.so"
    COMMENT "Staging a second copy as libonnxruntime-b.so (per-mode isolation)"
    VERBATIM)
endif()

# A minimal OFX host that drives the probe outside Flame. It cannot answer the DEEPBIND
# question -- nothing here preloads a competing runtime -- but it does prove the probe
# itself works: actions wired correctly, bundled runtime found by relative path, embedded
# model parsed, ORT call sequence run, arithmetic checked. Without it, the first execution
# of that code would be inside Flame on an airgapped box.
add_executable(ortprobe_driver ortprobe_driver.cpp)
target_link_libraries(ortprobe_driver PRIVATE whitewater::ofx_headers ${CMAKE_DL_LIBS})
add_dependencies(ortprobe_driver WhiteWaterOrtProbe)

add_test(NAME ortprobe::runs
  COMMAND ortprobe_driver
    "$<TARGET_FILE:WhiteWaterOrtProbe>")
