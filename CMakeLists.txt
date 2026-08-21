cmake_minimum_required(VERSION 3.20)

# These must be set before project(): CMake configures the compiler during project(), and
# setting them afterwards leaves a host-architecture-only build. They are ignored on
# platforms other than macOS, so no guard is needed -- and a guard is not possible here,
# since APPLE is not defined until project() has run.
#
# 12.0 rather than warp-drive's 11.0: ONNX Runtime's CoreML execution provider wants a
# recent CoreML, and every Apple Silicon machine that runs Flame is well past 12.
set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version")
if(NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "macOS build architecture (Apple Silicon only)")
endif()

project(WhiteWater
  VERSION 0.1.0
  DESCRIPTION "Machine-learning optical flow tracking for OpenFX hosts"
  LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Every include in this project is rooted at src/ or at a submodule, so an editor with no
# compilation database sees nothing but "file not found". Generating one costs nothing and
# is the difference between a language server that works and one that reports the whole
# tree as broken.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(WHITEWATER_BUILD_HOSTPROBE "Build the host capability probe plugin" ON)
option(WHITEWATER_BUILD_TESTS "Build the test host harness and tests" ON)

# The production ONNX Runtime backend is off until Phase 3; the Phase 0B ortprobe stages its
# own qualified runtime independently. Everything above src/infer is testable without ORT via
# the synthetic estimator, and keeping the core build free of the GPU payload makes it a build
# people will actually run.
option(WHITEWATER_ENABLE_ONNXRUNTIME "Build the ONNX Runtime inference backend" OFF)

# Where `cmake --install` puts bundles. These are the paths hosts scan by default; the
# OFX_PLUGIN_PATH environment variable overrides them at runtime.
if(APPLE)
  set(WHITEWATER_OFX_INSTALL_DIR "/Library/OFX/Plugins" CACHE STRING "OFX plugin install directory")
else()
  set(WHITEWATER_OFX_INSTALL_DIR "/usr/OFX/Plugins" CACHE STRING "OFX plugin install directory")
endif()

# --- OpenFX headers -----------------------------------------------------------------
# Headers only. The C++ support library is pulled in below for the shipping plugin; the
# probe deliberately uses the raw C API so it has nothing between it and the host.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/openfx/include/ofxCore.h")
  message(FATAL_ERROR
    "third_party/openfx is empty. Run: git submodule update --init --recursive")
endif()

add_library(whitewater_ofx_headers INTERFACE)
add_library(whitewater::ofx_headers ALIAS whitewater_ofx_headers)
target_include_directories(whitewater_ofx_headers INTERFACE
  "${CMAKE_SOURCE_DIR}/third_party/openfx/include")

# --- OpenFX C++ Support library -----------------------------------------------------
# Built as our own target rather than add_subdirectory(third_party/openfx): that project's
# top-level CMakeLists sets global compiler flags, its own macOS architectures and a set of
# GPU render options, none of which belong to this build.
#
# Visibility is deliberately left at the default here. The plugin module compiles its own
# sources with -fvisibility=hidden, but OfxGetNumberOfPlugins, OfxGetPlugin and OfxSetHost
# come out of this archive -- and a symbol hidden by -fvisibility cannot be made global
# again by the version script. The version script is what keeps everything else private.
set(whitewater_ofx_support_dir "${CMAKE_SOURCE_DIR}/third_party/openfx/Support")
add_library(whitewater_ofx_support STATIC
  "${whitewater_ofx_support_dir}/Library/ofxsCore.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsImageEffect.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsInteract.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsLog.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsMultiThread.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsParams.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsProperty.cpp"
  "${whitewater_ofx_support_dir}/Library/ofxsPropertyValidation.cpp")
add_library(whitewater::ofx_support ALIAS whitewater_ofx_support)
target_link_libraries(whitewater_ofx_support PUBLIC whitewater::ofx_headers)
# SYSTEM so the support library's own warnings stay out of our build log without turning
# warnings off for our code.
target_include_directories(whitewater_ofx_support SYSTEM PUBLIC
  "${whitewater_ofx_support_dir}/include")

# --- Platform baseline --------------------------------------------------------------
# Flame runs on a certified Rocky build, so the plugin must not depend on a newer libstdc++
# than the host machine has. Linking the C++ runtime statically is the simplest way to be
# immune to that, and the version script keeps those symbols private.
if(UNIX AND NOT APPLE)
  add_link_options(-static-libstdc++ -static-libgcc)
endif()

include(OfxBundle)

# Before any add_subdirectory: add_test() in a subdirectory is silently ignored unless
# testing was enabled first, which presents as a suite that quietly has fewer tests in it
# than it should -- the failure mode a test suite is least able to report.
enable_testing()

# Phase 0B's model provenance is executable data, not prose. Keep the structural gate
# dependency-free so the normal probe/core build does not need PyTorch or ONNX installed.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME models::searaft_manifest
    COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/models/check_searaft_manifest.py"
      "${CMAKE_SOURCE_DIR}/models/sea-raft-m.json")
else()
  message(STATUS "SEA-RAFT manifest gate skipped (Python 3 not found)")
endif()

add_subdirectory(src/core)
add_subdirectory(src/ofx)

if(WHITEWATER_BUILD_HOSTPROBE)
  add_subdirectory(tools/hostprobe)
endif()

# The ONNX Runtime isolation probe, built only when pointed at an unpacked ORT release.
# Off by default and never a hard error: the capability probe and the core must stay
# buildable on a machine that has never downloaded a 20 MB redistributable.
include(OnnxRuntime)
whitewater_find_onnxruntime()
if(found)
  add_subdirectory(tools/ortprobe)
else()
  message(STATUS "ORT probe       : skipped (set WHITEWATER_ORT_ROOT to build it)")
endif()

# The core boundary is a test rather than a comment, so it fails the build the first time
# somebody reaches for an OFX or ONNX Runtime header from inside the flow algebra.
add_test(NAME core::dependency_boundary
  COMMAND "${CMAKE_COMMAND}"
    "-DCORE_DIR=${CMAKE_SOURCE_DIR}/src/core"
    "-DCORE_CMAKE=${CMAKE_SOURCE_DIR}/src/core/CMakeLists.txt"
    -P "${CMAKE_SOURCE_DIR}/scripts/check-core-dependencies.cmake")

# Not yet configured, in dependency order. Each arrives with the phase that needs it --
# see docs/plan.md.
#
#   src/infer      Phase 3   ONNX Runtime, model registry, selected pairwise estimators
#   tools/ww-flow  Phase 2   offline flow CLI and the golden-test driver
#   tests/         Phase 1   Catch2 suites and the two-clip host harness

message(STATUS "")
message(STATUS "White Water ${PROJECT_VERSION}")
message(STATUS "  build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "  bundles         : ${CMAKE_BINARY_DIR}/bundles")
message(STATUS "  install prefix  : ${WHITEWATER_OFX_INSTALL_DIR}")
message(STATUS "  onnxruntime     : ${WHITEWATER_ENABLE_ONNXRUNTIME}")
message(STATUS "")
