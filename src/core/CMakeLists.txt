# The host-free core: the flow algebra and the resampler that consumes it.
#
# No OFX, no ONNX Runtime, no I/O. That boundary is not a convention -- it is checked
# mechanically by scripts/check-core-dependencies.cmake, which runs as a test. It is what
# lets the whole flow pipeline be exercised with the synthetic estimator on a laptop with
# no GPU and no model weights, which is where nearly all of the correctness work happens.

add_library(whitewater_core STATIC
  geom/Vec2.h
  image/Image.h
  image/Image.cpp
  image/OwnedFrame.h
  warp/WarpMap.h
  warp/Resampler.h
  warp/Resampler.cpp
  flow/Field.h
  flow/FlowLink.h
  flow/FlowLink.cpp
  flow/FlowCompose.h
  flow/FlowCompose.cpp
  flow/FlowChain.h
  flow/FlowChain.cpp
  flow/FlowCache.h
  flow/FlowCache.cpp
  flow/FlowWarpMap.h
  flow/Preprocess.h
  flow/Preprocess.cpp
  flow/StMap.h
  flow/StMap.cpp
  flow/Composite.h
  flow/Composite.cpp)

add_library(whitewater::core ALIAS whitewater_core)

# Includes are rooted at src/, so every include reads "core/flow/Field.h" and never a bare
# filename.
target_include_directories(whitewater_core PUBLIC "${CMAKE_SOURCE_DIR}/src")

# std::thread in the resampler's worker pool. glibc 2.34 merged libpthread into libc, but
# naming the dependency explicitly keeps the build honest on any older baseline.
find_package(Threads REQUIRED)
target_link_libraries(whitewater_core PUBLIC Threads::Threads)

set_target_properties(whitewater_core PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
