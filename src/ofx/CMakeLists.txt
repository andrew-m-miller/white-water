# The host boundary.
#
# A static library rather than part of the plugin module, for the reason warp-drive found:
# a MODULE library cannot be linked into anything else, so everything worth testing
# off-host -- the pixel format conversions, the host quirks, the frame capture -- would be
# untestable if it lived in the module. The module will contain exactly one translation
# unit, Plugin.cpp, whose reference to the factory is what pulls this library in.
#
add_library(whitewater_ofx STATIC
  HostQuirks.h
  HostQuirks.cpp
  PixelFormat.h
  PixelFormat.cpp
  HostImage.h
  HostImage.cpp
  FrameCapture.h
  FrameCapture.cpp
  FlowParameters.h
  FlowParameters.cpp
  OpticalFlowPlugin.h
  OpticalFlowPlugin.cpp)

add_library(whitewater::ofx ALIAS whitewater_ofx)

target_include_directories(whitewater_ofx PUBLIC "${CMAKE_SOURCE_DIR}/src")

target_link_libraries(whitewater_ofx PUBLIC
  whitewater::core
  whitewater::ofx_support
  ${CMAKE_DL_LIBS})

# Our own code is hidden; the support library's is not, because the three OFX entry points
# come out of it and a symbol hidden at compile time cannot be re-exported by the version
# script. See the note in the top-level CMakeLists.
set_target_properties(whitewater_ofx PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

# Keep exactly one translation unit in the module.  The implementation above remains in the
# static library so its pixel adapters and query actions can be linked by host-free tests;
# Plugin.cpp only enumerates the two factories and pulls this library into the bundle.
whitewater_add_ofx_bundle(
  NAME WhiteWater
  SOURCES Plugin.cpp
  LIBRARIES whitewater::ofx)
