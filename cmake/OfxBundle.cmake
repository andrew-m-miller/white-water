# Builds an OFX plugin as a correctly laid out .ofx.bundle.
#
# Adapted from warp-drive's cmake/OfxBundle.cmake, which is the version that has actually
# been loaded by Flame. The layout is dictated by the OpenFX specification and hosts are
# unforgiving about it -- a plugin in the wrong architecture directory is simply invisible,
# with no error message anywhere. Flame additionally only reads PNG icons out of
# Contents/Resources.
#
#   <Name>.ofx.bundle/
#     Contents/
#       Info.plist            (macOS only)
#       Linux-x86-64/<Name>.ofx
#       MacOS/<Name>.ofx
#       Resources/            (PNG icons only)
#       Resources/models/     (*.onnx -- see MODELS)
#       Libraries/            (private ONNX Runtime and its providers)
#
# Usage:
#   whitewater_add_ofx_bundle(NAME WhiteWaterHostProbe SOURCES hostprobe.cpp)
#   whitewater_add_ofx_bundle(NAME WhiteWater SOURCES Plugin.cpp
#                             LIBRARIES whitewater::ofx RUNTIME MODELS)

include(CMakeParseArguments)

function(whitewater_add_ofx_bundle)
  cmake_parse_arguments(ARG "RUNTIME;MODELS" "NAME" "SOURCES;RESOURCES;LIBRARIES" ${ARGN})

  if(NOT ARG_NAME)
    message(FATAL_ERROR "whitewater_add_ofx_bundle: NAME is required")
  endif()

  # The architecture directory name is part of the OFX spec, not a CMake convention.
  if(APPLE)
    set(ofx_arch_dir "MacOS")
  elseif(UNIX)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
      set(ofx_arch_dir "Linux-x86-64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      set(ofx_arch_dir "Linux-arm-64")
    else()
      message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
  else()
    message(FATAL_ERROR "Unsupported platform for OFX bundles")
  endif()

  set(bundle_root "${CMAKE_BINARY_DIR}/bundles/${ARG_NAME}.ofx.bundle")
  set(binary_dir "${bundle_root}/Contents/${ofx_arch_dir}")

  add_library(${ARG_NAME} MODULE ${ARG_SOURCES})

  target_link_libraries(${ARG_NAME} PRIVATE whitewater::ofx_headers ${ARG_LIBRARIES})

  set_target_properties(${ARG_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".ofx"
    LIBRARY_OUTPUT_DIRECTORY "${binary_dir}"
    # Keep the symbol table down to the three entry points the host looks up; anything else
    # exported is a chance to collide with the host's own symbols. See cmake/ofx.map for
    # why this matters especially once ONNX Runtime is linked in.
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

  # Multi-config generators append a per-config subdirectory unless told otherwise.
  foreach(config IN ITEMS DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(${ARG_NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_${config} "${binary_dir}")
  endforeach()

  if(APPLE)
    set(OFX_BUNDLE_NAME "${ARG_NAME}")
    configure_file("${CMAKE_SOURCE_DIR}/cmake/Info.plist.in"
                   "${bundle_root}/Contents/Info.plist" @ONLY)
    # Resolve any bundled dependency relative to the plugin itself, never the host's
    # library path -- Flame ships its own Qt, its own C++ runtime, and quite possibly its
    # own copy of something ONNX Runtime also carries.
    set_target_properties(${ARG_NAME} PROPERTIES
      INSTALL_RPATH "@loader_path/../Libraries"
      BUILD_WITH_INSTALL_RPATH ON)
  elseif(UNIX)
    set_target_properties(${ARG_NAME} PROPERTIES
      INSTALL_RPATH "$ORIGIN/../Libraries"
      BUILD_WITH_INSTALL_RPATH ON)
    # A plugin loaded into a host must not export or interpose symbols; a version script
    # keeps the three OFX entry points visible and hides the rest, including any statically
    # linked third-party code.
    target_link_options(${ARG_NAME} PRIVATE
      "-Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/ofx.map"
      "-Wl,--no-undefined")
  endif()

  foreach(resource IN LISTS ARG_RESOURCES)
    get_filename_component(resource_name "${resource}" NAME)
    add_custom_command(TARGET ${ARG_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${resource}" "${bundle_root}/Contents/Resources/${resource_name}")
  endforeach()

  # RUNTIME and MODELS are requests recorded as global properties rather than acted on
  # here, because the ONNX Runtime import target and the staged model directory are both
  # configured after this function runs. Keeping them as properties means the bundle
  # declares what it needs and the later stage finds it, in the same shape warp-drive uses
  # for its editor payload.
  if(ARG_RUNTIME)
    set_property(GLOBAL APPEND PROPERTY WHITEWATER_RUNTIME_BUNDLES "${bundle_root}")
    set_property(GLOBAL APPEND PROPERTY WHITEWATER_RUNTIME_BUNDLE_TARGETS "${ARG_NAME}")
  endif()
  if(ARG_MODELS)
    set_property(GLOBAL APPEND PROPERTY WHITEWATER_MODEL_BUNDLES "${bundle_root}")
    set_property(GLOBAL APPEND PROPERTY WHITEWATER_MODEL_BUNDLE_TARGETS "${ARG_NAME}")
  endif()

  install(DIRECTORY "${bundle_root}" DESTINATION "${WHITEWATER_OFX_INSTALL_DIR}")

  set_property(GLOBAL APPEND PROPERTY WHITEWATER_BUNDLES "${bundle_root}")
endfunction()
