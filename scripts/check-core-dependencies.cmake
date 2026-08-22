cmake_minimum_required(VERSION 3.20)

# This script intentionally has two policies. Core is host-free and may not know about
# either boundary or perform I/O. Inference is allowed to use its runtime and file access,
# but may not reach back up into OFX. Keeping the policies separate prevents a useful
# inference implementation from being rejected merely because it includes ONNX Runtime or
# reads a model file.

function(whitewater_collect_sources root out_var)
  file(GLOB_RECURSE sources LIST_DIRECTORIES FALSE
    "${root}/*.h" "${root}/*.hpp" "${root}/*.c" "${root}/*.cc"
    "${root}/*.cpp" "${root}/*.cxx")
  set(${out_var} "${sources}" PARENT_SCOPE)
endfunction()

function(whitewater_check_includes root policy)
  whitewater_collect_sources("${root}" sources)

  if(policy STREQUAL "core")
    set(forbidden_include
      # Host and inference boundaries, including both the C++ support headers (ofxs*) and
      # raw OFX C headers (ofxCore.h, ofxImageEffect.h, ...).
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?ofx[/A-Za-z0-9_.-]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?ofxs[A-Za-z0-9_.-]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?OFX[A-Za-z0-9_.-]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?infer[/A-Za-z0-9_.-]*[>\"]"
      # Runtime/GPU libraries.
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?onnxruntime[^>\"]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"](core/session|cuda|cudnn)[^>\"]*[>\"]"
      # Host/UI and process/file I/O.
      "^[ \t]*#[ \t]*include[ \t]*[<\"](Qt|Q[A-Z])[^>\"]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"](fstream|filesystem|iostream|cstdio|stdio\\.h|unistd\\.h|dirent\\.h|fcntl\\.h|windows\\.h|sys/)[^>\"]*[>\"]")
  elseif(policy STREQUAL "infer")
    # ONNX Runtime and I/O are intentionally absent from this deny-list. The inference
    # layer owns model loading and runtime calls; only the upward OFX dependency is banned.
    set(forbidden_include
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?ofx[/A-Za-z0-9_.-]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?ofxs[A-Za-z0-9_.-]*[>\"]"
      "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?OFX[A-Za-z0-9_.-]*[>\"]")
  else()
    message(FATAL_ERROR "Unknown dependency policy: ${policy}")
  endif()

  foreach(source IN LISTS sources)
    file(STRINGS "${source}" include_lines REGEX "^[ \t]*#[ \t]*include")
    foreach(line IN LISTS include_lines)
      foreach(pattern IN LISTS forbidden_include)
        if(line MATCHES "${pattern}")
          message(FATAL_ERROR
            "${policy} dependency boundary violation in ${source}: ${line}")
        endif()
      endforeach()
    endforeach()

    # I/O is a core policy violation even when a source hides the operation behind a
    # project-local wrapper rather than including one of the standard headers directly.
    if(policy STREQUAL "core")
      file(READ "${source}" source_text)
      if(source_text MATCHES "std::(i|o|f)fstream" OR
         source_text MATCHES "std::filesystem" OR
         source_text MATCHES "::(fopen|freopen|open|read|write|mkdir|rmdir|unlink|rename)[ \t\r\n]*\\(")
        message(FATAL_ERROR "I/O side effect in host-free core source: ${source}")
      endif()
    endif()
  endforeach()
endfunction()

function(whitewater_check_cmake cmake_file policy)
  if(NOT EXISTS "${cmake_file}")
    message(FATAL_ERROR "${policy} CMake file does not exist: ${cmake_file}")
  endif()

  file(READ "${cmake_file}" cmake_text)
  # Comments are not dependencies. Removing them also prevents a stale explanatory note
  # from tripping this gate while preserving the actual CMake code for the checks below.
  string(REGEX REPLACE "#[^\n\r]*" "" cmake_code "${cmake_text}")

  if(policy STREQUAL "core")
    if(cmake_code MATCHES "whitewater::(ofx|infer)" OR
       cmake_code MATCHES "Qt[0-9]*::" OR
       cmake_code MATCHES "onnxruntime" OR
       cmake_code MATCHES "find_package[ \t]*\\([ \t]*(Qt|onnxruntime|CUDA)")
      message(FATAL_ERROR "Host, UI, or inference build dependency in ${cmake_file}")
    endif()
  elseif(policy STREQUAL "infer")
    if(cmake_code MATCHES "whitewater::ofx" OR
       cmake_code MATCHES "find_package[ \t]*\\([ \t]*(OpenFX|OFX)")
      message(FATAL_ERROR "OFX build dependency in inference CMake file: ${cmake_file}")
    endif()
  else()
    message(FATAL_ERROR "Unknown dependency policy: ${policy}")
  endif()
endfunction()

if(DEFINED CORE_DIR)
  if(NOT IS_DIRECTORY "${CORE_DIR}")
    message(FATAL_ERROR "CORE_DIR must name the core source directory")
  endif()
  whitewater_check_includes("${CORE_DIR}" core)
  if(DEFINED CORE_CMAKE)
    whitewater_check_cmake("${CORE_CMAKE}" core)
  endif()
  message(STATUS "Core dependency boundary is clean (${CORE_DIR})")
endif()

if(DEFINED INFER_DIR)
  if(NOT IS_DIRECTORY "${INFER_DIR}")
    message(FATAL_ERROR "INFER_DIR must name the inference source directory")
  endif()
  whitewater_check_includes("${INFER_DIR}" infer)
  if(DEFINED INFER_CMAKE)
    whitewater_check_cmake("${INFER_CMAKE}" infer)
  endif()
  message(STATUS "Inference dependency boundary is clean (${INFER_DIR})")
endif()

if(NOT DEFINED CORE_DIR AND NOT DEFINED INFER_DIR)
  message(FATAL_ERROR "Pass CORE_DIR and/or INFER_DIR to select a dependency policy")
endif()
