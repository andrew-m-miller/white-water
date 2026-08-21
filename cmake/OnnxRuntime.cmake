# Locates an ONNX Runtime redistributable for the isolation probe.
#
# Deliberately does NOT create a link target. The probe reaches ONNX Runtime only through
# dlopen/dlsym, because linking it would record a DT_NEEDED that the loader resolves through
# the global scope -- binding the probe to Flame's already-loaded copy before any of our code
# runs, and answering a different question than the one being asked. All this module provides
# is the header directory and the path to the shared library, which gets copied into the
# bundle's Contents/Libraries.
#
# Point WHITEWATER_ORT_ROOT at an unpacked release, e.g.
#   https://github.com/microsoft/onnxruntime/releases -> onnxruntime-linux-x64-<ver>.tgz
#
# CPU build, not GPU -- because the CPU tarball is ~20 MB against ~3 GB, and the symbol-binding
# question it answers is a real one. It is NOT evidence about the CUDA provider. An earlier
# version of this comment claimed that isolating the CPU runtime isolates the CUDA one; that
# does not follow. The CUDA EP is a separate .so pulling in its own dependency closure, against
# a host that exposes libcudart, libcudnn, libcublas and libnvinfer globally. Measuring it is
# Phase 0B, and it is the last thing gating the inference design.
#
# Measured 2026-08-20: plain RTLD_LOCAL sufficed for the CPU runtime in Flame -- RTLD_DEEPBIND
# also worked but is not used, since it breaks malloc interposition and exception unwinding.
# See docs/host-notes.md.
#
# VERSION MATTERS. Flame 2026.2 carries ONNX Runtime 1.22.0. Deliberately use a *different*
# version here, so the version string the probe reports discriminates between our copy and
# the host's. If both are 1.22.0 the whole measurement is ambiguous and worth nothing.

set(WHITEWATER_ORT_ROOT "" CACHE PATH
  "Root of an unpacked ONNX Runtime release (contains include/ and lib/)")

function(whitewater_find_onnxruntime)
  set(found FALSE PARENT_SCOPE)

  if(NOT WHITEWATER_ORT_ROOT)
    message(STATUS "ONNX Runtime: WHITEWATER_ORT_ROOT not set; ORT probe will not be built")
    return()
  endif()

  set(header "${WHITEWATER_ORT_ROOT}/include/onnxruntime_c_api.h")
  if(NOT EXISTS "${header}")
    # Some packagings nest the headers a level deeper.
    file(GLOB_RECURSE header_candidates "${WHITEWATER_ORT_ROOT}/*/onnxruntime_c_api.h")
    if(header_candidates)
      list(GET header_candidates 0 header)
    endif()
  endif()
  if(NOT EXISTS "${header}")
    message(FATAL_ERROR
      "WHITEWATER_ORT_ROOT is set to '${WHITEWATER_ORT_ROOT}' but onnxruntime_c_api.h was "
      "not found under it. Expected <root>/include/onnxruntime_c_api.h.")
  endif()
  get_filename_component(include_dir "${header}" DIRECTORY)

  if(APPLE)
    set(library_glob "${WHITEWATER_ORT_ROOT}/lib/libonnxruntime*.dylib")
  else()
    set(library_glob "${WHITEWATER_ORT_ROOT}/lib/libonnxruntime.so*")
  endif()
  file(GLOB library_candidates "${library_glob}")
  if(NOT library_candidates)
    message(FATAL_ERROR
      "No ONNX Runtime shared library found matching ${library_glob}. "
      "Unpack a release tarball and point WHITEWATER_ORT_ROOT at it.")
  endif()

  # Prefer the real file over the soname symlinks: the bundle should carry one concrete
  # library, and a dangling symlink inside a .ofx.bundle is a support call.
  set(best "")
  foreach(candidate IN LISTS library_candidates)
    if(NOT IS_SYMLINK "${candidate}")
      set(best "${candidate}")
      break()
    endif()
  endforeach()
  if(NOT best)
    list(GET library_candidates 0 best)
  endif()

  set(WHITEWATER_ORT_INCLUDE_DIR "${include_dir}" PARENT_SCOPE)
  set(WHITEWATER_ORT_LIBRARY "${best}" PARENT_SCOPE)
  set(found TRUE PARENT_SCOPE)

  message(STATUS "ONNX Runtime: headers ${include_dir}")
  message(STATUS "ONNX Runtime: library ${best}")
endfunction()
