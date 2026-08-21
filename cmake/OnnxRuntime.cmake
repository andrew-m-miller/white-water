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

# Phase 0B inputs. The ONNX artifact and its manifest are generated outside this build and
# remain ignored by git; when the default artifact is present, use it automatically. An empty
# model path is valid while the probe is being built CPU-only, so a fresh checkout without
# exported weights remains buildable.
set(_whitewater_default_sea_raft_model
    "${CMAKE_SOURCE_DIR}/models/sea-raft-m-opset17.onnx")
if(NOT EXISTS "${_whitewater_default_sea_raft_model}")
  set(_whitewater_default_sea_raft_model "")
endif()
set(WHITEWATER_SEA_RAFT_MODEL "${_whitewater_default_sea_raft_model}" CACHE FILEPATH
  "Pinned SEA-RAFT M ONNX artifact for the Phase 0B probe (optional)")
set(WHITEWATER_SEA_RAFT_MANIFEST "${CMAKE_SOURCE_DIR}/models/sea-raft-m.json" CACHE FILEPATH
  "SEA-RAFT M manifest containing the exported ONNX SHA256")

# These are optional overrides. If left empty, whitewater_find_onnxruntime() discovers the
# provider files below WHITEWATER_ORT_ROOT/lib. CPU redistributables do not contain either
# provider, and that is expected: absence must leave the CPU probe usable.
set(WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY "" CACHE FILEPATH
  "Optional path to libonnxruntime_providers_shared for the ORT probe")
set(WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY "" CACHE FILEPATH
  "Optional path to libonnxruntime_providers_cuda for the ORT probe")

function(whitewater_validate_sea_raft_model)
  set(WHITEWATER_SEA_RAFT_MODEL_SHA256 "" PARENT_SCOPE)

  if(NOT WHITEWATER_SEA_RAFT_MODEL)
    message(STATUS
      "SEA-RAFT M model: not configured; ORT probe will remain Add-model-only")
    return()
  endif()

  if(NOT EXISTS "${WHITEWATER_SEA_RAFT_MODEL}")
    message(FATAL_ERROR
      "WHITEWATER_SEA_RAFT_MODEL points to '${WHITEWATER_SEA_RAFT_MODEL}', but that file "
      "does not exist. Export the pinned artifact or clear the cache entry.")
  endif()
  if(NOT EXISTS "${WHITEWATER_SEA_RAFT_MANIFEST}")
    message(FATAL_ERROR
      "SEA-RAFT M model '${WHITEWATER_SEA_RAFT_MODEL}' was supplied, but the manifest "
      "'${WHITEWATER_SEA_RAFT_MANIFEST}' was not found.")
  endif()

  file(READ "${WHITEWATER_SEA_RAFT_MANIFEST}" manifest_json)
  string(JSON expected_sha ERROR_VARIABLE manifest_error GET "${manifest_json}" export sha256)
  if(manifest_error)
    message(FATAL_ERROR
      "Could not read export.sha256 from SEA-RAFT manifest "
      "'${WHITEWATER_SEA_RAFT_MANIFEST}': ${manifest_error}")
  endif()

  file(SHA256 "${WHITEWATER_SEA_RAFT_MODEL}" actual_sha)
  string(TOLOWER "${expected_sha}" expected_sha)
  string(TOLOWER "${actual_sha}" actual_sha)
  if(NOT actual_sha STREQUAL expected_sha)
    message(FATAL_ERROR
      "SEA-RAFT M ONNX SHA256 mismatch for '${WHITEWATER_SEA_RAFT_MODEL}':\n"
      "  manifest: ${expected_sha}\n"
      "  actual:   ${actual_sha}\n"
      "Re-export the pinned artifact or update the manifest only after re-verification.")
  endif()

  set(WHITEWATER_SEA_RAFT_MODEL_SHA256 "${actual_sha}" PARENT_SCOPE)
  message(STATUS "SEA-RAFT M model: ${WHITEWATER_SEA_RAFT_MODEL}")
  message(STATUS "SEA-RAFT M SHA256: ${actual_sha} (manifest verified)")
  message(STATUS "SEA-RAFT M manifest: ${WHITEWATER_SEA_RAFT_MANIFEST}")
endfunction()

function(whitewater_find_optional_provider root stem out_var)
  if(APPLE)
    file(GLOB candidates LIST_DIRECTORIES false "${root}/lib/${stem}*.dylib")
  else()
    file(GLOB candidates LIST_DIRECTORIES false "${root}/lib/${stem}*.so*")
  endif()

  # Prefer a concrete file over a soname symlink, matching the main runtime staging rule.
  set(best "")
  foreach(candidate IN LISTS candidates)
    if(NOT IS_SYMLINK "${candidate}")
      set(best "${candidate}")
      break()
    endif()
  endforeach()
  if(NOT best AND candidates)
    list(GET candidates 0 best)
  endif()
  set(${out_var} "${best}" PARENT_SCOPE)
endfunction()

function(whitewater_find_onnxruntime)
  set(found FALSE PARENT_SCOPE)
  set(WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY "" PARENT_SCOPE)
  set(WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY "" PARENT_SCOPE)

  # Validate the model independently of the runtime lookup. This catches a stale or
  # mismatched payload even when a developer is configuring without an ORT redistributable.
  whitewater_validate_sea_raft_model()

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
  list(FILTER library_candidates EXCLUDE REGEX "onnxruntime_providers_")
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

  if(WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY)
    if(NOT EXISTS "${WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY}")
      message(FATAL_ERROR
        "WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY is set to "
        "'${WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY}', but that file does not exist.")
    endif()
    set(provider_shared "${WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY}")
  else()
    whitewater_find_optional_provider(
      "${WHITEWATER_ORT_ROOT}" "libonnxruntime_providers_shared" provider_shared)
  endif()

  if(WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY)
    if(NOT EXISTS "${WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY}")
      message(FATAL_ERROR
        "WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY is set to "
        "'${WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY}', but that file does not exist.")
    endif()
    set(provider_cuda "${WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY}")
  else()
    whitewater_find_optional_provider(
      "${WHITEWATER_ORT_ROOT}" "libonnxruntime_providers_cuda" provider_cuda)
  endif()

  set(WHITEWATER_ORT_PROVIDER_SHARED_LIBRARY "${provider_shared}" PARENT_SCOPE)
  set(WHITEWATER_ORT_PROVIDER_CUDA_LIBRARY "${provider_cuda}" PARENT_SCOPE)
  set(found TRUE PARENT_SCOPE)

  message(STATUS "ONNX Runtime: headers ${include_dir}")
  message(STATUS "ONNX Runtime: library ${best}")
  if(provider_shared)
    message(STATUS "ONNX Runtime shared provider: ${provider_shared}")
  else()
    message(STATUS
      "ONNX Runtime shared provider: not found; continuing with CPU-only probe")
  endif()
  if(provider_cuda)
    message(STATUS "ONNX Runtime CUDA provider: ${provider_cuda}")
  else()
    message(STATUS
      "ONNX Runtime CUDA provider: not found; CUDA probe staging is disabled")
  endif()
endfunction()
