cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CORE_DIR OR NOT IS_DIRECTORY "${CORE_DIR}")
  message(FATAL_ERROR "CORE_DIR must name the core source directory")
endif()

file(GLOB_RECURSE core_sources LIST_DIRECTORIES FALSE
  "${CORE_DIR}/*.h" "${CORE_DIR}/*.hpp" "${CORE_DIR}/*.c" "${CORE_DIR}/*.cc"
  "${CORE_DIR}/*.cpp" "${CORE_DIR}/*.cxx")

# infer/ and onnxruntime are white-water additions to the warp-drive original. The core
# holds the flow algebra -- compose, chain, ST map, resample -- and none of it may depend
# on a model actually existing. That is what lets every test above this line run with the
# synthetic estimator, with no weights and no GPU, on any machine.
set(forbidden_include
  "^[ \t]*#[ \t]*include[ \t]*[<\"]([^>\"]*/)?(ofx|infer)/"
  "^[ \t]*#[ \t]*include[ \t]*[<\"](OFX|Qt|Q[A-Z])"
  "^[ \t]*#[ \t]*include[ \t]*[<\"](onnxruntime|core/session|cuda|cudnn)"
  "^[ \t]*#[ \t]*include[ \t]*[<\"](fstream|filesystem|iostream|cstdio|stdio\\.h|unistd\\.h|dirent\\.h|fcntl\\.h|windows\\.h|sys/)"
)

foreach(source IN LISTS core_sources)
  file(STRINGS "${source}" include_lines REGEX "^[ \t]*#[ \t]*include")
  foreach(line IN LISTS include_lines)
    foreach(pattern IN LISTS forbidden_include)
      if(line MATCHES "${pattern}")
        message(FATAL_ERROR "Host, UI, IPC, or I/O dependency in ${source}: ${line}")
      endif()
    endforeach()
  endforeach()

  file(READ "${source}" source_text)
  if(source_text MATCHES "std::(i|o|f)fstream" OR
     source_text MATCHES "std::filesystem" OR
     source_text MATCHES "::(fopen|freopen|open|read|write|mkdir|rmdir|unlink|rename)[ \t\r\n]*\\(")
    message(FATAL_ERROR "I/O side effect in host-free core source: ${source}")
  endif()
endforeach()

if(DEFINED CORE_CMAKE)
  if(NOT EXISTS "${CORE_CMAKE}")
    message(FATAL_ERROR "CORE_CMAKE does not exist: ${CORE_CMAKE}")
  endif()
  file(READ "${CORE_CMAKE}" cmake_text)
  string(REGEX REPLACE "#[^\n\r]*" "" cmake_code "${cmake_text}")
  if(cmake_code MATCHES "whitewater::(ofx|infer)" OR
     cmake_code MATCHES "Qt[0-9]*::" OR
     cmake_code MATCHES "onnxruntime" OR
     cmake_code MATCHES "find_package[ \t]*\\([ \t]*(Qt|onnxruntime|CUDA)")
    message(FATAL_ERROR "Host, UI, or inference build dependency in ${CORE_CMAKE}")
  endif()
endif()

message(STATUS "Core dependency boundary is clean (${CORE_DIR})")
