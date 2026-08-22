cmake_minimum_required(VERSION 3.20)

foreach(required IN ITEMS BOUNDARY_POLICY SOURCE_DIR EXPECTED_FRAGMENT CHECK_SCRIPT)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()

if(BOUNDARY_POLICY STREQUAL "core")
  set(policy_argument "-DCORE_DIR=${SOURCE_DIR}")
elseif(BOUNDARY_POLICY STREQUAL "infer")
  set(policy_argument "-DINFER_DIR=${SOURCE_DIR}")
else()
  message(FATAL_ERROR "Unknown policy: ${BOUNDARY_POLICY}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" "${policy_argument}" -P "${CHECK_SCRIPT}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)

if(result EQUAL 0)
  message(FATAL_ERROR "Dependency boundary unexpectedly accepted ${SOURCE_DIR}")
endif()

set(combined "${output}\n${error}")
string(FIND "${combined}" "${EXPECTED_FRAGMENT}" fragment_offset)
if(fragment_offset EQUAL -1)
  message(FATAL_ERROR
    "Dependency boundary failed for the wrong reason.\n"
    "Expected diagnostic fragment: ${EXPECTED_FRAGMENT}\n"
    "Actual output:\n${combined}")
endif()

message(STATUS "Dependency boundary rejected the fixture with the expected diagnostic")
