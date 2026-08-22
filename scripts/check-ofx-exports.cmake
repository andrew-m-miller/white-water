cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "BINARY must name an existing OFX module")
endif()

if(NOT DEFINED PLATFORM)
  if(APPLE)
    set(PLATFORM APPLE)
  elseif(UNIX)
    set(PLATFORM UNIX)
  else()
    message(FATAL_ERROR "PLATFORM is required on unsupported host platforms")
  endif()
endif()

find_program(whitewater_nm NAMES nm)
if(NOT whitewater_nm)
  message(FATAL_ERROR "nm is required to inspect OFX exports")
endif()

if(PLATFORM STREQUAL "APPLE")
  execute_process(
    COMMAND "${whitewater_nm}" -gU "${BINARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error)
elseif(PLATFORM STREQUAL "UNIX")
  execute_process(
    COMMAND "${whitewater_nm}" -D --defined-only --extern-only "${BINARY}"
    RESULT_VARIABLE nm_result
    OUTPUT_VARIABLE nm_output
    ERROR_VARIABLE nm_error)
else()
  message(FATAL_ERROR "Unsupported OFX export inspection platform: ${PLATFORM}")
endif()

if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed for ${BINARY}: ${nm_error}")
endif()

set(exported)
string(REPLACE "\n" ";" nm_lines "${nm_output}")
foreach(line IN LISTS nm_lines)
  string(STRIP "${line}" line)
  if(line STREQUAL "")
    continue()
  endif()

  # nm's portable output puts the symbol in the final whitespace-delimited field.
  string(REGEX MATCH "([_A-Za-z][A-Za-z0-9_.$@]*)$" unused "${line}")
  if(NOT CMAKE_MATCH_1 STREQUAL "")
    set(symbol "${CMAKE_MATCH_1}")
    # Mach-O external symbols carry one leading object-file underscore; ELF symbols do not.
    if(PLATFORM STREQUAL "APPLE" AND symbol MATCHES "^_")
      string(SUBSTRING "${symbol}" 1 -1 symbol)
    endif()
    list(APPEND exported "${symbol}")
  endif()
endforeach()

list(REMOVE_DUPLICATES exported)
list(SORT exported)
set(expected OfxGetNumberOfPlugins OfxGetPlugin OfxSetHost)
list(SORT expected)

if(NOT exported STREQUAL expected)
  message(FATAL_ERROR
    "OFX module must export exactly OfxGetNumberOfPlugins, OfxGetPlugin, and OfxSetHost.\n"
    "  module: ${BINARY}\n"
    "  expected: ${expected}\n"
    "  actual:   ${exported}")
endif()

message(STATUS "OFX exports are exactly the three entry points: ${BINARY}")
