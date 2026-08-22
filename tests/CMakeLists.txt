# Host-free Phase 2 unit coverage and the raw OFX host contract harness.
add_subdirectory(unit)
add_subdirectory(hostharness)

# A module path is supplied by the top-level build once the plugin target exists. Without
# it, registering no host-contract tests would create an incomplete green suite.
if(NOT WHITEWATER_HOST_BINARY)
  message(FATAL_ERROR "WHITEWATER_HOST_BINARY is required for the Phase 1 host harness")
endif()

add_test(NAME hostharness::phase1
  COMMAND whitewater_hostharness "${WHITEWATER_HOST_BINARY}"
          --host-name org.whitewater.testharness)
add_test(NAME hostharness::phase1_flame
  COMMAND whitewater_hostharness "${WHITEWATER_HOST_BINARY}"
          --host-name com.autodesk.flame)
add_test(NAME hostharness::phase1_refuse_string_modes
  COMMAND whitewater_hostharness "${WHITEWATER_HOST_BINARY}"
          --host-name org.whitewater.testharness --refuse-string-modes)
