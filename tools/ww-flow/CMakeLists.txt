add_library(whitewater_pfm STATIC Pfm.h Pfm.cpp)
target_include_directories(whitewater_pfm PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(ww-flow main.cpp)
target_link_libraries(ww-flow PRIVATE whitewater::infer whitewater_pfm)

if(WHITEWATER_BUILD_TESTS)
  add_executable(ww_flow_pfm_tests pfm_tests.cpp)
  target_link_libraries(ww_flow_pfm_tests PRIVATE whitewater_pfm)
  add_test(NAME ww-flow::pfm COMMAND ww_flow_pfm_tests)

  add_executable(ww_flow_cli_tests cli_tests.cpp)
  target_link_libraries(ww_flow_cli_tests PRIVATE whitewater_pfm)
  add_test(NAME ww-flow::cli COMMAND ww_flow_cli_tests "$<TARGET_FILE:ww-flow>")
endif()
