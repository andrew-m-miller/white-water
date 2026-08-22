function(whitewater_add_unit_test target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE whitewater::core)
  add_test(NAME unit::${target} COMMAND ${target})
endfunction()

whitewater_add_unit_test(flow_algebra_tests flow_algebra_tests.cpp)
whitewater_add_unit_test(flow_cache_tests flow_cache_tests.cpp)
whitewater_add_unit_test(image_pipeline_tests image_pipeline_tests.cpp)

add_executable(infer_tests infer_tests.cpp)
target_link_libraries(infer_tests PRIVATE whitewater::infer)
add_test(NAME unit::infer_tests COMMAND infer_tests)
