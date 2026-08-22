add_library(whitewater_infer STATIC
  PairwiseFlowEstimator.h
  NullPairwiseEstimator.h
  NullPairwiseEstimator.cpp)

add_library(whitewater::infer ALIAS whitewater_infer)
target_include_directories(whitewater_infer PUBLIC "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(whitewater_infer PUBLIC whitewater::core)

set_target_properties(whitewater_infer PROPERTIES
  C_VISIBILITY_PRESET hidden
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
