# Runtime and file-oriented dependencies are allowed by the infer policy.
find_package(ONNXRuntime REQUIRED)
target_link_libraries(whitewater_infer_fixture PRIVATE whitewater::core)
