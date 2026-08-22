add_executable(whitewater_hostharness hostharness.cpp)

# The harness runs the same real-thread paths a host does.  Do not rely on glibc 2.34 having
# folded libpthread into libc: the EL8 artifact baseline still has a separate pthread library.
find_package(Threads REQUIRED)

target_link_libraries(whitewater_hostharness PRIVATE
  whitewater::ofx_headers
  Threads::Threads
  ${CMAKE_DL_LIBS})

target_compile_features(whitewater_hostharness PRIVATE cxx_std_17)
target_compile_options(whitewater_hostharness PRIVATE -Wall -Wextra -Wpedantic)

set_target_properties(whitewater_hostharness PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
