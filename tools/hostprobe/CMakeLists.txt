# The Phase 0 deliverable.
#
# Deliberately dependency-free: raw OFX C headers, no support library, no core, no ONNX
# Runtime. Anything between the probe and the host is something that could itself be the
# reason a measurement came out the way it did -- and the support library in particular
# fetches suites unconditionally at load, so a probe built on it could not report on a
# host that lacks one.
#
# The exception is the Phase 0 inference question (probe item 3), which cannot be answered
# without linking ONNX Runtime. That is built as a *separate* bundle so the capability
# probe stays loadable on a machine where the runtime does not initialise at all.

# ${CMAKE_DL_LIBS} is the one link dependency, and it is not optional on the baseline we
# build against. dlsym/dladdr/dlerror -- which the symbol-capture probe needs -- lived in
# libdl until glibc 2.34 merged it into libc. An EL9 build links them without asking; the
# EL8 build we actually ship fails with "undefined reference to dlerror". This expands to
# "dl" on Linux and to nothing on macOS, so it stays correct on both.
#
# The probe is otherwise still dependency-free: dl is libc's own, not a third-party library.
whitewater_add_ofx_bundle(
  NAME WhiteWaterHostProbe
  SOURCES hostprobe.cpp
  LIBRARIES ${CMAKE_DL_LIBS})
