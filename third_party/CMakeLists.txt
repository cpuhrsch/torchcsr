set(TORCHCSR_THIRD_PARTIES "")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")

add_library(libsputnik INTERFACE)
set(CUDA_ARCHS "70;75")
set(BUILD_TEST OFF)
set(DBUILD_BENCHMARK OFF)
add_subdirectory(sputnik)
message(STATUS "ASDF=${CMAKE_CURRENT_SOURCE_DIR}")
# find_dependency(sputnik REQUIRED)
target_include_directories(libsputnik INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/sputnik")

list(APPEND TORCHCSR_THIRD_PARTIES libsputnik)

set_property(GLOBAL PROPERTY TORCHCSR_THIRD_PARTIES "${TORCHCSR_THIRD_PARTIES}")

