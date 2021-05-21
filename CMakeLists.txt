cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Most of the configurations are taken from PyTorch
# https://github.com/pytorch/pytorch/blob/0c9fb4aff0d60eaadb04e4d5d099fb1e1d5701a9/CMakeLists.txt

project(torchcsr)

# check and set CMAKE_CXX_STANDARD
string(FIND "${CMAKE_CXX_FLAGS}" "-std=c++" env_cxx_standard)
if(env_cxx_standard GREATER -1)
  message(
      WARNING "C++ standard version definition detected in environment variable."
      "PyTorch requires -std=c++14. Please remove -std=c++ settings in your environment.")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_STANDARD 11)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
  enable_language(CUDA)
endif()

find_package(Torch REQUIRED)

# TORCH_CXX_FLAGS contains the same -D_GLIBCXX_USE_CXX11_ABI value as PyTorch
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall ${TORCH_CXX_FLAGS}")

add_subdirectory(third_party/kineto/libkineto)
add_subdirectory(torchcsr/csrc)
