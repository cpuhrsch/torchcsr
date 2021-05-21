get_property(TORCHCSR_THIRD_PARTIES GLOBAL PROPERTY TORCHCSR_THIRD_PARTIES)

# Based on torchaudio CMake setup

set(
  LIBTORCHCSR_SOURCES
  softmax.cpp
  )

add_library(
  _torchcsr
  SHARED
  py_init.cpp
  ${LIBTORCHCSR_SOURCES}
  )

set_target_properties(_torchcsr PROPERTIES PREFIX "")
target_compile_definitions(
  _torchcsr PRIVATE TORCH_API_INCLUDE_EXTENSION_H
)

if (USE_CUDA)
  target_compile_definitions(_torchcsr PRIVATE USE_CUDA)
endif()

target_include_directories(
  _torchcsr
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${Python_INCLUDE_DIR}
  )

# See https://github.com/pytorch/pytorch/issues/38122
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(
  _torchcsr
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
  ${TORCHCSR_THIRD_PARTIES}
  ${ADDITIONAL_ITEMS}
  )

install(
  TARGETS _torchcsr
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .  # For Windows
  )
