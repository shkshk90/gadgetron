include_directories(
    ${CMAKE_BINARY_DIR}/toolboxes/core
)


add_subdirectory(cpu)
if (CUDA_FOUND OR USE_SYCL)
  add_subdirectory(gpu)
endif()
