include_directories(
    ${CMAKE_BINARY_DIR}/toolboxes/core
)


add_subdirectory(cpu)
if (USE_SYCL)
  add_subdirectory(gpu)
endif()
