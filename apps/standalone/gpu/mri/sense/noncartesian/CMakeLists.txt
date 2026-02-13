if (NOT USE_AMALGAMATED)
    include_directories(
      ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
      )

    add_executable(sense_cg_generic_2d generic_cg.cpp)

    target_link_libraries(sense_cg_generic_2d gadgetron_toolbox_gpu gadgetron_toolbox_cpucore gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS sense_cg_generic_2d DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(sense_cg_generic_2d generic_cg_amalgamated.dp.cpp)
        target_compile_options(sense_cg_generic_2d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(sense_cg_generic_2d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(sense_cg_generic_2d ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(sense_cg_generic_2d generic_cg_amalgamated.cu)
        target_link_libraries(sense_cg_generic_2d ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS sense_cg_generic_2d DESTINATION bin COMPONENT main)
endif()

add_subdirectory(radial)
