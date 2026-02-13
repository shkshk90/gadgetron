if (WIN32)
add_definitions(-D_USE_MATH_DEFINES)
endif ()

if (NOT USE_AMALGAMATED)
    include_directories(${CMAKE_SOURCE_DIR}/toolboxes/dwt/gpu)
    add_executable(denoise_TV denoise_TV.cpp)

    target_link_libraries(denoise_TV
            gadgetron_toolbox_gpu
            gadgetron_toolbox_hostutils
            ${CUDA_LIBRARIES})

    install(TARGETS denoise_TV DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(denoise_TV denoise_TV_amalgamated.dp.cpp)
        target_compile_options(denoise_TV PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(denoise_TV PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(denoise_TV ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(denoise_TV denoise_TV_amalgamated.cu)
        target_link_libraries(denoise_TV ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS denoise_TV DESTINATION bin COMPONENT main)
endif()
