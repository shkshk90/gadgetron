if (NOT USE_AMALGAMATED)
    add_executable(nfft_2d_ms_radial nfft_main.cpp)
    add_executable(nffth_2d_ms_radial nffth_main.cpp)
    add_executable(nffth_2d_ms_generic nffth_generic.cpp)

    target_link_libraries(nfft_2d_ms_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_2d_ms_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_2d_ms_generic gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS nfft_2d_ms_radial nffth_2d_ms_radial nffth_2d_ms_generic DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(nfft_2d_ms_radial nfft_main_amalgamated.dp.cpp)
        target_compile_options(nfft_2d_ms_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nfft_2d_ms_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nfft_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(nffth_2d_ms_radial nffth_main_amalgamated.dp.cpp)
        target_compile_options(nffth_2d_ms_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nffth_2d_ms_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nffth_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(nffth_2d_ms_generic nffth_generic_amalgamated.dp.cpp)
        target_compile_options(nffth_2d_ms_generic PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nffth_2d_ms_generic PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nffth_2d_ms_generic ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(nfft_2d_ms_radial nfft_main_amalgamated.cu)
        target_link_libraries(nfft_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(nffth_2d_ms_radial nffth_main_amalgamated.cu)
        target_link_libraries(nffth_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(nffth_2d_ms_generic nffth_generic_amalgamated.cu)
        target_link_libraries(nffth_2d_ms_generic ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS nfft_2d_ms_radial nffth_2d_ms_radial nffth_2d_ms_generic DESTINATION bin COMPONENT main)
endif()
