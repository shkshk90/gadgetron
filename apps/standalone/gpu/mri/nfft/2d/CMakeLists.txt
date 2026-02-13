include_directories(
  ${CMAKE_SOURCE_DIR}/toolboxes/mri/pmri/gpu
)

if (NOT USE_AMALGAMATED)
    add_executable(nfft_2d_radial main_nfft.cpp)
    add_executable(nffth_2d_radial main_nffth.cpp)
    #add_executable(moco moco.cpp)
    add_executable(nffth_cg_2d_radial main_cg.cpp)
    add_executable(nffth_sb_2d_radial main_sb.cpp)

    target_link_libraries(nfft_2d_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_2d_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    #target_link_libraries(moco gpusolvers gpureg gpucore gpuparallelmri gpuoperators gpunfft hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_cg_2d_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_sb_2d_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS nfft_2d_radial nffth_2d_radial nffth_cg_2d_radial nffth_sb_2d_radial DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(nfft_2d_radial main_nfft_amalgamated.dp.cpp)
        target_compile_options(nfft_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nfft_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nfft_2d_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(nffth_2d_radial main_nffth_amalgamated.dp.cpp)
        target_compile_options(nffth_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nffth_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nffth_2d_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(nffth_cg_2d_radial main_cg_amalgamated.dp.cpp)
        target_compile_options(nffth_cg_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nffth_cg_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nffth_cg_2d_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(nffth_sb_2d_radial main_sb_amalgamated.dp.cpp)
        target_compile_options(nffth_sb_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(nffth_sb_2d_radial PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(nffth_sb_2d_radial ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(nfft_2d_radial main_nfft_amalgamated.cu)
        target_link_libraries(nfft_2d_radial ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(nffth_2d_radial main_nffth_amalgamated.cu)
        target_link_libraries(nffth_2d_radial ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(nffth_cg_2d_radial main_cg_amalgamated.cu)
        target_link_libraries(nffth_cg_2d_radial ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(nffth_sb_2d_radial main_sb_amalgamated.cu)
        target_link_libraries(nffth_sb_2d_radial ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS nfft_2d_radial nffth_2d_radial nffth_cg_2d_radial nffth_sb_2d_radial DESTINATION bin COMPONENT main)
endif()
