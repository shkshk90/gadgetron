if (NOT USE_AMALGAMATED)
    add_executable(register_CK_3d_gpu register_CK_3d.cpp)

    target_link_libraries(register_CK_3d_gpu
      gadgetron_toolbox_hostutils
      gadgetron_toolbox_gpu
      ${CUDA_LIBRARIES}
      )

    install(TARGETS register_CK_3d_gpu DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(register_CK_3d_gpu register_CK_3d_amalgamated.dp.cpp)
        target_compile_options(register_CK_3d_gpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(register_CK_3d_gpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(register_CK_3d_gpu ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(register_CK_3d_gpu register_CK_3d_amalgamated.cu)
        target_link_libraries(register_CK_3d_gpu ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS register_CK_3d_gpu DESTINATION bin COMPONENT main)
endif()
