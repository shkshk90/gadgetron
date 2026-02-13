if (NOT USE_AMALGAMATED)
    add_executable(register_CK_3d_cpu register_CK_3d.cpp)

    target_link_libraries(register_CK_3d_cpu
      gadgetron_toolbox_hostutils
      gadgetron_toolbox_cpureg
      gadgetron_toolbox_cpucore
      gadgetron_toolbox_cpucore_math
      armadillo
      )

    install(TARGETS register_CK_3d_cpu DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(register_CK_3d_cpu register_CK_3d_amalgamated.cpp.dp.cpp)
        target_compile_options(register_CK_3d_cpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(register_CK_3d_cpu PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(register_CK_3d_cpu ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(register_CK_3d_cpu register_CK_3d_amalgamated.cpp)
        target_link_libraries(register_CK_3d_cpu ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS register_CK_3d_cpu DESTINATION bin COMPONENT main)
endif()
