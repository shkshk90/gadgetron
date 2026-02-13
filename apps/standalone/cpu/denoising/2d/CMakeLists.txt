if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif ()



if (NOT USE_AMALGAMATED)
    add_executable(cpu_denoise_TV denoise_TV.cpp)
    target_link_libraries(cpu_denoise_TV
                        gadgetron_toolbox_cpucore
                        gadgetron_toolbox_cpucore_math
                        gadgetron_toolbox_hostutils
                        GTBLAS
                        Boost::system
                        )

    install(TARGETS cpu_denoise_TV DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(cpu_denoise_TV denoise_TV_amalgamated.cpp.dp.cpp)
        target_compile_options(cpu_denoise_TV PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(cpu_denoise_TV PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(cpu_denoise_TV ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(cpu_denoise_TV denoise_TV_amalgamated.cpp)
        target_link_libraries(cpu_denoise_TV ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS cpu_denoise_TV DESTINATION bin COMPONENT main)
endif()
