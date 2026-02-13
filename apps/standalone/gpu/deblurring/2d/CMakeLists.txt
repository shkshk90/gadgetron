if (WIN32)
add_definitions(-D_USE_MATH_DEFINES)
endif ()

if (NOT USE_AMALGAMATED)
    add_executable(blur_2d blur_2d.cpp)
    add_executable(deblur_2d_cg deblur_2d_cg.cpp)
    add_executable(deblur_2d_sb deblur_2d_sb.cpp)

    target_link_libraries(deblur_2d_cg gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(deblur_2d_sb gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(blur_2d gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS blur_2d deblur_2d_cg deblur_2d_sb DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(blur_2d blur_2d_amalgamated.dp.cpp)
        target_compile_options(blur_2d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(blur_2d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(blur_2d ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(deblur_2d_cg deblur_2d_cg_amalgamated.dp.cpp)
        target_compile_options(deblur_2d_cg PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(deblur_2d_cg PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(deblur_2d_cg ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(deblur_2d_sb deblur_2d_sb_amalgamated.dp.cpp)
        target_compile_options(deblur_2d_sb PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(deblur_2d_sb PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(deblur_2d_sb ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(blur_2d blur_2d_amalgamated.cu)
        target_link_libraries(blur_2d ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(deblur_2d_cg deblur_2d_cg_amalgamated.cu)
        target_link_libraries(deblur_2d_cg ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(deblur_2d_sb deblur_2d_sb_amalgamated.cu)
        target_link_libraries(deblur_2d_sb ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS blur_2d deblur_2d_cg deblur_2d_sb DESTINATION bin COMPONENT main)
endif()

