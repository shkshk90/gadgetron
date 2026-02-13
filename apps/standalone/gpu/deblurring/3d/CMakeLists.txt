if (WIN32)
add_definitions(-D_USE_MATH_DEFINES)
endif ()

if (NOT USE_AMALGAMATED)
    add_executable(blur_3d blur_3d.cpp)
    add_executable(deblur_3d_cg deblur_3d_cg.cpp)
    add_executable(deblur_3d_sb deblur_3d_sb.cpp)

    target_link_libraries(deblur_3d_cg gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(deblur_3d_sb gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(blur_3d gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS blur_3d deblur_3d_cg deblur_3d_sb DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(blur_3d blur_3d_amalgamated.dp.cpp)
        target_compile_options(blur_3d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(blur_3d PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(blur_3d ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(deblur_3d_cg deblur_3d_cg_amalgamated.dp.cpp)
        target_compile_options(deblur_3d_cg PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(deblur_3d_cg PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(deblur_3d_cg ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(deblur_3d_sb deblur_3d_sb_amalgamated.dp.cpp)
        target_compile_options(deblur_3d_sb PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(deblur_3d_sb PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(deblur_3d_sb ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(blur_3d blur_3d_amalgamated.cu)
        target_link_libraries(blur_3d ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(deblur_3d_cg deblur_3d_cg_amalgamated.cu)
        target_link_libraries(deblur_3d_cg ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(deblur_3d_sb deblur_3d_sb_amalgamated.cu)
        target_link_libraries(deblur_3d_sb ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS blur_3d deblur_3d_cg deblur_3d_sb DESTINATION bin COMPONENT main)
endif()
