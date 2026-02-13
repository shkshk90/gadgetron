find_package(dlib)

find_package(Eigen3)
find_package(Ceres)
include_directories(${EIGEN_INCLUDE_DIR})

if (NOT USE_AMALGAMATED)
    link_libraries(
        gadgetron_toolbox_cpucore
        gadgetron_toolbox_cpucore_math
        gadgetron_toolbox_cpufft
        gadgetron_toolbox_cpunfft
        gadgetron_toolbox_cpudwt
        gadgetron_toolbox_cpu_image
        gadgetron_toolbox_cpucore
        gadgetron_toolbox_log
        gadgetron_toolbox_cpuklt
        gadgetron_toolbox_image_analyze_io
        gadgetron_toolbox_mri_core
        gadgetron_toolbox_cpuoperator
        gadgetron_toolbox_cpu_image
        gadgetron_toolbox_cmr
        gadgetron_toolbox_pr
        ${BOOST_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${ARMADILLO_LIBRARIES}
        ${CERES_LIBRARIES}
        )
    add_executable(benchmark_curvefitting benchmark_curvefitting.cpp)
else()
    if (USE_SYCL)
        add_executable(benchmark_curvefitting benchmark_curvefitting_amalgamated.cpp.dp.cpp)
        target_compile_options(benchmark_curvefitting PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(benchmark_curvefitting PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(benchmark_curvefitting ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(benchmark_curvefitting benchmark_curvefitting_amalgamated.cpp)
        target_link_libraries(benchmark_curvefitting ${GADGETRON_AMALGAMATED_LIBS})
    endif()
endif()