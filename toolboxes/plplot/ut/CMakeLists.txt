enable_testing()

if(WIN32)
    link_directories(${Boost_LIBRARY_DIRS})
endif()


if (NOT USE_AMALGAMATED)
    add_executable(gadgetron_ut_plplot
        plplot_ut.cpp
        plplot_test.cpp )

    target_link_libraries(gadgetron_ut_plplot gadgetron_toolbox_plplot gadgetron_toolbox_image_analyze_io)
    install(TARGETS gadgetron_ut_plplot DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(gadgetron_ut_plplot plplot_ut_amalgamated.cpp.dp.cpp)
        target_compile_options(gadgetron_ut_plplot PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(gadgetron_ut_plplot PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(gadgetron_ut_plplot ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(gadgetron_ut_plplot plplot_ut_amalgamated.cpp)
        target_link_libraries(gadgetron_ut_plplot ${GADGETRON_AMALGAMATED_LIBS})
    endif()
    install(TARGETS gadgetron_ut_plplot DESTINATION bin COMPONENT main)
endif()

set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)