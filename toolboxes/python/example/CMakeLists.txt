if (NOT USE_AMALGAMATED)
    include_directories(${CMAKE_SOURCE_DIR}/toolboxes/python ${ISMRMRD_INCLUDE_DIR})

    add_executable(gadgetron_test_python test_python.cpp)
    target_link_libraries(gadgetron_test_python gadgetron_toolbox_python Boost::filesystem)

    add_executable(python_demo demo.cpp)
    target_link_libraries(python_demo gadgetron_toolbox_python gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math Boost::filesystem )

    install(TARGETS gadgetron_test_python DESTINATION bin COMPONENT main)
    install(TARGETS python_demo DESTINATION bin COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(gadgetron_test_python test_python_amalgamated.cpp.dp.cpp)
        target_compile_options(gadgetron_test_python PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(gadgetron_test_python PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(gadgetron_test_python ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(python_demo demo_amalgamated.cpp.dp.cpp)
        target_compile_options(python_demo PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(python_demo PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(python_demo ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(gadgetron_test_python test_python_amalgamated.cpp)
        target_link_libraries(gadgetron_test_python ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(python_demo demo_amalgamated.cpp)
        target_link_libraries(python_demo ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS gadgetron_test_python DESTINATION bin COMPONENT main)
    install(TARGETS python_demo DESTINATION bin COMPONENT main)
endif()
