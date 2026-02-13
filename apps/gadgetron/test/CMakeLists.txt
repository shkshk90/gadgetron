enable_testing()

if (NOT USE_AMALGAMATED)
    add_executable(server_tests
            storage_test.cpp
            socket_test.cpp
            ../connection/SocketStreamBuf.cpp)

    add_library(storage OBJECT
            ../storage.cpp)

    target_link_libraries(storage
            gadgetron_core
            ${CURL_LIBRARIES}
            Boost::filesystem
            Boost::program_options)

    target_link_libraries(server_tests
            storage
            gadgetron_core
            gadgetron_toolbox_log
            GTest::GTest
            GTest::Main
            GTest::gtest
            GTest::gtest_main
            )
else()
    if (USE_SYCL)
        add_executable(server_tests storage_test_amalgamated.cpp.dp.cpp)
        target_compile_options(server_tests PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(server_tests PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(server_tests ${GADGETRON_AMALGAMATED_LIBS} GTest::GTest GTest::Main GTest::gtest GTest::gtest_main sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(server_tests storage_test_amalgamated.cpp)
        target_link_libraries(server_tests ${GADGETRON_AMALGAMATED_LIBS} GTest::GTest GTest::Main GTest::gtest GTest::gtest_main)
    endif()
endif()

