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
    add_executable(server_tests storage_test_amalgamated.cpp)
    target_link_libraries(server_tests ${GADGETRON_AMALGAMATED_LIBS} GTest::GTest GTest::Main GTest::gtest GTest::gtest_main)
endif()

