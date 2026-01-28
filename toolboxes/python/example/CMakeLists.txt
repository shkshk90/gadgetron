if (NOT USE_AMALGAMATED)
    include_directories(${CMAKE_SOURCE_DIR}/toolboxes/python ${ISMRMRD_INCLUDE_DIR})

    add_executable(gadgetron_test_python test_python.cpp)
    target_link_libraries(gadgetron_test_python gadgetron_toolbox_python Boost::filesystem)

    add_executable(python_demo demo.cpp)
    target_link_libraries(python_demo gadgetron_toolbox_python gadgetron_toolbox_cpucore gadgetron_toolbox_cpucore_math Boost::filesystem )

    install(TARGETS gadgetron_test_python DESTINATION bin COMPONENT main)
    install(TARGETS python_demo DESTINATION bin COMPONENT main)
else()
    add_executable(gadgetron_test_python test_python_amalgamated.cpp)
    target_link_libraries(gadgetron_test_python ${GADGETRON_AMALGAMATED_LIBS})

    add_executable(python_demo demo_amalgamated.cpp)
    target_link_libraries(python_demo ${GADGETRON_AMALGAMATED_LIBS})

    install(TARGETS gadgetron_test_python DESTINATION bin COMPONENT main)
    install(TARGETS python_demo DESTINATION bin COMPONENT main)
endif()
