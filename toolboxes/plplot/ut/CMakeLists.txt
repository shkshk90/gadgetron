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
    add_executable(gadgetron_ut_plplot plplot_ut_amalgamated.cpp)
    target_link_libraries(gadgetron_ut_plplot ${GADGETRON_AMALGAMATED_LIBS})
    install(TARGETS gadgetron_ut_plplot DESTINATION bin COMPONENT main)
endif()

set(GADGETRON_BUILD_RPATH "${CMAKE_CURRENT_BINARY_DIR};${GADGETRON_BUILD_RPATH}" PARENT_SCOPE)