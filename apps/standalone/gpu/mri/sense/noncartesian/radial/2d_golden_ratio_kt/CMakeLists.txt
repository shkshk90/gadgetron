if (NOT USE_AMALGAMATED)
    add_executable(ktsense_radial_2d  main.cpp)

    target_link_libraries(ktsense_radial_2d gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS ktsense_radial_2d DESTINATION bin COMPONENT main)
else()
    add_executable(ktsense_radial_2d main_amalgamated.cu)
    target_link_libraries(ktsense_radial_2d ${GADGETRON_AMALGAMATED_LIBS})
    install(TARGETS ktsense_radial_2d DESTINATION bin COMPONENT main)
endif()
