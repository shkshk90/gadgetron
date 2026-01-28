if (NOT USE_AMALGAMATED)
    add_executable(nfft_2d_ms_radial nfft_main.cpp)
    add_executable(nffth_2d_ms_radial nffth_main.cpp)
    add_executable(nffth_2d_ms_generic nffth_generic.cpp)

    target_link_libraries(nfft_2d_ms_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_2d_ms_radial gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})
    target_link_libraries(nffth_2d_ms_generic gadgetron_toolbox_gpu gadgetron_toolbox_hostutils ${CUDA_LIBRARIES})

    install(TARGETS nfft_2d_ms_radial nffth_2d_ms_radial nffth_2d_ms_generic DESTINATION bin COMPONENT main)
else()
    add_executable(nfft_2d_ms_radial nfft_main_amalgamated.cu)
    target_link_libraries(nfft_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS})

    add_executable(nffth_2d_ms_radial nffth_main_amalgamated.cu)
    target_link_libraries(nffth_2d_ms_radial ${GADGETRON_AMALGAMATED_LIBS})

    add_executable(nffth_2d_ms_generic nffth_generic_amalgamated.cu)
    target_link_libraries(nffth_2d_ms_generic ${GADGETRON_AMALGAMATED_LIBS})

    install(TARGETS nfft_2d_ms_radial nffth_2d_ms_radial nffth_2d_ms_generic DESTINATION bin COMPONENT main)
endif()
