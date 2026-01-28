if (WIN32)
add_definitions(-D_USE_MATH_DEFINES)
endif ()

if (NOT USE_AMALGAMATED)
    include_directories(${CMAKE_SOURCE_DIR}/toolboxes/dwt/gpu)
    add_executable(denoise_TV denoise_TV.cpp)

    target_link_libraries(denoise_TV
            gadgetron_toolbox_gpu
            gadgetron_toolbox_hostutils
            ${CUDA_LIBRARIES})

    install(TARGETS denoise_TV DESTINATION bin COMPONENT main)
else()
    add_executable(denoise_TV denoise_TV_amalgamated.cu)
    target_link_libraries(denoise_TV ${GADGETRON_AMALGAMATED_LIBS})
    install(TARGETS denoise_TV DESTINATION bin COMPONENT main)
endif()
