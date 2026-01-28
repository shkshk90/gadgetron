if (WIN32)
    add_definitions(-D_USE_MATH_DEFINES)
endif ()



if (NOT USE_AMALGAMATED)
    add_executable(cpu_denoise_TV denoise_TV.cpp)
    target_link_libraries(cpu_denoise_TV
                        gadgetron_toolbox_cpucore
                        gadgetron_toolbox_cpucore_math
                        gadgetron_toolbox_hostutils
                        GTBLAS
                        Boost::system
                        )

    install(TARGETS cpu_denoise_TV DESTINATION bin COMPONENT main)
else()
    add_executable(cpu_denoise_TV denoise_TV_amalgamated.cpp)
    target_link_libraries(cpu_denoise_TV ${GADGETRON_AMALGAMATED_LIBS})
    install(TARGETS cpu_denoise_TV DESTINATION bin COMPONENT main)
endif()
