include_directories(
	${cmake_source_dir}/toolboxes/core
	${cmake_source_dir}/toolboxes/core/cpu
	${cmake_source_dir}/toolboxes/core/cpu/math
	${cmake_source_dir}/toolboxes/core/cpu/hostutils
	${cmake_source_dir}/toolboxes/nfft/cpu
	${cmake_source_dir}/toolboxes/solvers
	${cmake_source_dir}/toolboxes/solvers/cpu
)

if (NOT USE_AMALGAMATED)
    add_executable(nfft nfft.cpp)

    target_link_libraries(nfft
        gadgetron_toolbox_cpucore
        gadgetron_toolbox_cpucore_math
        gadgetron_toolbox_hostutils
        gadgetron_toolbox_cpunfft
    )

    install(TARGETS nfft DESTINATION bin COMPONENT main)
else()
    add_executable(nfft nfft_amalgamated.cpp)
    target_link_libraries(nfft ${GADGETRON_AMALGAMATED_LIBS})
    install(TARGETS nfft DESTINATION bin COMPONENT main)
endif()
