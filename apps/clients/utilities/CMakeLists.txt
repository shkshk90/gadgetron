set(Boost_NO_BOOST_CMAKE ON)

if(WIN32)
  find_package(Boost COMPONENTS thread system date_time chrono REQUIRED)
else()
  find_package(Boost COMPONENTS thread system REQUIRED)
endif()

if(WIN32)
  link_directories(${Boost_LIBRARY_DIRS})
endif()


if (NOT USE_AMALGAMATED)
    add_executable(gt_alive gt_alive.cpp)
    add_executable(gtdependencyquery gt_query.cpp DependencyQueryReader.h)

    target_link_libraries(gt_alive gadgetron_toolbox_cpucore
                                   gadgetron_toolbox_gadgettools
                                   gadgetron_toolbox_log)

    target_link_libraries(gtdependencyquery gadgetron_toolbox_cpucore
                                            gadgetron_toolbox_gadgettools
                                            gadgetron_toolbox_log
                                            boost
                                             )

    install(TARGETS gt_alive gtdependencyquery DESTINATION bin COMPONENT main)
    install(FILES DependencyQueryReader.h DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)
else()
    if (USE_SYCL)
        add_executable(gt_alive gt_alive_amalgamated.cpp.dp.cpp)
        target_compile_options(gt_alive PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(gt_alive PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(gt_alive ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)

        add_executable(gtdependencyquery gt_query_amalgamated.cpp.dp.cpp)
        target_compile_options(gtdependencyquery PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_options(gtdependencyquery PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        target_link_libraries(gtdependencyquery ${GADGETRON_AMALGAMATED_LIBS} sycl OpenCL stdc++fs mkl_sycl mkl_core mkl_sycl_blas mkl_sycl_sparse)
    else()
        add_executable(gt_alive gt_alive_amalgamated.cpp)
        target_link_libraries(gt_alive ${GADGETRON_AMALGAMATED_LIBS})

        add_executable(gtdependencyquery gt_query_amalgamated.cpp)
        target_link_libraries(gtdependencyquery ${GADGETRON_AMALGAMATED_LIBS})
    endif()

    install(TARGETS gt_alive gtdependencyquery DESTINATION bin COMPONENT main)
endif()
