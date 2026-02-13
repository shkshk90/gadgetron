include(GoogleTest)

if (NOT USE_AMALGAMATED)
    if (WIN32)
        link_directories(${Boost_LIBRARY_DIRS})
    endif ()
    set(test_src_files
            tests.cpp
            hoNDArray_elemwise_test.cpp
            hoNDArray_blas_test.cpp
            hoNDArray_utils_test.cpp
            hoNDArray_reductions_test.cpp
            read_writer_test.cpp
            hoNDFFT_test.cpp
            hoNFFT_test.cpp
            hoNDWavelet_test.cpp
            curveFitting_test.cpp
            image_morphology_test.cpp
            IsmrmrdContextVariables_test.cpp
            StorageSpaces_test.cpp
            pattern_recognition_test.cpp
            cmr_mapping_test.cpp
            hoNDArray_linalg_test.cpp
            core_test.cpp
            core_primitive_io_test.cpp
            threadpool_test.cpp
            from_string_test.cpp
            hoNDArrayView_test.cpp
            ChannelAlgorithmsTest.cpp
            cmr_strain_test.cpp
            cmr_thickening_test.cpp
            cmr_analytical_strain_test.cpp
            #lapack_test.cpp
            hoSDC_test.cpp
            nhlbi_compression_tests.cpp
            mri_core_stream_test.cpp
            gadgets/setup_gadget.h
            gadgets/AcquisitionAccumulateTrigget_test.cpp
            gadgets/FlagTriggerParsing_test.cpp
            )

    if (PYTHONLIBS_FOUND)
        set(test_src_files ${test_src_files} python_converter_test.cpp)
    endif ()

    if (CUDA_FOUND)
        set(test_src_files ${test_src_files}
                cuNDArray_elemwise_test.cpp
                cuNDArray_operators_test.cpp
                cuNDArray_blas_test.cpp
                cuNDArray_utils_test.cpp
                #        vector_td_test.cpp
                cuVector_td_test_kernels.h
                cuVector_td_test_kernels.cu
                cuNDFFT_test.cpp
                cuSDC_test.cpp
                cuNFFT_test.cpp
                )

        cuda_add_executable(test_all
                ${test_src_files}
                )
    else ()
        add_executable(test_all
                ${test_src_files})
    endif ()

    if (CUDA_FOUND)
        target_link_libraries(test_all
                gadgetron_toolbox_gpu
                )
    endif ()
    target_link_libraries(test_all
            gadgetron_core
            gadgetron_core_readers
            gadgetron_core_writers
            gadgetron_mricore
            gadgetron_toolbox_cpucore
            gadgetron_toolbox_cpucore_math
            gadgetron_toolbox_cpufft
            gadgetron_toolbox_cpunfft
            gadgetron_toolbox_cpudwt
            gadgetron_toolbox_cpu_image
            gadgetron_toolbox_cpucore
            gadgetron_toolbox_log
            gadgetron_toolbox_cpuklt
            gadgetron_toolbox_image_analyze_io
            gadgetron_toolbox_mri_core
            gadgetron_toolbox_cpuoperator
            gadgetron_toolbox_cpu_image
            gadgetron_toolbox_cmr
            gadgetron_toolbox_pr
            gadgetron_toolbox_cpusdc
            ${GTEST_LIBRARIES}
            GTest::gmock
            )

    if (PYTHONLIBS_FOUND)
        target_link_libraries(test_all
                gadgetron_toolbox_python
                python)
    endif ()

    gtest_discover_tests(test_all DISCOVERY_MODE PRE_TEST)

    install(TARGETS test_all DESTINATION bin COMPONENT main)
else()
    # Amalgamated build
    if (USE_SYCL)
        
        # SYCL amalgamated build - use converted .dp.cpp file
        set(AMALGAMATED_SOURCE "${CMAKE_SOURCE_DIR}/sycl_output/test/tests_amalgamated.dp.cpp")
        if (NOT EXISTS ${AMALGAMATED_SOURCE})
            # Fall back to local sycl converted file
            set(AMALGAMATED_SOURCE "tests_amalgamated.dp.cpp")
        endif()
        find_package(CBLAS REQUIRED)
        add_executable(test_all ${AMALGAMATED_SOURCE})
        
        target_include_directories(test_all PUBLIC "/downloads/rt/include" "/dpcpp_home/include" "/opt/intel/oneapi/dpcpp-ct/2025.3/include" "/opt/intel/oneapi/dpl/latest/include" )
        target_compile_options(test_all PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 )
        target_link_options(test_all PRIVATE -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 )
        # target_link_directories(test_all PRIVATE /dpcpp_home/install/lib /oneMKLwithCublas/lib )
        target_link_libraries(test_all 
            Boost::system
            Boost::filesystem
            Boost::program_options
            Boost::date_time
            Boost::timer
            Boost::coroutine
            ISMRMRD::ISMRMRD
            # FFTW
            armadillo
            # ${CURL_LIBRARIES}
            # ${CMAKE_DL_LIBS}
            ${GTEST_LIBRARIES} GTest::gmock
            # ${BOOST_LIBRARIES}
            "/dpcpp_home/install/lib/libsycl.so"
            # "/dpcpp_home/install/lib/libsycl-jit.so"
            # OpenCL 
            stdc++fs
            # oneMath dispatch + NVIDIA backends (cuBLAS, cuFFT, cuSOLVER, cuRAND, cuSPARSE)
            "/oneMKLwithCublas/lib/libonemath.so"
            "/oneMKLwithCublas/lib/libonemath_blas_cublas.so"
            "/oneMKLwithCublas/lib/libonemath_sparse_blas_cusparse.so"
            # MKL CPU-only libraries (host-side math, no mkl_sycl_* GPU offload libs)
            # $<LINK_ONLY:MKL::MKL>
            
            # mkl_core 
            # mkl_intel_ilp64 
            # mkl_gf_ilp64 

            # mkl_tbb_thread
            # "/opt/intel/oneapi/tbb/2022.3/lib/libtbb.so"
            ${CBLAS_LIBRARIES}
        )
        if (oneDPL_FOUND)
            target_link_libraries(test_all oneDPL)
        endif()
        gtest_discover_tests(test_all DISCOVERY_MODE PRE_TEST)
        install(TARGETS test_all DESTINATION bin COMPONENT main)
    elseif (USE_CUDA)
        # CUDA amalgamated build
        add_executable(test_all tests_amalgamated.cu)
        target_link_libraries(test_all ${GADGETRON_AMALGAMATED_LIBS} ${GTEST_LIBRARIES} GTest::gmock)
        gtest_discover_tests(test_all DISCOVERY_MODE PRE_TEST)
        install(TARGETS test_all DESTINATION bin COMPONENT main)
    else()
        # CPU-only amalgamated build
        add_executable(test_all tests_amalgamated.cpp)
        target_link_libraries(test_all ${GADGETRON_AMALGAMATED_LIBS} ${GTEST_LIBRARIES} GTest::gmock)
        gtest_discover_tests(test_all DISCOVERY_MODE PRE_TEST)
        install(TARGETS test_all DESTINATION bin COMPONENT main)
    endif()
endif()

