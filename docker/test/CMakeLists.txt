include(GoogleTest)

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
cmake_minimum_required(VERSION 3.17...3.21 FATAL_ERROR)

if (${CMAKE_VERSION} VERSION_LESS 3.17)
    cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
else ()
    cmake_policy(VERSION 3.17)
endif()
project(GADGETRON LANGUAGES CXX C)

find_program(CCACHE_FOUND ccache)
if (CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif (CCACHE_FOUND)

if (EXISTS $ENV{CMAKE_PREFIX_PATH})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH})
endif ()
if (EXISTS $ENV{ISMRMRD_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{ISMRMRD_HOME})
endif ()
if (EXISTS $ENV{TORCH_HOME})
    list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_HOME})
endif ()
if (CMAKE_PREFIX_PATH)
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
endif ()
message(STATUS "Looking for packages in : ${CMAKE_PREFIX_PATH}")
# ---

# check the compiler version
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if (NOT CMAKE_CXX_COMPILER_VERSION) #Compiler version is not set on Ubuntu 12.02 (gcc 4.6)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CMAKE_CXX_COMPILER_VERSION)
    endif ()
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
        message("CXX COMPILER VERSION: ${CMAKE_CXX_COMPILER_ID} : ${CMAKE_CXX_COMPILER_VERSION}")
        message(FATAL_ERROR "Gadgetron requires GCC version >= 6.0")
    endif ()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
        message(FATAL_ERROR "Gadgetron requires Clang version >= 3.5")
    endif ()
    add_compile_definitions(BOOST_NO_CXX98_FUNCTION_BASE)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 18.0)
        message(FATAL_ERROR "Gadgetron requires MSVC 2013 or above ")
    endif ()
else ()
    message(WARNING "Unsupported compiler!")
endif ()

#VERSIONING AND INSTALL PATHS
set(GADGETRON_VERSION_MAJOR 4)
set(GADGETRON_VERSION_MINOR 7)
set(GADGETRON_VERSION_PATCH 2)
set(GADGETRON_VERSION_STRING ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR}.${GADGETRON_VERSION_PATCH})
set(GADGETRON_SOVERSION ${GADGETRON_VERSION_MAJOR}.${GADGETRON_VERSION_MINOR})
find_package(Git)
if (GIT_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY
            ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE GADGETRON_GIT_SHA1 ERROR_VARIABLE GADGETRON_GIT_STDERR)
    string(STRIP "${GADGETRON_GIT_SHA1}" GADGETRON_GIT_SHA1)
    string(LENGTH "${GADGETRON_GIT_SHA1}" GADGETRON_GIT_SHA1_LEN)
    if (${GADGETRON_GIT_SHA1_LEN} LESS 40)
        message(WARNING "Could not determine SHA-1 hash: ${GADGETRON_GIT_STDERR}")
        set(GADGETRON_GIT_SHA1 "NA")
    endif ()
else ()
    set(GADGETRON_GIT_SHA1 "NA")
endif ()
set(GADGETRON_INSTALL_CMAKE_PATH share/gadgetron/cmake)
set(GADGETRON_INSTALL_CONFIG_PATH share/gadgetron/config)
set(GADGETRON_INSTALL_PYTHON_MODULE_PATH share/gadgetron/python)
set(GADGETRON_INSTALL_SCHEMA_PATH share/gadgetron/schema)
set(GADGETRON_INSTALL_INCLUDE_PATH include/gadgetron)
set(GADGETRON_INSTALL_CHROOT_SCRIPTS_PATH share/gadgetron/chroot)
if (WIN32)
    set(GADGETRON_VAR_DIR $ENV{APPDATA}/gadgetron CACHE PATH "Path where Gadgetron will store calibration data, etc" FORCE)
else ()
    set(GADGETRON_VAR_DIR /var/lib/gadgetron/ CACHE PATH "Path where Gadgetron will store calibration data, etc" FORCE)
endif ()
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


#Set the build type to Release if not specified
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING
            "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
            FORCE)
endif ()

# build options for 64 bits system
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message("64bit system is found")
    set(HAS_64_BIT On CACHE BOOL "64bit build")
else ()
    message("32bit system is found")
    set(HAS_64_BIT Off CACHE BOOL "64bit build")
endif ()


#set(CMAKE_MACOSX_RPATH 1)
set(CMAKE_INSTALL_RPATH ".:${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# whether to install dependencies
option(GADGETRON_INSTALL_DEPENDENCIES "Install gadgetron dependencies" Off)
option(DISABLE_FORK "Disables forking in release mode, causing Gadgetron to always run in a single process" Off)

if (DISABLE_FORK)
    add_definitions(-DGADGETRON_DISABLE_FORK)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(CheckCXXCompilerFlag)

# add_compile_options(-Werror)
check_cxx_compiler_flag(-Werror=return-type COMPILER_SUPPORTS_WRETURNTYPE)
if (COMPILER_SUPPORTS_WRETURNTYPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=return-type")
endif ()
check_cxx_compiler_flag(-Werror=terminate COMPILER_SUPPORTS_WTERMINATE)
if (COMPILER_SUPPORTS_WTERMINATE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=terminate")
endif ()
check_cxx_compiler_flag(-Werror=non-virtual-dtor COMPILER_SUPPORT_WNON_VIRTUAL)
if (COMPILER_SUPPORT_WNON_VIRTUAL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=non-virtual-dtor")
endif ()

check_cxx_compiler_flag(-Werror=format-overflow COMPILER_SUPPORT_WOVERFLOW)
if (COMPILER_SUPPORT_WOVERFLOW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=format-overflow")
endif()

# whether to suppress compilation warnings
option(BUILD_SUPPRESS_WARNINGS "Build package while suppressing warnings" Off)
if (BUILD_SUPPRESS_WARNINGS)
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    else ()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -Wno-deprecated-declarations -Wno-inconsistent-missing-override")
    endif ()
endif ()


set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems

# We actually only use system and thread explicitly, but they require linking in date_time and chrono
find_package(Boost 1.80.0 COMPONENTS coroutine system date_time program_options filesystem timer REQUIRED )
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
if (NOT DEFINED Boost_VERSION_STRING)
    set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()


option(USE_SYCL "Build with SYCL support (Intel DPC++)" ON)

# When USE_SYCL is enabled, automatically enable USE_AMALGAMATED and USE_MKL
if (USE_SYCL)
    set(USE_AMALGAMATED ON CACHE BOOL "Build using amalgamated source files" FORCE)
    message(STATUS "USE_SYCL enabled: auto-enabling USE_AMALGAMATED and USE_MKL")
endif()


if (USE_SYCL)
    # SYCL support using Intel DPC++ or AdaptiveCpp
    if (DEFINED DPCPP_HOME)
        set(SYCL_COMPILER "${DPCPP_HOME}/bin/clang++")
        set(CMAKE_CXX_COMPILER "${DPCPP_HOME}/bin/clang++")
    else ()
        find_package(IntelSYCL QUIET)
        if (IntelSYCL_FOUND)
            message(STATUS "Found Intel DPC++ compiler")
            set(SYCL_COMPILER "dpcpp")
        else()
            message(FATAL_ERROR "USE_SYCL is ON but no SYCL compiler found. "
                        "Install Intel oneAPI DPC++ or AdaptiveCpp.")
            # Try to find icpx (Intel oneAPI 2024+)
            find_program(ICPX_COMPILER icpx)
            if (ICPX_COMPILER)
                message(STATUS "Found Intel icpx compiler: ${ICPX_COMPILER}")
                set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
                set(SYCL_COMPILER "icpx")
            else()
                # Try AdaptiveCpp (formerly hipSYCL)
                find_package(AdaptiveCpp QUIET)
                if (AdaptiveCpp_FOUND)
                    message(STATUS "Found AdaptiveCpp")
                    set(SYCL_COMPILER "acpp")
                else()
                    message(FATAL_ERROR "USE_SYCL is ON but no SYCL compiler found. "
                        "Install Intel oneAPI DPC++ or AdaptiveCpp.")
                endif()
            endif()
        endif()
    endif()

    add_definitions(-DUSE_SYCL)
    set(GADGETRON_SYCL_FOUND_BOOL 1)

    # SYCL compile flags
    if (SYCL_COMPILER STREQUAL "icpx" OR SYCL_COMPILER STREQUAL "dpcpp")
        set(SYCL_FLAGS "-fsycl ")
        # Target both SPIR-V (Intel) and CUDA (NVIDIA) if available
        if (EXISTS "/usr/local/cuda")
            set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 ")
        else()
            set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=spir64")
            message(FATAL_ERROR "/usr/local/cuda doesn't exist.")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")
    elseif(DEFINED DPCPP_HOME)
        set(SYCL_FLAGS "-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 ")
    endif()

    # Find oneDPL for parallel algorithms (Thrust replacement)
    find_package(oneDPL QUIET)
    if (oneDPL_FOUND)
        message(STATUS "Found oneDPL for parallel algorithms")
    endif()

    message(STATUS "SYCL support enabled with compiler: ${SYCL_COMPILER}")
else()
    set(GADGETRON_SYCL_FOUND_BOOL 0)
endif()


find_package(Armadillo REQUIRED)
add_library(armadillo INTERFACE)
set_property(TARGET armadillo PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIR})
add_definitions(-DARMA_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_64BIT_WORD)
install(TARGETS armadillo EXPORT gadgetron-export)

find_package(CURL REQUIRED)
find_package(PugiXML REQUIRED)
find_package(ISMRMRD CONFIG REQUIRED)


find_package(FFTW3 REQUIRED COMPONENTS single double)
add_library(FFTW INTERFACE)
target_link_libraries(FFTW INTERFACE ${FFTW3_LIBRARIES})
set_property(TARGET FFTW PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
install(TARGETS FFTW EXPORT gadgetron-export)

set(GADGETRON_BUILD_RPATH "")

set(is_dpcpp TRUE)

option(USE_MKL OFF)

find_package(CBLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(LAPACKE REQUIRED)
add_library(GTBLAS INTERFACE)
set_property(TARGET GTBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${CBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})

find_package(PkgConfig)
pkg_check_modules(PC_lapacke lapacke)


install(TARGETS GTBLAS EXPORT gadgetron-export)



message("Searching for OpenGL, GLEW, GLUT, and Qt. These libraries are only used in a single standalone application and are thus non-essential.")
if(WIN32)
    message("For Windows users in particular, for ease of installation we do not recommend installing these libraries.")
endif()

find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)
find_package(Qt4 4.6)
find_package(PLplot)

option(USE_AMALGAMATED "Build using amalgamated source files" ON)

if (USE_AMALGAMATED)
   
    # Collect all external dependencies into a single variable for amalgamated builds
    set(GADGETRON_AMALGAMATED_LIBS
        Boost::system
        Boost::filesystem
        Boost::program_options
        Boost::date_time
        Boost::timer
        Boost::coroutine
        ISMRMRD::ISMRMRD
        FFTW
        # GTBLAS
        armadillo
        ${CURL_LIBRARIES}
        ${CMAKE_DL_LIBS}
    )
    if (USE_SYCL)
        include_directories("${DPCPP_HOME}/include" "/oneMKLwithCublas/include" "${DPCPP_HOME}/../_deps/level-zero-loader-src/include" )
        # SYCL runtime is linked via compiler flags, but add oneDPL if available
        if (oneDPL_FOUND)
            # list(APPEND GADGETRON_AMALGAMATED_LIBS oneDPL)
        endif()
        # Add oneMKL for FFT/BLAS if using Intel SYCL
        # find_package(MKL QUIET COMPONENTS sycl)
        # if (MKL_FOUND)
        #     # list(APPEND GADGETRON_AMALGAMATED_LIBS MKL::MKL_DPCPP)
        #     # list(APPEND GADGETRON_AMALGAMATED_LIBS sycl OpenCL stdc++fs  mkl_core "/oneMKLwithCublas/lib/libonemath_blas_cublas.so" "/oneMKLwithCublas/lib/libonemath.so" )
        # endif()
    endif()
endif()

option(BUILD_TESTING "Enable test building" On)
if (BUILD_TESTING)
    enable_testing()
    find_package(GTest)
    if (NOT GTEST_FOUND)
        find_path(GTEST_SRC_DIR src/gtest.cc HINTS /usr/src/gtest)
        find_path(GTEST_INCLUDE_DIRS gtest.h HINTS /usr/include/gtest)
        if (GTEST_SRC_DIR AND GTEST_INCLUDE_DIRS)
            message("GTest src package found. Compiling as part of Gadgetron.")
            add_subdirectory(${GTEST_SRC_DIR} ${CMAKE_BINARY_DIR}/gtest )
            include_directories(${GTEST_INCLUDE_DIRS})
            set(GTEST_FOUND 1)
            set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
        else()
            message(FATAL_ERROR "GTest required for building tests")
        endif ()

    endif ()
    MESSAGE("GTEST FOUND: ${GTEST_FOUND}")
else()
    MESSAGE("Testing not being built")
endif()


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################


# SYCL amalgamated build - use converted .dp.cpp file
set(AMALGAMATED_SOURCE "${CMAKE_SOURCE_DIR}/tst.cpp")
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
    FFTW
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
    
