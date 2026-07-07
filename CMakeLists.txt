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

# create the Gadgetron version header
configure_file("${CMAKE_SOURCE_DIR}/cmake/gadgetron_sha1.h.in" ${PROJECT_BINARY_DIR}/include/gadgetron_sha1.h @ONLY)
include_directories(${PROJECT_BINARY_DIR}/include)
install(FILES ${PROJECT_BINARY_DIR}/include/gadgetron_sha1.h DESTINATION ${GADGETRON_INSTALL_INCLUDE_PATH} COMPONENT main)


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

# SYCL support option (defined early so it can influence OpenMP/MKL)
option(USE_SYCL "Build with SYCL support (replaces CUDA)" Off)

if (USE_SYCL)
    # Force OpenMP, CUDA, and MKL off when using SYCL
    set(USE_OPENMP OFF CACHE BOOL "Use OpenMP" FORCE)
    set(USE_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)
    set(USE_MKL OFF CACHE BOOL "Use MKL" FORCE)

    # Check that compiler supports -fsycl
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-fsycl COMPILER_SUPPORTS_FSYCL)
    if (NOT COMPILER_SUPPORTS_FSYCL)
        message(FATAL_ERROR "USE_SYCL is enabled but the compiler does not support -fsycl. Use a SYCL-capable compiler (e.g. Intel DPC++ / icpx).")
    endif()

    # Which GPU vendor the SYCL build targets. Defaults to cuda so existing
    # NVIDIA configure/build invocations are unaffected.
    set(SYCL_TARGET_BACKEND "cuda" CACHE STRING "SYCL target backend: cuda, intel, or amd")
    set_property(CACHE SYCL_TARGET_BACKEND PROPERTY STRINGS cuda intel amd)
    if (NOT SYCL_TARGET_BACKEND STREQUAL "cuda" AND NOT SYCL_TARGET_BACKEND STREQUAL "intel" AND NOT SYCL_TARGET_BACKEND STREQUAL "amd")
        message(FATAL_ERROR "SYCL_TARGET_BACKEND must be one of: cuda, intel, amd (got '${SYCL_TARGET_BACKEND}')")
    endif()

    # Required SYCL variables
    if (NOT DEFINED DPCT_INCLUDE_PATH)
        message(FATAL_ERROR "USE_SYCL requires DPCT_INCLUDE_PATH to be set (e.g. -DDPCT_INCLUDE_PATH=/path/to/dpct/include)")
    endif()
    if (NOT DEFINED LIBSYCL)
        message(FATAL_ERROR "USE_SYCL requires LIBSYCL to be set (e.g. -DLIBSYCL=/path/to/libsycl.so)")
    endif()
    if (NOT DEFINED ONEMKL_PATH)
        message(FATAL_ERROR "USE_SYCL requires ONEMKL_PATH to be set (e.g. -DONEMKL_PATH=/path/to/onemkl)")
    endif()
    if (NOT DEFINED ZERO_LEVEL_INCLUDE_PATH)
        message(FATAL_ERROR "USE_SYCL requires ZERO_LEVEL_INCLUDE_PATH to be set (e.g. -DZERO_LEVEL_INCLUDE_PATH=/path/to/level-zero/include)")
    endif()
    if (NOT DEFINED ONEAPI_DPL_INCLUDE_PATH)
        message(FATAL_ERROR "USE_SYCL requires ONEAPI_DPL_INCLUDE_PATH to be set (e.g. -DONEAPI_DPL_INCLUDE_PATH=/path/to/oneapi/include)")
    endif()
    if (NOT DEFINED MKL_INCLUDE_PATH)
        message(FATAL_ERROR "USE_SYCL requires MKL_INCLUDE_PATH to be set (e.g. -DMKL_INCLUDE_PATH=/path/to/mkl/include)")
    endif()
    if (NOT DEFINED SYCL_INCLUDE_PATH)
        message(FATAL_ERROR "USE_SYCL requires SYCL_INCLUDE_PATH to be set (e.g. -DSYCL_INCLUDE_PATH=/path/to/sycl/include)")
    endif()

    # SYCL compile and link flags (applied per-target, NOT globally,
    # to avoid SYCL's <cmath> wrapper breaking pure-CPU code).
    # Target triple + backend arch flag depend on SYCL_TARGET_BACKEND. The
    # cuda branch keeps its exact historical token sequence (including the
    # stray trailing comma on the compile-flags triple, absent from the
    # link-flags triple) so default behavior is byte-identical to before
    # this option was introduced.
    set(SYCL_BACKEND_WARNING_FLAGS "")
    set(SYCL_BACKEND_LINK_FLAGS "")
    if (SYCL_TARGET_BACKEND STREQUAL "cuda")
        set(SYCL_COMPILE_TARGET_FLAGS -fsycl-targets=nvptx64-nvidia-cuda, -Xsycl-target-backend --cuda-gpu-arch=sm_80)
        set(SYCL_LINK_TARGET_FLAGS -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80)
    elseif (SYCL_TARGET_BACKEND STREQUAL "intel")
        # spir64 is generic/JIT'd SPIR-V — it doesn't guarantee ahead-of-time
        # codegen for a specific GPU and can fall back to CPU/OpenCL
        # depending on runtime device selection. intel_gpu_pvc is the
        # explicit AOT device alias for Ponte Vecchio (confirmed against
        # intel/llvm's own compiler docs), so device code is compiled
        # directly for the PVC GPU this pipeline targets.
        set(SYCL_COMPILE_TARGET_FLAGS -fsycl-targets=intel_gpu_pvc)
        set(SYCL_LINK_TARGET_FLAGS -fsycl-targets=intel_gpu_pvc)
    elseif (SYCL_TARGET_BACKEND STREQUAL "amd")
        set(SYCL_COMPILE_TARGET_FLAGS -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx942)
        set(SYCL_LINK_TARGET_FLAGS -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx942)
        # oneDPL's SYCL backend requests reqd_sub_group_size(32) (NVIDIA/Intel
        # warp width) as a codegen hint; amdgcn only supports 64-wide
        # wavefronts, so the compiler rejects the attribute as invalid. This
        # is harmless: oneDPL's reduce/scan kernels re-query the actual
        # runtime sub-group width (via
        # __get_reduce_then_scan_actual_sub_group_size(item.get_sub_group()))
        # and use that for all loop bounds/shuffle counts — the attribute is
        # a hint only, guarded by oneDPL's own
        # _ONEDPL_SYCL_REQD_SUB_GROUP_SIZE_IF_SUPPORTED macro for exactly
        # this case. Confirmed by reading oneDPL's
        # parallel_backend_sycl_reduce_then_scan.h source; no correctness
        # impact. Separately, Intel's own from-source SYCL runtime headers
        # (sycl/ext/oneapi/experimental/opportunistic_group.hpp, an unused
        # experimental extension pulled in transitively via <sycl/sycl.hpp>)
        # trip an unrelated missing-return warning under the HIP target.
        #
        # NOTE: our from-source clang++ (built for the HIP backend) segfaults
        # inside LLVM's loop vectorizer (LoopVectorizationPlanner::computeBestVF,
        # "VPlan cost model and legacy cost model disagreed") while generating
        # amdgcn device code for some gadgetron2 GPU kernels (first hit:
        # cuCartesianSenseOperator.sycl.cpp's mult_M kernel). This is a known,
        # recurring class of LLVM vectorizer bug (see llvm/llvm-project issues
        # #100464, #105713, #107171, #107473, #111874, #114860). Four attempts
        # to work around it via flags were tried and reverted per user
        # decision (compile-scope -fno-vectorize/-fno-slp-vectorize, the same
        # at link scope, -fno-sycl-rdc, and lld's -plugin-opt vectorizer
        # flags) — none fully resolved it or they broke something else
        # (-fno-sycl-rdc broke SYCL's own SYCL_EXTERNAL macro visibility).
        # Left un-worked-around for now; AMD build is expected to still hit
        # this crash until a real fix (e.g. a newer/patched LLVM, or isolating
        # the exact kernel pattern that triggers it) is found.
        set(SYCL_BACKEND_WARNING_FLAGS -Wno-incorrect-sub-group-size -Wno-return-type)
        set(SYCL_BACKEND_LINK_FLAGS "")
        # A nightly build tried earlier (since reverted back to the pinned
        # v6.3.0 release) had a clang++ that failed to auto-locate its own
        # libspirv bitcode for the amdgcn target at the default resource-dir
        # path — apparent driver-lookup regression specific to that nightly,
        # not seen on v6.3.0. Kept as an opportunistic, non-fatal fallback:
        # native_build.amd.sh only sets SYCL_LIBSPIRV_PATH if its own search
        # finds the file, so this is a no-op on v6.3.0. Scoped only to SYCL
        # compile/link flags — NOT general CMAKE_CXX_FLAGS, since clang
        # rejects it as an unused argument (-Werror,-Wunused-command-line-argument)
        # on the many gadgetron2 files that don't pass -fsycl at all.
        if (DEFINED SYCL_LIBSPIRV_PATH)
            set(SYCL_BACKEND_WARNING_FLAGS ${SYCL_BACKEND_WARNING_FLAGS} -fsycl-libspirv-path=${SYCL_LIBSPIRV_PATH})
            set(SYCL_BACKEND_LINK_FLAGS ${SYCL_BACKEND_LINK_FLAGS} -fsycl-libspirv-path=${SYCL_LIBSPIRV_PATH})
        endif()
    endif()

    set(SYCL_COMPILE_FLAGS
        -fsycl ${SYCL_COMPILE_TARGET_FLAGS}
        -Wno-array-bounds -Wno-switch -Wno-dynamic-class-memaccess
        -Wno-undefined-var-template -Wno-parentheses -Wno-inconsistent-missing-override
        -Wno-unused-value -Wno-deprecated-declarations
        -Wno-bitwise-op-parentheses -Wno-reorder-ctor
        -Wno-comment -Wno-switch-bool
        ${SYCL_BACKEND_WARNING_FLAGS}
    )
    set(SYCL_LINK_FLAGS -fsycl ${SYCL_LINK_TARGET_FLAGS} ${SYCL_BACKEND_LINK_FLAGS})

    add_definitions(-DUSE_SYCL)
    add_definitions(-DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING)

    # DPCT and Level Zero include paths
    include_directories(${DPCT_INCLUDE_PATH})
    include_directories(${ZERO_LEVEL_INCLUDE_PATH})
    include_directories(${ONEAPI_DPL_INCLUDE_PATH})
    include_directories(${MKL_INCLUDE_PATH})
    include_directories(${SYCL_INCLUDE_PATH})

    # SYCL / oneMath libraries (available globally via SYCL_LIBRARIES).
    # Backend .so names follow oneMath's libonemath_<domain>_<backend>.so
    # convention (confirmed against the oneMath v0.9 support matrix: MKLGPU
    # for Intel GPU, rocSPARSE/rocBLAS for AMD GPU both ship a sparse_blas
    # backend, same as cuSPARSE does for the existing cuda path below).
    if (SYCL_TARGET_BACKEND STREQUAL "cuda")
        set(SYCL_LIBRARIES
            ${LIBSYCL}
            ${ONEMKL_PATH}/lib/libonemath.so
            ${ONEMKL_PATH}/lib/libonemath_blas_cublas.so
            ${ONEMKL_PATH}/lib/libonemath_sparse_blas_cusparse.so
        )
    elseif (SYCL_TARGET_BACKEND STREQUAL "intel")
        # libonemath_blas_mklgpu.so / libonemath_sparse_blas_mklgpu.so are
        # thin dispatch wrappers around real Intel MKL's own SYCL libraries
        # (oneapi::mkl::blas::*, oneapi::mkl::sparse::* — note the older
        # oneapi::mkl:: namespace, not oneMath's oneapi::math::) — they do
        # not embed those implementations themselves. Confirmed by comparing
        # to oneMath's own successful build link line, which explicitly
        # listed these same real MKL libs. MKLROOT_LIB must be passed in by
        # the build script (-DMKLROOT_LIB=<mkl>/lib).
        if (NOT DEFINED MKLROOT_LIB)
            message(FATAL_ERROR "SYCL_TARGET_BACKEND=intel requires MKLROOT_LIB to be set (e.g. -DMKLROOT_LIB=/path/to/mkl/lib)")
        endif()
        set(SYCL_LIBRARIES
            ${LIBSYCL}
            ${ONEMKL_PATH}/lib/libonemath.so
            ${ONEMKL_PATH}/lib/libonemath_blas_mklgpu.so
            ${ONEMKL_PATH}/lib/libonemath_sparse_blas_mklgpu.so
            ${MKLROOT_LIB}/libmkl_sycl_blas.so
            ${MKLROOT_LIB}/libmkl_sycl_sparse.so
            ${MKLROOT_LIB}/libmkl_intel_ilp64.so
            ${MKLROOT_LIB}/libmkl_sequential.so
            ${MKLROOT_LIB}/libmkl_core.so
            OpenCL
        )
    elseif (SYCL_TARGET_BACKEND STREQUAL "amd")
        set(SYCL_LIBRARIES
            ${LIBSYCL}
            ${ONEMKL_PATH}/lib/libonemath.so
            ${ONEMKL_PATH}/lib/libonemath_blas_rocblas.so
            ${ONEMKL_PATH}/lib/libonemath_sparse_blas_rocsparse.so
        )
    endif()

    message(STATUS "SYCL support enabled")
    message(STATUS "  SYCL_TARGET_BACKEND: ${SYCL_TARGET_BACKEND}")
    message(STATUS "  DPCT_INCLUDE_PATH: ${DPCT_INCLUDE_PATH}")
    message(STATUS "  ZERO_LEVEL_INCLUDE_PATH: ${ZERO_LEVEL_INCLUDE_PATH}")
    message(STATUS "  ONEAPI_DPL_INCLUDE_PATH: ${ONEAPI_DPL_INCLUDE_PATH}")
    message(STATUS "  MKL_INCLUDE_PATH: ${MKL_INCLUDE_PATH}")
    message(STATUS "  SYCL_INCLUDE_PATH: ${SYCL_INCLUDE_PATH}")
    message(STATUS "  LIBSYCL: ${LIBSYCL}")
    message(STATUS "  ONEMKL_PATH: ${ONEMKL_PATH}")
    message(STATUS "  SYCL_LIBRARIES: ${SYCL_LIBRARIES}")
endif()

# build options for OpenMP support
find_package(OpenMP)
option(USE_OPENMP "Use OpenMP" On)
if (OPENMP_FOUND)
    if (USE_OPENMP)
        message("OpenMP multithreading enabled")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        add_definitions(-DUSE_OMP)
    else ()
        message("OpenMP multithreading is supported, but disabled")
    endif ()
else ()
    message("OpenMP multithreading not supported")
endif ()

set(CMAKE_CXX_STANDARD 17)
if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    add_definitions(-DWIN32 -D_WIN32 -D_WINDOWS -DWIN -D_AMD64_)
    add_definitions(-Zc:__cplusplus)
    add_definitions(-DNOMINMAX)
    #add_definitions(-DUNICODE -D_UNICODE)
    add_definitions("/wd4251") #disable warnings, 4251: needs to have dll-interface to be used by clients/
    if (HAS_64_BIT)
        add_definitions(-DWIN64 -D_WIN64)
    endif ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /MP /bigobj /openmp:experimental /permissive- /FS /DNOGDI /we4667")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W1")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc /MP /bigobj /w1")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} /INCREMENTAL:NO")
    set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} /INCREMENTAL:NO")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "/debug /INCREMENTAL:NO")
    # The two flags below is to fix Windows problems in relation to multiple defined operators new/delete and some constructors that are defined in our headers
    #  set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /FORCE:MULTIPLE")
    #  set (CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /FORCE:MULTIPLE")
else ()
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    include(CheckCXXCompilerFlag)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang|IntelLLVM")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-array-bounds -Wno-switch -Wno-dynamic-class-memaccess -Wno-undefined-var-template -Wno-parentheses -Wno-inconsistent-missing-override -Wno-unused-value -Wno-deprecated-declarations -Wno-bitwise-op-parentheses -Wno-reorder-ctor")
    endif ()
    if (APPLE)
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        endif ()
    endif ()

    add_compile_options(-Werror)
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
endif ()

# whether to suppress compilation warnings
option(BUILD_SUPPRESS_WARNINGS "Build package while suppressing warnings" Off)
if (BUILD_SUPPRESS_WARNINGS)
    if (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    elseif (WIN32)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif ()
endif ()


set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
# necessary for Windows and RHEL <=6 systems

if(WIN32)
    add_definitions( -DHAVE_SNPRINTF) #Python decides to overwrite snprintf if this is not defined, because ofcourse it does.
endif()
# We actually only use system and thread explicitly, but they require linking in date_time and chrono
if (WIN32)
    find_package(Boost 1.80.0 COMPONENTS coroutine system date_time chrono program_options filesystem timer exception REQUIRED )
else()
    find_package(Boost 1.80.0 COMPONENTS coroutine system date_time program_options filesystem timer REQUIRED )
endif()
add_definitions("-DBOOST_BIND_GLOBAL_PLACEHOLDERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
add_definitions("-DBOOST_ALLOW_DEPRECATED_HEADERS") # To suppress messages in later versions of boost, in version 1.76.0 there are internal conflicts in boost
if (NOT DEFINED Boost_VERSION_STRING)
    set(Boost_VERSION_STRING "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}")
endif()


option(USE_CUDA "Build with CUDA support" Off)
if (USE_CUDA)
    message(FATAL_ERROR "CUDA is not supported in gadgetron2. Use -DUSE_SYCL=ON instead.")
endif()

option(REQUIRE_SIGNED_CONFIG "Force Gadgetron to accept only signed config files." Off)
if (REQUIRE_SIGNED_CONFIG)
    message("Compile with signed config; GTBabylon is needed")
    add_definitions(-DUSE_GTBABYLON)
    find_package(GTBabylon REQUIRED)
endif()

find_package(Armadillo REQUIRED)
add_library(armadillo INTERFACE)
set_property(TARGET armadillo PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ARMADILLO_INCLUDE_DIR})
add_definitions(-DARMA_DONT_USE_WRAPPER -DARMA_USE_CXX11 -DARMA_64BIT_WORD)
install(TARGETS armadillo EXPORT gadgetron-export)

find_package(CURL REQUIRED)
find_package(PugiXML REQUIRED)
find_package(ISMRMRD CONFIG REQUIRED)

# BART requires CUDA (not available in SYCL build)
if (NOT USE_SYCL)
    find_package(BART CONFIG QUIET)
endif()

find_package(FFTW3 REQUIRED COMPONENTS single double)
add_library(FFTW INTERFACE)
target_link_libraries(FFTW INTERFACE ${FFTW3_LIBRARIES})
set_property(TARGET FFTW PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${FFTW3_INCLUDE_DIR})
install(TARGETS FFTW EXPORT gadgetron-export)

set(GADGETRON_BUILD_RPATH "")

option(USE_MKL Off)
if (USE_MKL)
    if (LINUX)
        list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/mkl/latest")
    endif()
    find_package(MKL REQUIRED)
    link_directories(GTBLAS BEFORE INTERFACE "${MKL_ROOT}/lib/intel64/")
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${MKL_LIBRARIES})
    message("MKL LIB DIR ${MKL_ROOT}/lib/intel64")

    add_definitions(-DUSE_MKL -DARMA_BLAS_LONG_LONG -DARMA_USE_BLAS -DARMA_USE_LAPACK)
else ()
    find_package(CBLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(LAPACKE REQUIRED)
    add_library(GTBLAS INTERFACE)
    set_property(TARGET GTBLAS PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${CBLAS_INCLUDE_DIR} ${LAPACK_INCLUDE_DIR} ${LAPACKE_INCLUDE_DIR})
    set_property(TARGET GTBLAS PROPERTY INTERFACE_LINK_LIBRARIES ${CBLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${LAPACKE_LIBRARIES})

    find_package(PkgConfig)
    pkg_check_modules(PC_lapacke lapacke)

    string(COMPARE EQUAL ${BLA_VENDOR} "OpenBLAS" USING_OPENBLAS)
    if (USING_OPENBLAS)
        message("OpenBLAS detected; setting max number of OpenBLAS pthreads to one.")
        add_definitions(-DFORCE_LIMIT_OPENBLAS_NUM_THREADS)
    endif()
endif ()

install(TARGETS GTBLAS EXPORT gadgetron-export)

if(EXISTS $ENV{TORCH_HOME})
    find_package(Torch)
    if (TORCH_FOUND)
        add_definitions(-DUSE_TORCH)
        message("Find Pytorch c++ library at ${TORCH_LIBRARIES}")
    else()
        message("Do not find Pytorch c++ library")
    endif ()
else()
    message("Do not find TORCH_HOME")
endif()

option(BUILD_PYTHON_SUPPORT "Build Python gadgets etc" On)

if (BUILD_PYTHON_SUPPORT)
    find_package(Python3 COMPONENTS Development NumPy REQUIRED)
    find_package(Boost COMPONENTS python3${Python3_VERSION_MINOR} REQUIRED)

    message("Found numpy : ${Python3_NumPy_INCLUDE_DIRS} ${Python3_NumPy_LIBRARIES} ${Python3_INCLUDE_DIRS} ${Python3_LIBRARIES}")

    if (Python3_VERSION_MINOR LESS 7)
        message(FATAL_ERROR "Gadgetron Python support requires Python 3.7 or newer. Found: ${Python3_VERSION}")
    endif()

    message("Boost Version: ${Boost_VERSION_STRING} (with Python ${Python3_VERSION})")
    message("Secret: ${Boost_PYTHON3_TARGET}")

    add_library(python INTERFACE)
    set_property(TARGET python PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
    target_link_libraries(python INTERFACE Python3::Python Python3::NumPy Boost::python3${Python3_VERSION_MINOR})

    install(TARGETS python EXPORT gadgetron-export)
endif()

message("Searching for OpenGL, GLEW, GLUT, and Qt. These libraries are only used in a single standalone application and are thus non-essential.")
if(WIN32)
    message("For Windows users in particular, for ease of installation we do not recommend installing these libraries.")
endif()

find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)
find_package(Qt4 4.6)
find_package(PLplot)

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
    add_subdirectory(test)
else()
    MESSAGE("Testing not being built")
endif()


add_subdirectory(core)
add_subdirectory(toolboxes)
add_subdirectory(gadgets)
add_subdirectory(apps)

option(BUILD_WITH_GPERFTOOLS_PROFILER "Build with the gperftools profiler." Off)
if (BUILD_WITH_GPERFTOOLS_PROFILER)
    message("Adding gpertools to build.")
    find_package(Gperftools REQUIRED)
endif()


add_subdirectory(chroot)
add_subdirectory(cmake)

option(BUILD_DOCUMENTATION "Build Gadgetron documentation." Off)
if(BUILD_DOCUMENTATION)
    add_subdirectory(doc)
endif()

# install dependencies for WIN32
if (WIN32)
    if (GADGETRON_INSTALL_DEPENDENCIES)
        include(${CMAKE_SOURCE_DIR}/cmake/InstallWinDependencies.cmake)
    endif ()
endif ()

if (UNIX)
    if (NOT APPLE)
        if (GADGETRON_INSTALL_DEPENDENCIES)
            include(${CMAKE_SOURCE_DIR}/cmake/InstallLinuxDependencies.cmake)
        endif ()
    endif ()
endif ()

#  ---   Main Library  (end) ----
include(GNUInstallDirs)
install(EXPORT gadgetron-export
        FILE gadgetron-targets.cmake
        NAMESPACE
        Gadgetron::
        DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/cmake/gadgetron)

include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_SOURCE_DIR}/cmake/gadgetron-config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/gadgetron-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gadgetron
    PATH_VARS GADGETRON_INSTALL_PYTHON_MODULE_PATH GADGETRON_INSTALL_CONFIG_PATH GADGETRON_INSTALL_SCHEMA_PATH)

write_basic_package_version_file(
    ${CMAKE_BINARY_DIR}/cmake/gadgetron-config-version.cmake
    VERSION ${GADGETRON_VERSION_STRING}
    COMPATIBILITY AnyNewerVersion)

install(
    FILES
    ${CMAKE_BINARY_DIR}/cmake/gadgetron-config.cmake
    ${CMAKE_BINARY_DIR}/cmake/gadgetron-config-version.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gadgetron)

# Create package
string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER)
include(${CMAKE_SOURCE_DIR}/cmake/gadgetron_cpack.cmake)
if(CPACK_GENERATOR)
    message(STATUS "Found CPack generators: ${CPACK_GENERATOR}")
    configure_file("${CMAKE_SOURCE_DIR}/cmake/cpack_options.cmake.in" ${GADGETRON_CPACK_CFG_FILE} @ONLY)
    set(CPACK_PROJECT_CONFIG_FILE ${GADGETRON_CPACK_CFG_FILE})
    include(CPack)
endif()
