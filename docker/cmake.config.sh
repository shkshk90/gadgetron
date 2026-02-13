#!/usr/bin/env bash

set -euo pipefail


# CLANG with CUDA

# mkdir -p /build/clang
# touch /build/clang/CMakeCache.txt
# rm -f /build/clang/CMakeCache.txt

# cmake                                                               \
#   -S /gadgetron                                                     \
#   -B /build/clang                                                   \
#   -G Ninja                                                          \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                                \
#   -DCMAKE_CXX_FLAGS=-I${HOME}/.local/include                        \
#   -DCMAKE_C_COMPILER=clang                                          \
#   -DCMAKE_CXX_COMPILER=clang++                                      \
#   -DCMAKE_CUDA_HOST_COMPILER=clang                                  \
#   -DBoost_NO_BOOST_CMAKE=TRUE                                       \
#   -DBoost_NO_SYSTEM_PATHS=TRUE                                      \
#   -DBOOST_ROOT:PATHNAME=/boost                                      \
#   -DBoost_LIBRARY_DIRS:PATH=/boost/lib                              \
#   -DMKL_ROOT:PATH=/opt/intel/oneapi/2025.2                          \
#   -DISMRMRD_DIR:PATH=/ismrmrd/lib/cmake/ISMRMRD                     \
#   -DHDF5_DIR:PATH=/libhdf5/./HDF_Group/HDF5/1.14.3/cmake            \
#   -DPLPLOT_PATH:PATH=/plplot/include                                \
#   -DPLPLOT_CXX_LIB:FILEPATH=/plplot/lib/libplplotcxx.so.15.0.0      \
#   -DPLPLOT_LIB:FILEPATH=/plplot/lib/libplplot.so.17.0.0             \
#   -DBUILD_PYTHON_SUPPORT=FALSE                                      \
#   -DBART_ROOT:PATH=/bart                                            \
#   -DUSE_MKL=ON                                                      \
#   -DUSE_CUDA=ON                                                     \
#   -DUSE_SYCL=OFF                                                    \
#   -DCMAKE_INSTALL_PREFIX=/install                                   \
#   -DBUILD_SUPPRESS_WARNINGS=TRUE                                    \
#   -DUSE_AMALGAMATED=TRUE                                            \
#   -DUSE_OPENMP=OFF                                                  \
#   -DBUILD_TESTING=ON                                                \
#   -Wno-dev


# SYCL build example (uncomment to use):
# First run: ./docker/convert_to_sycl.sh --build-dir /build/clang
# Then configure with SYCL:

export LD_LIBRARY_PATH=/oneMKLwithCublas/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/oneMKLwithCublas/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=/oneMKLwithCublas/include


mkdir -p /build/sycl.intel.clang.ii
touch /build/sycl.intel.clang.ii/CMakeCache.txt
rm -f /build/sycl.intel.clang.ii/CMakeCache.txt

# . /opt/intel/oneapi/setvars.sh

# -S /gadgetron                                                   
cmake                                                             \
  -S /gadgetron/docker/test                                                   \
  -B /build/sycl.intel.clang.ii                                      \
  -G Ninja                                                        \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                              \
  -DCMAKE_C_COMPILER=/dpcpp_home/install/bin/clang                \
  -DCMAKE_CXX_COMPILER=/dpcpp_home/install/bin/clang++            \
  -DBoost_NO_BOOST_CMAKE=TRUE                                     \
  -DBoost_NO_SYSTEM_PATHS=TRUE                                    \
  -DBOOST_ROOT:PATHNAME=/boost                                    \
  -DBoost_LIBRARY_DIRS:PATH=/boost/lib                            \
  -DISMRMRD_DIR:PATH=/ismrmrd/lib/cmake/ISMRMRD                   \
  -DHDF5_DIR:PATH=/libhdf5/./HDF_Group/HDF5/1.14.3/cmake          \
  -DPLPLOT_PATH:PATH=/plplot/include                              \
  -DPLPLOT_CXX_LIB:FILEPATH=/plplot/lib/libplplotcxx.so.15.0.0    \
  -DPLPLOT_LIB:FILEPATH=/plplot/lib/libplplot.so.17.0.0           \
  -DBUILD_PYTHON_SUPPORT=FALSE                                    \
  -DUSE_MKL=OFF                                                    \
  -DUSE_CUDA=OFF                                                  \
  -DUSE_SYCL=ON                                                   \
  -DUSE_OPENMP=OFF                                                \
  -DDPCPP_HOME=/dpcpp_home/install                                \
  -DCMAKE_INSTALL_PREFIX=/install                                 \
  -DBUILD_SUPPRESS_WARNINGS=TRUE                                  \
  -DUSE_AMALGAMATED=TRUE                                          \
  -DBUILD_TESTING=ON                                              \
  -Wno-dev  