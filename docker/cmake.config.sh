#!/usr/bin/env bash

set -euo pipefail


# CUDA 

mkdir -p /build/cuda

touch /build/cuda/CMakeCache.txt
rm -f /build/cuda/CMakeCache.txt

cmake                                                               \
  -S /gadgetron                                                     \
  -B /build/cuda                                                    \
  -G Ninja                                                          \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                                \
  -DCMAKE_CXX_FLAGS=-I${HOME}/.local/include                        \
  -DCMAKE_CUDA_HOST_COMPILER=gcc-14                                 \
  -DBoost_NO_BOOST_CMAKE=TRUE                                       \
  -DBoost_NO_SYSTEM_PATHS=TRUE                                      \
  -DBOOST_ROOT:PATHNAME=/boost                                      \
  -DBoost_LIBRARY_DIRS:PATH=/boost/lib                              \
  -DISMRMRD_DIR:PATH=/ismrmrd/lib/cmake/ISMRMRD                     \
  -DHDF5_DIR:PATH=/libhdf5/./HDF_Group/HDF5/1.14.3/cmake            \
  -DPLPLOT_PATH:PATH=/plplot/include                                \
  -DPLPLOT_CXX_LIB:FILEPATH=/plplot/lib/libplplotcxx.so.15.0.0      \
  -DPLPLOT_LIB:FILEPATH=/plplot/lib/libplplot.so.17.0.0             \
  -DBUILD_PYTHON_SUPPORT=FALSE                                      \
  -DBART_ROOT:PATH=/bart                                            \
  -DUSE_MKL=ON                                                      \
  -DUSE_CUDA=ON                                                     \
  -DCMAKE_INSTALL_PREFIX=/install                                   \
  -DBUILD_SUPPRESS_WARNINGS=TRUE                                    \
  -Wno-dev

# ICPX

# mkdir -p /build/icpx
# touch /build/icpx/CMakeCache.txt
# rm -f /build/icpx/CMakeCache.txt

# cmake                                                               \
#   -S /gadgetron                                                     \
#   -B /build/icpx                                                    \
#   -G Ninja                                                          \
#   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                                \
#   -DCMAKE_CXX_FLAGS=-I${HOME}/.local/include                        \
#   -DCMAKE_C_COMPILER=icx                                            \
#   -DCMAKE_CXX_COMPILER=icpx                                         \
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
#   -DCMAKE_INSTALL_PREFIX=/install                                   \
#   -DBUILD_SUPPRESS_WARNINGS=TRUE                                    \
#   -Wno-dev