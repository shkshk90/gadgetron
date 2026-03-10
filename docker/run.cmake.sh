#!/usr/bin/env bash

set -euo pipefail

if [ ! -d "/oneMKLwithCublas/lib" ]; then
    echo "Running install mkl"
    /gadgetron/docker/install_mkl.sh
fi


cmake --build /build/sycl.intel.clang.ii  -j16 --config Release # --verbose # --target test_all
mkdir -p /install/new/sycl.intel.clang.ii
cmake --install /build/sycl.intel.clang.ii --prefix /install/new/sycl.intel.clang.ii # --component test_all

echo "run :: "
echo "       LD_LIBRARY_PATH=/dpcpp_home/install/lib:/oneMKLwithCublas/lib:/opt/intel/oneapi/2025.3/lib:\$LD_LIBRARY_PATH /install/new/sycl.intel.clang.ii/bin/test_all"

# Failing tests:
# /install/icpx/bin/test_all --gtest_filter=cuNDArray_blas_Cplx*

