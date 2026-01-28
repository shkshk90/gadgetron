#!/usr/bin/env bash

set -euo pipefail

if [ ! -d "/oneMKLwithCublas/lib" ]; then
    echo "Running install mkl"
    /tmp/install_mkl.sh
fi

cmake --build /build/clang  -j16
mkdir -p /install/new/clang
cmake --install /build/clang --prefix /install/new/clang

echo "run :: "
echo "       LD_LIBRARY_PATH=/install/new/clang/lib:\$LD_LIBRARY_PATH /install/new/clang/bin/test_all"

# Failing tests:
# /install/icpx/bin/test_all --gtest_filter=cuNDArray_blas_Cplx*

