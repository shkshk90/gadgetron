#!/usr/bin/env bash

set -euo pipefail

if [ ! -d "/oneMKLwithCublas/lib" ]; then
    echo "Running install mkl"
    /tmp/install_mkl.sh
fi

cmake --build /build/cuda -j 12

mkdir -p /install/new/cuda


cmake --install /build/cuda --prefix /install/new/cuda

echo "run ::    "
echo "          LD_LIBRARY_PATH=/install/new/cuda/lib:\$LD_LIBRARY_PATH /install/new/cuda/bin/test/test_all"

exit 0

cmake --build /build/icpx -j 12
mkdir -p /install/new/icpx
cmake --install /build/icpx --prefix /install/new/icpx
# 38385
echo "run ::    LD_LIBRARY_PATH=/install/new/icpx/lib:\$LD_LIBRARY_PATH "
echo "          /install/new/icpx/bin/test_all"

# Failing tests:
# /install/icpx/bin/test_all --gtest_filter=curveFitting_test/0.T2SE
# /install/icpx/bin/test_all --gtest_filter=cuNDArray_blas_Cplx*

# Failures:

# [  FAILED  ] 33 tests, listed below:
# [  FAILED  ] curveFitting_test/0.T2SE, where TypeParam = float
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/0, where GetParam() = (42, 0, 2)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/1, where GetParam() = (42, 0, 3)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/2, where GetParam() = (42, 0, 4)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/3, where GetParam() = (42, 0, 5)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/4, where GetParam() = (42, 0, 6)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/5, where GetParam() = (42, 0, 7)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/6, where GetParam() = (42, 0, 8)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/7, where GetParam() = (42, 0, 9)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/8, where GetParam() = (42, 1000, 2)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/9, where GetParam() = (42, 1000, 3)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/10, where GetParam() = (42, 1000, 4)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/11, where GetParam() = (42, 1000, 5)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/12, where GetParam() = (42, 1000, 6)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/13, where GetParam() = (42, 1000, 7)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/14, where GetParam() = (42, 1000, 8)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/15, where GetParam() = (42, 1000, 9)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/16, where GetParam() = (287, 0, 2)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/17, where GetParam() = (287, 0, 3)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/18, where GetParam() = (287, 0, 4)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/19, where GetParam() = (287, 0, 5)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/20, where GetParam() = (287, 0, 6)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/21, where GetParam() = (287, 0, 7)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/22, where GetParam() = (287, 0, 8)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/23, where GetParam() = (287, 0, 9)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/24, where GetParam() = (287, 1000, 2)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/25, where GetParam() = (287, 1000, 3)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/26, where GetParam() = (287, 1000, 4)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/27, where GetParam() = (287, 1000, 5)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/28, where GetParam() = (287, 1000, 6)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/29, where GetParam() = (287, 1000, 7)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/30, where GetParam() = (287, 1000, 8)
# [  FAILED  ] TestMatrixForCompression/NHLBICompression.Roundtrip/31, where GetParam() = (287, 1000, 9)
