#!/usr/bin/env bash

set -e

mkdir -p /downloads

curl --output /downloads/oneMath-v0.8.tar.gz --silent --location https://github.com/uxlfoundation/oneMath/archive/refs/tags/v0.8.tar.gz
tar xzf /downloads/oneMath-v0.8.tar.gz -C /downloads

mv /downloads/oneMath-0.8 /downloads/oneMKL
mkdir -p /oneMKLwithCublas

source /opt/intel/oneapi/setvars.sh  --include-intel-llvm
cmake -S /downloads/oneMKL -B /oneMKLwithCublas     \
        -DCMAKE_CXX_COMPILER=icpx                   \
        -DCMAKE_C_COMPILER=icx                      \
        -DENABLE_MKLGPU_BACKEND=OFF                 \
        -DENABLE_MKLCPU_BACKEND=OFF                 \
        -DENABLE_CUBLAS_BACKEND=ON                  \
        -DTARGET_DOMAINS=blas

cd /oneMKLwithCublas
make     -j 8  
rm -rf /downloads