#!/usr/bin/env bash

set -e

mkdir -p /downloads
rm -rf /downloads/*

curl --output /downloads/oneMath-v0.9.tar.gz --silent --location https://github.com/uxlfoundation/oneMath/archive/refs/tags/v0.9.tar.gz
tar xzf /downloads/oneMath-v0.9.tar.gz -C /downloads

mv /downloads/oneMath-0.9 /downloads/oneMKL
mkdir -p /oneMKLwithCublas
rm -rf /oneMKLwithCublas/*


cmake -S /downloads/oneMKL -B /downloads/oneMKLwithCublas     \
        -DCMAKE_CXX_COMPILER=/dpcpp_home/install/bin/clang++                   \
        -DCMAKE_C_COMPILER=/dpcpp_home/install/bin/clang                      \
        -DENABLE_MKLGPU_BACKEND=OFF                 \
        -DENABLE_MKLCPU_BACKEND=OFF                 \
        -DENABLE_CUBLAS_BACKEND=ON                  \
        -DENABLE_CUFFT_BACKEND=ON       \
        -DENABLE_CURAND_BACKEND=ON       \
        -DENABLE_CUSOLVER_BACKEND=ON       \
        -DENABLE_CUSPARSE_BACKEND=ON       \
        -DBUILD_FUNCTIONAL_TESTS=False \
        -DBUILD_EXAMPLES=True  \
        -DONEMATH_SYCL_IMPLEMENTATION=dpc++


cmake --build /downloads/oneMKLwithCublas --config Release -j 12
cmake --install /downloads/oneMKLwithCublas --prefix /oneMKLwithCublas

mkdir -p /OneMKL/
rm -rf /OneMKL/*

mv /downloads/oneMKL                    /OneMKL/src 
mv /downloads/oneMKLwithCublas          /OneMKL/bld

rm -rf /downloads