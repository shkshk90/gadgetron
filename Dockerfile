FROM golang:tip-alpine3.22 AS mrd

RUN mkdir -p /downloads \
        && wget -O /downloads/mrd-storage-server-v0.0.12.tar.gz https://github.com/ismrmrd/mrd-storage-server/archive/refs/tags/v0.0.12.tar.gz  \
        && tar xzf /downloads/mrd-storage-server-v0.0.12.tar.gz -C      /downloads              \
        && mv /downloads/mrd-storage-server-0.0.12                      /downloads/mrd          \
        && cd /downloads/mrd                                                                    \
        && go build                                                 
        # cp   /downloads/mrd/mrd-storage-server    /usr/local/bin

FROM ubuntu:24.04 AS deps

RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf
RUN sed -i 's|http://archive.ubuntu.com|http://de.archive.ubuntu.com|g' /etc/apt/sources.list.d/ubuntu.sources \
        && apt-get update \
        && apt-get -y install cmake wget curl g++ libpugixml-dev libhdf5-dev make \
        && apt-get autoremove -y && apt-get clean 

RUN mkdir -p /downloads \
        && curl --location https://github.com/ismrmrd/ismrmrd/archive/refs/tags/v1.14.3.tar.gz --silent --output /downloads/ismrmrd-v1.14.3.tar.gz \
        && tar xzf /downloads/ismrmrd-v1.14.3.tar.gz -C /downloads              \          
        && mv   /downloads/ismrmrd-1.14.3       /downloads/ismrmrd              \
        && cmake -S /downloads/ismrmrd -B /downloads/ismrmrd/build              \
        && cmake --build /downloads/ismrmrd/build -j 8                          \
        && cmake --install /downloads/ismrmrd/build --prefix /ismrmrd           \
        && rm -rf /downloads

RUN mkdir -p /downloads  \
        && curl --output /downloads/boost_1_80_0.tar.gz --silent --location https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz  \
        && tar xzf /downloads/boost_1_80_0.tar.gz       -C /downloads                   \
        && mv /downloads/boost_1_80_0           /downloads/boost                        \
        && cd /downloads/boost                                                          \
        && ./bootstrap.sh --prefix=/boost                                               \
        && ./b2                                                                         \
        && ./b2 install --prefix=/boost                                                 \
        && rm -rf /downloads

RUN mkdir -p /downloads  \
        && curl --output /downloads/plplot-5.15.0.tar.gz --silent --location https://deac-fra.dl.sourceforge.net/project/plplot/plplot/5.15.0%20Source/plplot-5.15.0.tar.gz?viasf=1  \
        && tar xzf /downloads/plplot-5.15.0.tar.gz       -C /downloads                          \
        && mv /downloads/plplot-5.15.0           /downloads/plplot                              \
        && cd /downloads/plplot                                                                 \
        && mkdir -p /plplot                                                                     \
        && cmake -S /downloads/plplot -B /downloads/plplot/build -DCMAKE_INSTALL_PREFIX=/plplot \
        && cmake --build /downloads/plplot/build -j 8                                           \
        && cmake --install /downloads/plplot/build --prefix /plplot                             \
        && rm -rf /downloads




FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

COPY --from=deps /ismrmrd                               /ismrmrd
COPY --from=deps /boost                                 /boost
COPY --from=deps /plplot                                /plplot

COPY --from=mrd /downloads/mrd/mrd-storage-server       /usr/local/bin/mrd-storage-server

# COPY docker/oneapi-for-nvidia-gpus-2025.2.0-linux.sh /tmp/oneapi-for-nvidia-gpus-2025.2.0-linux.sh

RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf \
    && sed -i 's|http://archive.ubuntu.com|http://ftp.uni-stuttgart.de|g' /etc/apt/sources.list \
    && sed -i 's|https://archive.ubuntu.com|https://ftp.uni-stuttgart.de|g' /etc/apt/sources.list \
    && DEBIAN_FRONTEND=noninteractive apt update \
    && DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt -y install tzdata \
    && DEBIAN_FRONTEND=noninteractive apt install -y --allow-change-held-packages --fix-missing \
        breathe-doc \
        build-essential \
        cmake \
        curl \
        doxygen \
        dcmtk \
        git \
        graphviz \
        jq \
        autoconf \
        sudo \
        libtool \
        ninja-build \
        nlohmann-json3-dev \
        yq \
        zfp \
        libarmadillo-dev \
        libbart-dev \
        ocl-icd-opencl-dev \
        libhwloc-dev \
        numactl \
        ocl-icd-libopencl1 \
        # libboost-system-dev \
        # libboost-coroutine-dev \
        # libboost-timer-dev \
        # libboost-python-dev \
        libcublas-12-8 \
        libcublas-dev-12-8 \
        libcufftw11 \
        libcurl4t64 \
        libdcmtk-dev \
        libfftw3-dev \
        libglew-dev \
        libglut-dev \
        libgmock-dev \
        libgtest-dev \
        libhdf5-serial-dev \
        libhowardhinnant-date-dev \
        liblapacke-dev \
        libnvonnxparsers-dev \
        libonnx-dev \
        # libplplot-dev \
        libpugixml-dev \
        librange-v3-dev \
        libzfp-dev \
        pkg-config \
        python3-numpy \
        python3-matplotlib \
        python3-breathe \
        python3-deepdiff \
        python3-doxypypy \
        python3-scipy \
        wget \
    && apt-get autoremove -y && apt-get clean 

RUN mkdir -p /downloads \
    && curl --output /downloads/bart-v0.9.00.tar.gz --silent --location https://github.com/mrirecon/bart/archive/refs/tags/v0.9.00.tar.gz \
    && tar xzf /downloads/bart-v0.9.00.tar.gz -C /downloads                                     \
    && mv /downloads/bart-0.9.00 /downloads/bart                                                \
    && cd /downloads/bart                                                                       \
    && mkdir -p /bart                                                                           \
    && make                                                                                     \
    && make shared-lib                                                                          \
    && cp   /downloads/bart/libbart.so                /bart/libbart.so                          \
    && cp   /downloads/bart/src/bart_embed_api.h      /bart/bart_embed_api.h                    \
    && rm -rf /downloads
    
    
    
RUN mkdir -p /downloads \
    && curl --output /downloads/zstd-1.5.7.tar.gz --silent --location https://github.com/facebook/zstd/releases/download/v1.5.7/zstd-1.5.7.tar.gz \
    && tar -xzf /downloads/zstd-1.5.7.tar.gz -C /downloads \ 
    && cd /downloads/zstd-1.5.7 \
    && CFLAGS="-fPIC" CXXFLAGS="-fPIC" make -j12 \
    && make install \
    && rm -rf /downloads 

RUN mkdir -p /downloads \
    && curl --output /downloads/v6.3.0.tar.gz --silent --location https://github.com/intel/llvm/archive/refs/tags/v6.3.0.tar.gz \
    && tar -xzf /downloads/v6.3.0.tar.gz -C /downloads \ 
    && mkdir -p /sycl_linux \ 
    && mkdir -p /dpcpp_home \ 
    && mv /downloads/llvm-6.3.0 /sycl_linux/llvm \ 
    && rm -rf /downloads \ 
    && python3 /sycl_linux/llvm/buildbot/configure.py -t Release --use-zstd --no-assertions  --cuda -o /dpcpp_home \
    && python3 /sycl_linux/llvm/buildbot/compile.py -o /dpcpp_home -j 12

RUN mkdir -p /downloads \
    && curl --output /downloads/intel-oneapi-base-toolkit-2025.3.1.36_offline.sh --silent --location \
        https://registrationcenter-download.intel.com/akdlm/IRC_NAS/6caa93ca-e10a-4cc5-b210-68f385feea9e/intel-oneapi-base-toolkit-2025.3.1.36_offline.sh \
    && chmod +x /downloads/intel-oneapi-base-toolkit-2025.3.1.36_offline.sh   \
    && /downloads/intel-oneapi-base-toolkit-2025.3.1.36_offline.sh  \ 
                --remove-extracted-files yes   \
                --log   /tmp/intel-oneapi-base-toolkit.log      \
                -a \
                --silent \
                --action install \
                --components all \
                --eula accept \
                --intel-sw-improvement-program-consent decline  \
                --download-cache /tmp   \
                --download-dir /tmp \
    && rm -rf /downloads \
    && ln -s /opt/intel/oneapi/mkl/2025.3/lib/libmkl_core.so  /usr/lib/x86_64-linux-gnu/libmkl_core.so  \
    && ln -s /opt/intel/oneapi/mkl/2025.3/lib/libmkl_gnu_thread.so  /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so \
    && ln -s /opt/intel/oneapi/mkl/2025.3/lib/libmkl_intel_lp64.so /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so 
    # && echo "source /opt/intel/oneapi/setvars.sh --include-intel-llvm"                  >> /etc/bash.bashrc

RUN mkdir -p /downloads \
    && cd /downloads \
    && curl -O --silent --location https://github.com/intel/intel-graphics-compiler/releases/download/v2.27.10/intel-igc-core-2_2.27.10+20617_amd64.deb \
    && curl -O --silent --location https://github.com/intel/intel-graphics-compiler/releases/download/v2.27.10/intel-igc-opencl-2_2.27.10+20617_amd64.deb \
    && curl -O --silent --location https://github.com/intel/compute-runtime/releases/download/26.01.36711.4/intel-ocloc_26.01.36711.4-0_amd64.deb \
    && curl -O --silent --location https://github.com/intel/compute-runtime/releases/download/26.01.36711.4/intel-opencl-icd_26.01.36711.4-0_amd64.deb \
    && curl -O --silent --location https://github.com/intel/compute-runtime/releases/download/26.01.36711.4/libigdgmm12_22.9.0_amd64.deb \
    && curl -O --silent --location https://github.com/intel/compute-runtime/releases/download/26.01.36711.4/libze-intel-gpu1_26.01.36711.4-0_amd64.deb \
    && curl -O --silent --location https://github.com/intel/cm-compiler/releases/download/cmclang-1.0.119/intel-igc-cm-1.0-119.u18.04-release.x86_64.deb \
    && curl -O --silent --location https://github.com/oneapi-src/level-zero/releases/download/v1.26.3/level-zero_1.26.3+u22.04_amd64.deb \
    && dpkg -i *.deb \
    && rm -rf /downloads 

COPY docker/install_mkl.sh /tmp/install_mkl.sh
RUN chmod +x /tmp/install_mkl.sh 

RUN DEBIAN_FRONTEND=noninteractive apt install -y \
        libpfm4-dev python3-dev \
        libbabeltrace-dev libcapstone-dev \
        libtraceevent-dev systemtap-sdt-dev libslang2-dev \
        libdebuginfod-dev libdw-dev build-essential flex bison libelf-dev \
        gdb \
    && apt-get autoremove -y && apt-get clean 
RUN mkdir -p /downloads \
    && curl --output /downloads/linux-6.16.1.tar.gz --silent --location https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.16.1.tar.gz \
    && tar -xzf /downloads/linux-6.16.1.tar.gz -C /downloads \
    && make -C /downloads/linux-6.16.1/tools/perf install \
    && mv /downloads/linux-6.16.1/tools/perf /perf \
    && rm -rf /downloads

    # && /tmp/install_mkl.sh \
    # && echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/install/lib"                     >> /etc/bash.bashrc     \
    # && echo "export LD_LIBRARY_PATH=/oneMKLwithCublas/lib:\$LD_LIBRARY_PATH"            >> /etc/bash.bashrc     \
    # && echo "export LIBRARY_PATH=/oneMKLwithCublas/lib:\$LIBRARY_PATH"                  >> /etc/bash.bashrc     \
    # && echo "export CPLUS_INCLUDE_DIR=/oneMKLwithCublas/include:\$CPLUS_INCLUDE_DIR"    >> /etc/bash.bashrc     \
    # && echo "export CPLUS_INCLUDE_DIR=/include:\$CPLUS_INCLUDE_DIR"                     >> /etc/bash.bashrc


# RUN mkdir -p /downloads                                         \
#         && curl --output /downloads/oneMath-v0.8.tar.gz --silent --location https://github.com/uxlfoundation/oneMath/archive/refs/tags/v0.8.tar.gz \
#         && tar xzf /downloads/oneMath-v0.8.tar.gz -C /downloads \
#         && mv /downloads/oneMath-0.8 /downloads/oneMKL          \
#         && . /opt/intel/oneapi/setvars.sh  --include-intel-llvm \
#         && mkdir -p /oneMKLwithCublas                           \
#         && cmake -S /downloads/oneMKL -B /oneMKLwithCublas      \
#                 -DCMAKE_CXX_COMPILER=icpx                       \
#                 -DCMAKE_C_COMPILER=icx                          \
#                 -DENABLE_MKLGPU_BACKEND=OFF                     \
#                 -DENABLE_MKLCPU_BACKEND=OFF                     \
#                 -DENABLE_CUBLAS_BACKEND=ON                      \
#                 -DTARGET_DOMAINS=blas                           \
#         && cd /oneMKLwithCublas                                 \
#         && make     -j 8                                        \
#     && echo "source /opt/intel/oneapi/setvars.sh --include-intel-llvm"                  >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/build/cmake-install/lib"         >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=/oneMKLwithCublas/lib:\$LD_LIBRARY_PATH"            >> /etc/bash.bashrc     \
#     && echo "export LIBRARY_PATH=/oneMKLwithCublas/lib:\$LIBRARY_PATH"                  >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/oneMKLwithCublas/include:\$CPLUS_INCLUDE_DIR"    >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/include:\$CPLUS_INCLUDE_DIR"                     >> /etc/bash.bashrc     \
#     && rm -rf /downloads



# RUN chmod +x /tmp/install_mkl.sh        \
#     && /tmp/install_mkl.sh              \
#     && echo "source /opt/intel/oneapi/setvars.sh --include-intel-llvm"                  >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/build/cmake-install/lib"         >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=/oneMKLwithCublas/lib:\$LD_LIBRARY_PATH"            >> /etc/bash.bashrc     \
#     && echo "export LIBRARY_PATH=/oneMKLwithCublas/lib:\$LIBRARY_PATH"                  >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/oneMKLwithCublas/include:\$CPLUS_INCLUDE_DIR"    >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/include:\$CPLUS_INCLUDE_DIR"                     >> /etc/bash.bashrc

CMD [ "/bin/bash" ]