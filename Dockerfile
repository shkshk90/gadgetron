FROM golang:tip-alpine3.22 AS mrd

RUN mkdir -p /downloads \
        && wget -O /downloads/mrd-storage-server-v0.0.12.tar.gz https://github.com/ismrmrd/mrd-storage-server/archive/refs/tags/v0.0.12.tar.gz  \
        && tar xzf /downloads/mrd-storage-server-v0.0.12.tar.gz -C      /downloads              \
        && mv /downloads/mrd-storage-server-0.0.12                      /downloads/mrd          \
        && cd /downloads/mrd                                                                    \
        && go build                                                 
        # cp   /downloads/mrd/mrd-storage-server    /usr/local/bin

FROM debian:trixie-slim AS deps

RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf
RUN sed -i 's|http://deb.debian.org|http://ftp.de.debian.org|g' /etc/apt/sources.list.d/debian.sources \
        && apt-get update \
        && apt-get -y install cmake wget curl g++ libpugixml-dev libhdf5-dev  \
        && apt-get autoremove -y && apt-get clean 

RUN mkdir -p /downloads  \
        && wget -O /downloads/ismrmrd-v1.14.3.tar.gz https://github.com/ismrmrd/ismrmrd/archive/refs/tags/v1.14.3.tar.gz \
        && tar xzf /downloads/ismrmrd-v1.14.3.tar.gz -C /downloads              \          
        && mv   /downloads/ismrmrd-1.14.3       /downloads/ismrmrd              \
        && cmake -S /downloads/ismrmrd -B /downloads/ismrmrd/build              \
        && cmake --build /downloads/ismrmrd/build -j 8                          \
        && cmake --install /downloads/ismrmrd/build --prefix /ismrmrd           \
        && rm -rf /downloads

RUN mkdir -p /downloads  \
        && wget -O /downloads/boost_1_80_0.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz  \
        && tar xzf /downloads/boost_1_80_0.tar.gz       -C /downloads                   \
        && mv /downloads/boost_1_80_0           /downloads/boost                        \
        && cd /downloads/boost                                                          \
        && ./bootstrap.sh --prefix=/boost                                               \
        && ./b2                                                                         \
        && ./b2 install --prefix=/boost                                                 \
        && rm -rf /downloads

# ls /ismrmrd/
# bin  include  lib  share
#
# ls /boost
# include  lib

#CMD [ "/bin/bash" ]






# FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# COPY docker/oneapi-for-nvidia-gpus-2025.2.0-linux.sh /tmp/oneapi-for-nvidia-gpus-2025.2.0-linux.sh
# RUN sed -i 's|http://archive.ubuntu.com|http://ftp.uni-stuttgart.de|g' /etc/apt/sources.list \
#     && sed -i 's|https://archive.ubuntu.com|https://ftp.uni-stuttgart.de|g' /etc/apt/sources.list \
#     && DEBIAN_FRONTEND=noninteractive apt update \
#     && DEBIAN_FRONTEND=noninteractive TZ=Europe/Berlin apt -y install \
#         tzdata \
#     && apt-get autoremove -y && apt-get clean \





# RUN DEBIAN_FRONTEND=noninteractive apt install -y --allow-change-held-packages --fix-missing \
#         breathe-doc \
#         build-essential \
#         cmake \
#         curl \
#         doxygen \
#         dcmtk \
#         git \
#         golang \
#         graphviz \
#         # ismrmrd-tools \
#         jq \
#         ninja-build \
#         nlohmann-json3-dev \
#         yq \
#         zfp \
#         libarmadillo-dev \
        # libbart-dev \
#  Dockerfile        # libboost-system-dev \
#         # libboost-coroutine-dev \
#         # libboost-timer-dev \
#         # libboost-python-dev \
#         libcublas-12-8 \
#         libcublas-dev-12-8 \
#         libcufftw11 \
#         libcurl4t64 \
#         libdcmtk-dev \
#         libfftw3-dev \
#         libglew-dev \# CMD [ "/bin/bash" ]
#         libglut-dev \
#         libgmock-dev \
#         libgtest-dev \
#         libhdf5-serial-dev \
#         libhowardhinnant-date-dev \
#         liblapacke-dev \
#         libnvonnxparsers-dev \
#         libonnx-dev \
#         # libplplot-dev \
#         libpugixml-dev \
#         librange-v3-dev \
#         libzfp-dev \
#         pkg-config \
#         python3-numpy \
#         python3-matplotlib \
#         python3-breathe \
#         python3-deepdiff \
#         python3-doxypypy \
#         python3-scipy \
#         wget \
#     && apt clean 

# RUN mkdir -p /downloads \
#     && mkdir -p /build \
#     && wget -O /downloads/intel-oneapi-base-toolkit-2025.3.0.375_offline.sh https://registrationcenter-download.intel.com/akdlm/IRC_NAS/d640da34-77cc-4ab2-8019-ac5592f4ec19/intel-oneapi-base-toolkit-2025.3.0.375_offline.sh \
#     && wget -O /downloads/bart-v0.9.00.tar.gz https://github.com/mrirecon/bart/archive/refs/tags/v0.9.00.tar.gz \
#     && tar xzf /downloads/bart-v0.9.00.tar.gz -C /downloads/bart \
#     && wget -O /downloads/ismrmrd-v1.14.3.tar.gz https://github.com/ismrmrd/ismrmrd/archive/refs/tags/v1.14.3.tar.gz \
#     && tar xzf /downloads/ismrmrd-v1.14.3.tar.gz -C /downloads/ismrmrd \
# #     && wget -O /downloads/mrd-storage-server-v0.0.12.tar.gz https://github.com/ismrmrd/mrd-storage-server/archive/refs/tags/v0.0.12.tar.gz \
# #     && tar xzf /downloads/mrd-storage-server-v0.0.12.tar.gz -C /downloads/mrd \
#     && wget -O /downloads/oneMath-v0.8.tar.gz https://github.com/uxlfoundation/oneMath/archive/refs/tags/v0.8.tar.gz \
#     && tar xzf /downloads/oneMath-v0.8.tar.gz -C /downloads/oneMKL \
#     && sh /downloads/intel-oneapi-base-toolkit-2025.3.0.375_offline.sh -a \
#                 --silent \
#                 --action install \
#                 --components all \
#                 --eula accept \
#                 --intel-sw-improvement-program-consent decline \
#     && chmod +x /tmp/oneapi-for-nvidia-gpus-2025.2.0-linux.sh \
#     && /tmp/oneapi-for-nvidia-gpus-2025.2.0-linux.sh --yes

# RUN     rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotada.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplot_octave.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotjavac_wrap.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotada_static.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_tclmatrix.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplottcltk.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplottcltk_Main.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_pltcl.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plserver.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_Pltk_init.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotc.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotwxwidgets.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotluac.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplotqt.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plplot_pyqt5.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_cairo.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_qt.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_mem.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_ntk.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_null.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_pdf.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_ps.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_svg.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_tk.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_tkwin.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_wxwidgets.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_xfig.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_xwin.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_pltek.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_wxPLViewer.cmake && \
#         rm /usr/lib/x86_64-linux-gnu/cmake/plplot/export_plfortrandemolib.cmake && \
#         ln -s /opt/intel/oneapi/mkl/2025.2/lib/libmkl_core.so  /usr/lib/x86_64-linux-gnu/libmkl_core.so && \
#         ln -s /opt/intel/oneapi/mkl/2025.2/lib/libmkl_gnu_thread.so  /usr/lib/x86_64-linux-gnu/libmkl_gnu_thread.so && \
#         ln -s /opt/intel/oneapi/mkl/2025.2/lib/libmkl_intel_lp64.so /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so

        

        
# RUN     cd /downloads/bart                                          \
#         && make                                                     \
#         && make shared-lib                                          \
#         && cp   /bart/libbart.so                /usr/lib/           \
#         && cp   /bart/src/bart_embed_api.h      /usr/include/bart

# RUN     source /opt/intel/oneapi/setvars.sh                     \
#         && mkdir -p /oneMKLwithCublas                           \
#         && cmake -S /downloads/oneMKL -B /oneMKLwithCublas      \
#                 -DCMAKE_CXX_COMPILER=icpx                       \
#                 -DCMAKE_C_COMPILER=icx                          \
#                 -DENABLE_MKLGPU_BACKEND=OFF                     \
#                 -DENABLE_MKLCPU_BACKEND=OFF                     \
#                 -DENABLE_CUBLAS_BACKEND=ON                      \
#                 -DTARGET_DOMAINS=blas                           \
#         && cd /oneMKLwithCublas                                 \
#         && make

# RUN echo "source /opt/intel/oneapi/setvars.sh --include-intel-llvm"                     >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/build/cmake-install/lib"         >> /etc/bash.bashrc     \
#     && echo "export LD_LIBRARY_PATH=/oneMKLwithCublas/lib:\$LD_LIBRARY_PATH"            >> /etc/bash.bashrc     \
#     && echo "export LIBRARY_PATH=/oneMKLwithCublas/lib:\$LIBRARY_PATH"                  >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/oneMKLwithCublas/include:\$CPLUS_INCLUDE_DIR"    >> /etc/bash.bashrc     \
#     && echo "export CPLUS_INCLUDE_DIR=/include:\$CPLUS_INCLUDE_DIR"                     >> /etc/bash.bashrc     

# CMD [ "/bin/bash" ]