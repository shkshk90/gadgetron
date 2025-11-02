# Getting started

## Environment

```sh
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
```

## Launch docker

`docker compose run --rm thesis /bin/bash`

## From inside docker

```sh
# cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -S gadgetron -B build/cmake-build -DUSE_MKL=ON -DUSE_CUDA=ON -DCMAKE_INSTALLATION_PREFIX=/build/cmake-install

cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -S /gadgetron -B /build/cmake-build -DUSE_MKL=ON -DMKL_THREADING="intel_thread" -DUSE_CUDA=ON -DCMAKE_INSTALLATION_PREFIX=/build/cmake-install
cmake --build build/cmake-build -j 8
cmake --install build/cmake-build --prefix /build/cmake-install
```

## Start Running

```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/build/cmake-install/lib
cd /build/cmake-install/bin
./gadgetron
```

# Useful Links

## OneMLKL has `cuBLAS` backend disabled by default, and how to fix:

link to `llama.cpp` documentation:

```
https://gitlab.informatik.uni-halle.de/ambcj/llama.cpp/-/blob/52604860f93063ef98863921da697576af1c7665/README-sycl.md
```

Effectively, what you want is build OneMKL

```sh
git clone https://github.com/oneapi-src/oneMKL
cd oneMKL
mkdir -p buildWithCublas && cd buildWithCublas
cmake ../ -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DENABLE_MKLGPU_BACKEND=OFF -DENABLE_MKLCPU_BACKEND=OFF -DENABLE_CUBLAS_BACKEND=ON -DTARGET_DOMAINS=blas
make

# then

export LD_LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/path/to/oneMKL/buildWithCublas/lib:$LIBRARY_PATH
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/buildWithCublas/include:$CPLUS_INCLUDE_DIR
export CPLUS_INCLUDE_DIR=/path/to/oneMKL/include:$CPLUS_INCLUDE_DIR

```

## Painfully slow docker ? 

#### Prefer IPv4

In the docker file, use the following to give precedence for IPv4 instead of IPv6:

```Dockerfile
RUN echo 'precedence ::ffff:0:0/96 100' >> /etc/gai.conf
```

### Use good DNS servers in docker daemon config file

Add the entry to the config file, `~/.config/docker/daemon.json` or `/etc/docker/daemon.json`:

```json
{
  "dns": ["1.1.1.1", "8.8.8.8"],
}
```

### Move storage to `fuse-overlayfs`

Add the entry to the config file, `~/.config/docker/daemon.json` or `/etc/docker/daemon.json`:

```json
{
  "storage-driver": "fuse-overlayfs"
}
```
