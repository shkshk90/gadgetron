# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gadgetron is an open-source medical image reconstruction framework (v4.7.2). It uses a message-passing pipeline architecture where data flows through configurable chains of "gadgets" (processing nodes). The project is C++20 and primarily targets MRI reconstruction with both CPU and GPU (CUDA) acceleration.

## Build Commands

The project uses CMake with Ninja. Builds are typically done inside Docker containers due to extensive dependencies.

```bash
# Configure (from Docker container, see docker/cmake.config.sh for reference)
cmake -S /gadgetron -B /build -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DUSE_CUDA=ON \
  -DUSE_MKL=ON \
  -DBUILD_SUPPRESS_WARNINGS=TRUE \
  -DBUILD_TESTING=ON \
  # ... plus dependency paths (Boost, ISMRMRD, HDF5, etc.)

# Build
ninja -C /build

# Run all tests (GTest-based, single executable)
cd /build && ctest

# Run a single test by name
cd /build && ctest -R <test_name>

# Or run test binary directly with GTest filter
/build/test/test_all --gtest_filter="<TestSuite>.<TestName>"
```

Key CMake options: `USE_CUDA` (ON), `USE_MKL` (OFF), `BUILD_PYTHON_SUPPORT` (ON), `BUILD_TESTING` (ON), `BUILD_SUPPRESS_WARNINGS` (OFF), `GADGETRON_CUDA_ALL_COMPUTE_MODEL` (OFF).

## Architecture

### Layer Structure

1. **core/** - Framework infrastructure: message types, channels (thread-safe MPMC queues), Node/Gadget base classes, property system, readers/writers, parallel/distributed execution
2. **toolboxes/** - Algorithmic building blocks (CPU and GPU implementations): FFT, NFFT, wavelets, solvers, operators, registration, denoising, MRI-specific algorithms
3. **gadgets/** - High-level reconstruction modules composed from toolboxes: MRI core, cartesian, EPI, spiral, GRAPPA, cardiac MRI, etc.
4. **apps/** - Executables: main gadgetron server, ISMRMRD client, standalone algorithm runners

### Key Patterns

**Gadget hierarchy** (`core/Node.h`, `core/Gadget.h`):
- `ChannelGadget<TYPELIST...>` - typed input channel, generic output; override `process(InputChannel&, OutputChannel&)`
- `PureGadget<RETURN, INPUT>` - 1-to-1 typed transformation; override `process_function(INPUT)`
- Gadgets are dynamically loaded via `GADGETRON_GADGET_EXPORT()` macro and Boost.DLL

**Data types** (`core/Types.h`):
- `Acquisition` = (ISMRMRD header, hoNDArray k-space data, optional trajectory)
- `Waveform` = (header, waveform data)
- `Image<T>` = (header, hoNDArray image data, optional metadata)

**Array types**:
- `hoNDArray<T>` - host (CPU) N-dimensional array (in `toolboxes/core/`)
- `cuNDArray<T>` - device (GPU) N-dimensional array (in `toolboxes/core/gpu/`)
- Operators and solvers are templated to work with both

**GPU toolbox compilation**: All GPU `.cu` files are compiled into a single `gadgetron_toolbox_gpu` shared library to avoid Thrust/CUB static template symbol conflicts across multiple libraries.

**Toolbox build order** (dependency chain in `toolboxes/CMakeLists.txt`):
log → operators, solvers → fft, core, mri_core → mri_image → klt, fatwater, deblur, nfft, dwt, registration → ffd, image, pattern_recognition, denoise → image_io, T1 → python → plplot

### Server Architecture (apps/gadgetron/)

The server accepts client connections, parses XML pipeline configurations, dynamically loads gadgets, and constructs processing streams. Supports distributed execution (Pool/Worker pattern) and external language bindings (Python, Julia, MATLAB) through `connection/nodes/External.cpp`.

### Current Development

The `amalgamated` branch contains work on SYCL migration (from CUDA) for GPU vendor portability. The `docker/amalgamate.py` script generates single-compilation-unit `*_amalgamated.cpp` files.

## Build Notes

- `-Werror` is enabled on non-Windows builds; warnings must be fixed
- ccache is automatically used if available
- OpenMP is enabled by default
- Boost >= 1.80.0 required (coroutine, system, date_time, program_options, filesystem, timer)
- ISMRMRD, Armadillo, FFTW3, CURL, PugiXML are required dependencies
- CUDA, PyTorch, Python, PLplot, OpenGL/Qt4, BART are optional
