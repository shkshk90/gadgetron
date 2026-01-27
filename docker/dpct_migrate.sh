#!/usr/bin/env bash
#
# dpct_migrate.sh - Migrate all CUDA code to SYCL using Intel dpct
#
# Converts .cu kernel files (.cu -> .dp.cpp) and patches .cpp/.h test files
# that use CUDA APIs (cudaMemcpy, thrust, etc.) in a single pass.
#
# Run ONCE inside the Docker container before building.
# Usage:  /gadgetron/docker/dpct_migrate.sh
#

set -euo pipefail

GADGETRON_SRC="/gadgetron"
DPCT_OUT="/gadgetron_sycl_migrated"
CUDA_INCLUDE="/usr/local/cuda/include"
LOG="/tmp/dpct_migration.log"

echo "=== Gadgetron CUDA to SYCL Migration ==="
echo ""

# Verify dpct is available
if ! command -v dpct &> /dev/null; then
    echo "ERROR: dpct not found. Make sure Intel oneAPI is installed and setvars.sh is sourced."
    exit 1
fi

echo "Using dpct version:"
dpct --version
echo ""

# Clean previous migration output
rm -rf "${DPCT_OUT}"
mkdir -p "${DPCT_OUT}"

# ============================================================================
# Include directories needed by the CUDA code
# ============================================================================

EXTRA_ARGS=""

INCLUDE_DIRS=(
    "${GADGETRON_SRC}/toolboxes"
    "${GADGETRON_SRC}/toolboxes/core"
    "${GADGETRON_SRC}/toolboxes/core/cpu"
    "${GADGETRON_SRC}/toolboxes/core/gpu"
    "${GADGETRON_SRC}/toolboxes/dwt/gpu"
    "${GADGETRON_SRC}/toolboxes/fft/gpu"
    "${GADGETRON_SRC}/toolboxes/log"
    "${GADGETRON_SRC}/toolboxes/mri/hyper"
    "${GADGETRON_SRC}/toolboxes/mri/pmri/gpu"
    "${GADGETRON_SRC}/toolboxes/mri_core"
    "${GADGETRON_SRC}/toolboxes/nfft"
    "${GADGETRON_SRC}/toolboxes/nfft/gpu"
    "${GADGETRON_SRC}/toolboxes/operators"
    "${GADGETRON_SRC}/toolboxes/operators/gpu"
    "${GADGETRON_SRC}/toolboxes/registration/optical_flow"
    "${GADGETRON_SRC}/toolboxes/registration/optical_flow/gpu"
    "${GADGETRON_SRC}/toolboxes/mri/sdc"
    "${GADGETRON_SRC}/toolboxes/mri/sdc/gpu"
    "${GADGETRON_SRC}/toolboxes/solvers"
    "${GADGETRON_SRC}/toolboxes/solvers/gpu"
    "${GADGETRON_SRC}/toolboxes/image_io"
    "${GADGETRON_SRC}/toolboxes/core/cpu/image"
    "${GADGETRON_SRC}/core"
    "${GADGETRON_SRC}/gadgets/pmri"
    "${GADGETRON_SRC}/gadgets/grappa"
    "${GADGETRON_SRC}/gadgets/grappa/common"
    "${GADGETRON_SRC}/gadgets/grappa/gpu"
    "${GADGETRON_SRC}/gadgets/moco"
    "${GADGETRON_SRC}/apps/gadgetron"
    "${GADGETRON_SRC}/test"
)

EXTERNAL_DIRS=(
    "/boost/include"
    "/ismrmrd/include"
    "/plplot/include"
)

if [ -d "/opt/intel/oneapi/mkl/2025.2/include" ]; then
    EXTERNAL_DIRS+=("/opt/intel/oneapi/mkl/2025.2/include")
fi

for dir in "${INCLUDE_DIRS[@]}"; do
    [ -d "$dir" ] && EXTRA_ARGS="${EXTRA_ARGS} --extra-arg=-I${dir}"
done
for dir in "${EXTERNAL_DIRS[@]}"; do
    [ -d "$dir" ] && EXTRA_ARGS="${EXTRA_ARGS} --extra-arg=-I${dir}"
done

# ============================================================================
# Collect every file that contains CUDA code
# ============================================================================

FILES=()

# 1) All .cu kernel files under toolboxes/ and test/
while IFS= read -r -d '' f; do
    FILES+=("$f")
done < <(find "${GADGETRON_SRC}/toolboxes" "${GADGETRON_SRC}/test" \
              -name "*.cu" -print0 2>/dev/null)

# 2) Test .cpp/.h files that directly use CUDA APIs
TEST_CUDA_FILES=(
    "${GADGETRON_SRC}/test/cuNDArray_elemwise_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_blas_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_operators_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_utils_test.cpp"
    "${GADGETRON_SRC}/test/cuNDFFT_test.cpp"
    "${GADGETRON_SRC}/test/cuNFFT_test.cpp"
    "${GADGETRON_SRC}/test/cuSDC_test.cpp"
    "${GADGETRON_SRC}/test/cuVector_td_test_kernels.h"
    "${GADGETRON_SRC}/toolboxes/core/gpu/CUBLASContextProvider.h"
    "${GADGETRON_SRC}/toolboxes/core/gpu/CUBLASContextProvider.cpp"
    "${GADGETRON_SRC}/toolboxes/core/gpu/cudaDeviceManager.cpp"
    "${GADGETRON_SRC}/toolboxes/core/gpu/cudaDeviceManager.h"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuFFTCachedPlan.h"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuFFTCachedPlan.hpp"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuFFTPlan.h"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuFFTPlan.hpp"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuNDFFT.cpp"
    "${GADGETRON_SRC}/toolboxes/fft/gpu/cuNDFFT.h"
)

for f in "${TEST_CUDA_FILES[@]}"; do
    [ -f "$f" ] && FILES+=("$f")
done

echo "Files to migrate: ${#FILES[@]}"
printf '  %s\n' "${FILES[@]}"
echo ""

# ============================================================================
# Run dpct
# ============================================================================

echo "Running dpct..."
echo "  in-root:  ${GADGETRON_SRC}"
echo "  out-root: ${DPCT_OUT}"
echo ""

dpct \
    --in-root="${GADGETRON_SRC}" \
    --out-root="${DPCT_OUT}" \
    --cuda-include-path="${CUDA_INCLUDE}" \
    --extra-arg="-std=c++17" \
    --extra-arg="-DUSE_CUDA" \
    --extra-arg="-DUSE_MKL" \
    ${EXTRA_ARGS} \
    --gen-helper-function \
    --sycl-named-lambda \
    --process-all \
    --keep-original-code \
    "${FILES[@]}" \
    2>&1 | tee "${LOG}" # || true


echo ""

# ============================================================================
# Copy migrated output back to source tree
# ============================================================================

echo "Copying migrated files back..."

for subdir in include toolboxes test gadgets apps core; do
    src="${DPCT_OUT}/${subdir}"
    if [ -d "$src" ]; then
        echo "  ${subdir}/"
        cp -r "$src/" "${GADGETRON_SRC}/${subdir}/"
    fi
done

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "=== Migration Summary ==="
echo ""
echo "Generated .dp.cpp files:"
find "${GADGETRON_SRC}/toolboxes" "${GADGETRON_SRC}/test" \
     -name "*.dp.cpp" 2>/dev/null | sort
echo ""
echo "dpct helper headers:"
find "${GADGETRON_SRC}/include" -name "*.hpp" -o -name "*.h" 2>/dev/null \
     | sort || echo "  (none)"
echo ""
echo "Full log: ${LOG}"
echo ""
echo "Next steps:"
echo "  1. Review warnings in ${LOG}"
echo "  2. Configure: /config.sh"
echo "  3. Build:     /run.sh"
echo ""
