#!/usr/bin/env bash
#
# migrate_tests_to_sycl.sh - Migrate CUDA test files to SYCL using Intel dpct
#
# Run this script inside the Docker container AFTER running migrate_to_sycl.sh.
# It targets only the test/ .cpp and .h files that contain CUDA API calls.
#
# Usage:  /gadgetron/docker/migrate_tests_to_sycl.sh
#

set -euo pipefail

GADGETRON_SRC="/gadgetron"
DPCT_OUT="/gadgetron_tests_sycl_migrated"
CUDA_INCLUDE="/usr/local/cuda/include"

echo "=== Gadgetron Test Files: CUDA to SYCL Migration ==="
echo ""

# Verify dpct is available
if ! command -v dpct &> /dev/null; then
    echo "ERROR: dpct not found. Make sure Intel oneAPI is installed and setvars.sh is sourced."
    exit 1
fi

echo "Using dpct version:"
dpct --version
echo ""

# Clean previous output
rm -rf "${DPCT_OUT}"
mkdir -p "${DPCT_OUT}"

# ============================================================================
# Include directories needed by the test CUDA code
# ============================================================================

EXTRA_ARGS=""

INCLUDE_DIRS=(
    "${GADGETRON_SRC}/toolboxes"
    "${GADGETRON_SRC}/toolboxes/core"
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
    "${GADGETRON_SRC}/core"
    "${GADGETRON_SRC}/test"
)

EXTERNAL_DIRS=(
    "/boost/include"
    "/ismrmrd/include"
)

if [ -d "/opt/intel/oneapi/mkl/2025.2/include" ]; then
    EXTERNAL_DIRS+=("/opt/intel/oneapi/mkl/2025.2/include")
fi

for dir in "${INCLUDE_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --extra-arg=-I${dir}"
    fi
done
for dir in "${EXTERNAL_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --extra-arg=-I${dir}"
    fi
done

# ============================================================================
# List the specific test files that contain CUDA code
# ============================================================================

TEST_FILES=(
    "${GADGETRON_SRC}/test/cuNDArray_elemwise_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_blas_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_operators_test.cpp"
    "${GADGETRON_SRC}/test/cuNDArray_utils_test.cpp"
    "${GADGETRON_SRC}/test/cuNDFFT_test.cpp"
    "${GADGETRON_SRC}/test/cuNFFT_test.cpp"
    "${GADGETRON_SRC}/test/cuSDC_test.cpp"
    "${GADGETRON_SRC}/test/cuVector_td_test_kernels.h"
)

# Filter to only files that actually exist
EXISTING_FILES=()
for f in "${TEST_FILES[@]}"; do
    if [ -f "$f" ]; then
        EXISTING_FILES+=("$f")
    else
        echo "  SKIP (not found): $f"
    fi
done

echo "Migrating ${#EXISTING_FILES[@]} test files..."
for f in "${EXISTING_FILES[@]}"; do
    echo "  $f"
done
echo ""

# ============================================================================
# Run dpct
# ============================================================================

echo "Running dpct..."

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
    --keep-original-code \
    "${EXISTING_FILES[@]}" \
    2>&1 | tee /tmp/dpct_tests_migration.log || true

echo ""
echo "dpct finished. Copying migrated files back..."

# ============================================================================
# Copy only the migrated test files back (not the whole tree)
# ============================================================================

for f in "${EXISTING_FILES[@]}"; do
    # Compute relative path from GADGETRON_SRC
    rel="${f#${GADGETRON_SRC}/}"
    migrated="${DPCT_OUT}/${rel}"
    if [ -f "$migrated" ]; then
        cp -v "$migrated" "$f"
    else
        echo "  WARNING: no migrated output for $rel"
    fi
done

# Also copy dpct helper headers if they were generated and don't exist yet
if [ -d "${DPCT_OUT}/include/dpct" ] && [ ! -d "${GADGETRON_SRC}/include/dpct" ]; then
    echo "Copying dpct helper headers..."
    mkdir -p "${GADGETRON_SRC}/include"
    cp -r "${DPCT_OUT}/include/" "${GADGETRON_SRC}/include/"
fi

echo ""
echo "=== Test Migration Complete ==="
echo "Log: /tmp/dpct_tests_migration.log"
echo ""
