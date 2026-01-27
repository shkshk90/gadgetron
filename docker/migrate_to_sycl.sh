#!/usr/bin/env bash
#
# migrate_to_sycl.sh - Migrate all CUDA code to SYCL using Intel dpct
#
# Run this script ONCE inside the Docker container before building with SYCL.
# Usage:  /migrate_to_sycl.sh
#
# Prerequisites:
#   - Intel oneAPI toolkit installed (with dpct)
#   - CUDA toolkit installed (dpct needs CUDA headers to parse)
#   - The gadgetron source tree mounted at /gadgetron
#

set -euo pipefail

GADGETRON_SRC="/gadgetron"
DPCT_OUT="/gadgetron_sycl_migrated"
CUDA_INCLUDE="/usr/local/cuda/include"

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
# Gather all include directories needed by the CUDA code
# ============================================================================

EXTRA_ARGS=""

# Gadgetron internal include directories
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
    "${GADGETRON_SRC}/gadgets/pmri"
    "${GADGETRON_SRC}/gadgets/grappa"
    "${GADGETRON_SRC}/gadgets/grappa/common"
    "${GADGETRON_SRC}/gadgets/grappa/gpu"
    "${GADGETRON_SRC}/gadgets/moco"
    "${GADGETRON_SRC}/apps/gadgetron"
    "${GADGETRON_SRC}/test"
)

# External include directories
EXTERNAL_DIRS=(
    "/boost/include"
    "/ismrmrd/include"
    "/plplot/include"
)

# Add MKL if available
if [ -d "/opt/intel/oneapi/mkl/2025.2/include" ]; then
    EXTERNAL_DIRS+=("/opt/intel/oneapi/mkl/2025.2/include")
fi

# Build extra-arg flags
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
# Collect all CUDA source files to migrate
# ============================================================================

echo "Collecting CUDA source files..."

CUDA_FILES=()

# All .cu files under toolboxes/
while IFS= read -r -d '' file; do
    CUDA_FILES+=("$file")
done < <(find "${GADGETRON_SRC}/toolboxes" -name "*.cu" -print0 2>/dev/null)

# Test .cu files
while IFS= read -r -d '' file; do
    CUDA_FILES+=("$file")
done < <(find "${GADGETRON_SRC}/test" -name "*.cu" -print0 2>/dev/null)

echo "Found ${#CUDA_FILES[@]} CUDA source files to migrate."
echo ""

# ============================================================================
# Run dpct migration
# ============================================================================

echo "Running dpct migration..."
echo "  Input root:  ${GADGETRON_SRC}"
echo "  Output root: ${DPCT_OUT}"
echo ""

# Run dpct on the entire source tree
# --process-all: process all CUDA files found under in-root
# --gen-helper-function: generate dpct helper functions
# --use-experimental-features=all: enable all experimental migration features
# --sycl-named-lambda: generate named lambda for SYCL kernels
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
    "${CUDA_FILES[@]}" \
    2>&1 | tee /tmp/dpct_migration.log || true

echo ""
echo "dpct migration completed. Output in: ${DPCT_OUT}"
echo ""

# ============================================================================
# Copy migrated files back to source tree
# ============================================================================

echo "Copying migrated files back to source tree..."

# Copy the dpct-generated helper headers
if [ -d "${DPCT_OUT}/include" ]; then
    echo "  Copying dpct helper headers to ${GADGETRON_SRC}/include/"
    cp -r "${DPCT_OUT}/include/" "${GADGETRON_SRC}/include/"
fi

# Copy migrated toolboxes
if [ -d "${DPCT_OUT}/toolboxes" ]; then
    echo "  Copying migrated toolboxes/"
    cp -r "${DPCT_OUT}/toolboxes/" "${GADGETRON_SRC}/toolboxes/"
fi

# Copy migrated test files
if [ -d "${DPCT_OUT}/test" ]; then
    echo "  Copying migrated test/"
    cp -r "${DPCT_OUT}/test/" "${GADGETRON_SRC}/test/"
fi

# Copy migrated gadgets
if [ -d "${DPCT_OUT}/gadgets" ]; then
    echo "  Copying migrated gadgets/"
    cp -r "${DPCT_OUT}/gadgets/" "${GADGETRON_SRC}/gadgets/"
fi

# Copy migrated apps
if [ -d "${DPCT_OUT}/apps" ]; then
    echo "  Copying migrated apps/"
    cp -r "${DPCT_OUT}/apps/" "${GADGETRON_SRC}/apps/"
fi

# Copy migrated core
if [ -d "${DPCT_OUT}/core" ]; then
    echo "  Copying migrated core/"
    cp -r "${DPCT_OUT}/core/" "${GADGETRON_SRC}/core/"
fi

echo ""

# ============================================================================
# Post-migration: list generated files
# ============================================================================

echo "=== Migration Summary ==="
echo ""
echo "Generated .dp.cpp files:"
find "${GADGETRON_SRC}/toolboxes" "${GADGETRON_SRC}/test" -name "*.dp.cpp" 2>/dev/null | sort
echo ""

echo "Generated dpct helper headers:"
find "${GADGETRON_SRC}/include" -name "*.hpp" -o -name "*.h" 2>/dev/null | sort || echo "  (none found)"
echo ""

echo "dpct migration log saved to: /tmp/dpct_migration.log"
echo ""
echo "=== Migration Complete ==="
echo ""
echo "Next steps:"
echo "  1. Review the migration log for any warnings: /tmp/dpct_migration.log"
echo "  2. Configure with: /config.sh"
echo "  3. Build with:     /run.sh"
echo ""
