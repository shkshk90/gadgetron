#!/usr/bin/env bash
#
# Gadgetron CUDA to SYCL Conversion Script
#
# Converts .cpp and .cu files to SYCL using Intel's DPC++ Compatibility Tool (dpct).
#
# Usage:
#     # Convert using compilation database (recommended)
#     ./docker/convert_to_sycl.sh --build-dir /build/clang
#
#     # Convert specific files
#     ./docker/convert_to_sycl.sh --files file1.cu file2.cpp
#
#     # Convert amalgamated files only
#     ./docker/convert_to_sycl.sh --amalgamated-only
#
# Requirements:
#     - Intel DPC++ Compatibility Tool (dpct) must be in PATH
#     - For database mode: compile_commands.json must exist in build directory

set -euo pipefail

# Default values
REPO_ROOT=""
OUTPUT_ROOT=""
BUILD_DIR=""
AMALGAMATED_ONLY=false
DRY_RUN=false
DPCT_PATH=""
FILES=()
EXTRA_ARGS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Convert CUDA code to SYCL using Intel dpct.

Options:
    --repo-root DIR       Repository root (default: auto-detect)
    --output-root DIR     Output directory (default: <repo>/sycl_output)
    --build-dir DIR       Build directory with compile_commands.json
    --files FILE...       Specific files to convert
    --amalgamated-only    Only convert *_amalgamated.cu/cpp files
    --dpct-path PATH      Path to dpct executable
    --extra-args ARGS     Extra arguments to pass to dpct
    --dry-run             Show what would be converted
    -h, --help            Show this help message

Examples:
    $(basename "$0") --build-dir /build/clang
    $(basename "$0") --amalgamated-only --dry-run
    $(basename "$0") --files test/tests_amalgamated.cu
EOF
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Find dpct executable
find_dpct() {
    if [[ -n "$DPCT_PATH" ]] && [[ -x "$DPCT_PATH" ]]; then
        echo "$DPCT_PATH"
        return 0
    fi

    # Check PATH
    if command -v dpct &>/dev/null; then
        command -v dpct
        return 0
    fi

    # Check common Intel oneAPI locations
    local common_paths=(
        "/opt/intel/oneapi/dpcpp-ct/latest/bin/dpct"
        "$HOME/intel/oneapi/dpcpp-ct/latest/bin/dpct"
        "/opt/intel/oneapi/2024.0/bin/dpct"
        "/opt/intel/oneapi/2025.0/bin/dpct"
    )

    for path in "${common_paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    return 1
}

# Find CUDA files in repository
find_cuda_files() {
    local repo="$1"
    local amalgamated_only="$2"

    if [[ "$amalgamated_only" == "true" ]]; then
        find "$repo" -type f \( -name '*_amalgamated.cu' -o -name '*_amalgamated.cpp' \) \
            ! -path '*/build/*' ! -path '*/.git/*' ! -path '*/sycl_output/*' \
            2>/dev/null | sort
    else
        find "$repo" -type f -name '*.cu' \
            ! -path '*/build/*' ! -path '*/.git/*' ! -path '*/sycl_output/*' \
            2>/dev/null | sort
    fi
}

# Convert a single file using dpct
convert_file() {
    local dpct="$1"
    local source_file="$2"
    local repo_root="$3"
    local output_root="$4"
    shift 4
    local extra_args=("$@")

    local rel_path="${source_file#$repo_root/}"
    local output_file="$output_root/$rel_path"

    # Change extension
    if [[ "$output_file" == *.cu ]]; then
        output_file="${output_file%.cu}.dp.cpp"
    elif [[ "$output_file" == *.cpp ]]; then
        output_file="${output_file%.cpp}.dp.cpp"
    fi

    # Create output directory
    mkdir -p "$(dirname "$output_file")"

    local cmd=(
        "$dpct"
        "--in-root=$repo_root"
        "--out-root=$output_root"
        "--cuda-include-path=/usr/local/cuda/include"
        "--gen-helper-function"
        "--sycl-named-lambda"
        "--use-experimental-features=all"
        "--use-dpcpp-extensions=intel_device_math"
        "--comments"
        "--keep-original-code"
        '--extra-arg=-DUSE_MKL'
        '--extra-arg=-DUSE_CUDA'
        '--extra-arg=-std=c++20'
        '--extra-arg=-I/boost/include'
        '--extra-arg=-I/ismrmrd/include'
        '--extra-arg=-I/plplot/include'
        '--extra-arg=-I/opt/intel/oneapi/mkl/latest/include'
        '--extra-arg=-I/usr/local/cuda/include'
        '--extra-arg=-I/usr/include/hdf5/serial'
    )

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    cmd+=("$source_file")

    log_info "Running: ${cmd[*]}"
    if "${cmd[@]}" 2>&1; then
        echo -e "  ${GREEN}[OK]${NC} $rel_path"
        return 0
    else
        echo -e "  ${RED}[FAILED]${NC} $rel_path"
        return 1
    fi
}

# Convert using compilation database
convert_with_database() {
    local dpct="$1"
    local build_dir="$2"
    local repo_root="$3"
    local output_root="$4"
    shift 4
    local extra_args=("$@")

    local compile_db="$build_dir/compile_commands.json"
    if [[ ! -f "$compile_db" ]]; then
        log_error "compile_commands.json not found in $build_dir"
        log_error "Run CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON first."
        return 1
    fi

    local cmd=(
        "$dpct"
        "-p=$build_dir"
        "--in-root=$repo_root"
        "--out-root=$output_root"
        "--cuda-include-path=/usr/local/cuda/include"
        "--gen-helper-function"
        "--sycl-named-lambda"
        "--use-experimental-features=all"
        "--use-dpcpp-extensions=intel_device_math"
        "--comments"
        "--keep-original-code"
        '--extra-arg=-DUSE_MKL'
        '--extra-arg=-DUSE_CUDA'
        '--extra-arg=-std=c++20'
        '--extra-arg=-I/boost/include'
        '--extra-arg=-I/ismrmrd/include'
        '--extra-arg=-I/plplot/include'
        '--extra-arg=-I/opt/intel/oneapi/mkl/latest/include'
        '--extra-arg=-I/usr/local/cuda/include'
        '--extra-arg=-I/usr/include/hdf5/serial'
        "--process-all"
    )

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    log_info "Running dpct with compilation database..."
    log_info "Command: ${cmd[*]}"

    "${cmd[@]}"
}

# Generate SYCL CMake configuration
generate_sycl_cmake() {
    local output_root="$1"

    cat > "$output_root/SYCLConfig.cmake" <<'EOF'
# SYCL Build Configuration
# Auto-generated by docker/convert_to_sycl.sh
# Include this file when building with SYCL instead of CUDA

# Find SYCL compiler
find_package(IntelDPCPP QUIET)

if (NOT IntelDPCPP_FOUND)
    find_program(ICPX_COMPILER icpx)
    if (ICPX_COMPILER)
        set(CMAKE_CXX_COMPILER ${ICPX_COMPILER})
    endif()
endif()

# SYCL compile flags
set(SYCL_FLAGS "-fsycl")
if (EXISTS "/usr/local/cuda")
    set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=spir64,nvptx64-nvidia-cuda")
else()
    set(SYCL_FLAGS "${SYCL_FLAGS} -fsycl-targets=spir64")
endif()

# Add SYCL definitions
add_definitions(-DUSE_SYCL)
add_definitions(-DSYCL_LANGUAGE_VERSION=2020)

# Helper function to add SYCL executable
function(add_sycl_executable target)
    add_executable(${target} ${ARGN})
    target_compile_options(${target} PRIVATE ${SYCL_FLAGS})
    target_link_options(${target} PRIVATE ${SYCL_FLAGS})
endfunction()

# Helper function to add SYCL library
function(add_sycl_library target)
    add_library(${target} ${ARGN})
    target_compile_options(${target} PRIVATE ${SYCL_FLAGS})
    target_link_options(${target} PRIVATE ${SYCL_FLAGS})
endfunction()

# oneDPL for parallel algorithms (replaces Thrust)
find_package(oneDPL QUIET)

# oneMKL for FFT and BLAS
find_package(MKL QUIET COMPONENTS sycl)
EOF

    log_info "Generated SYCL CMake config: $output_root/SYCLConfig.cmake"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo-root)
                REPO_ROOT="$2"
                shift 2
                ;;
            --output-root)
                OUTPUT_ROOT="$2"
                shift 2
                ;;
            --build-dir)
                BUILD_DIR="$2"
                shift 2
                ;;
            --files)
                shift
                while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                    FILES+=("$1")
                    shift
                done
                ;;
            --amalgamated-only)
                AMALGAMATED_ONLY=true
                shift
                ;;
            --dpct-path)
                DPCT_PATH="$2"
                shift 2
                ;;
            --extra-args)
                shift
                while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                    EXTRA_ARGS+=("$1")
                    shift
                done
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

main() {
    parse_args "$@"

    # Determine repo root
    if [[ -z "$REPO_ROOT" ]]; then
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
    REPO_ROOT="$(realpath "$REPO_ROOT")"

    # Determine output root
    if [[ -z "$OUTPUT_ROOT" ]]; then
        OUTPUT_ROOT="$REPO_ROOT/sycl_output"
    fi
    OUTPUT_ROOT="$(realpath -m "$OUTPUT_ROOT")"

    log_info "Repository root: $REPO_ROOT"
    log_info "Output root: $OUTPUT_ROOT"

    # Find dpct
    if ! DPCT_PATH="$(find_dpct)"; then
        log_error "dpct not found. Please install Intel DPC++ Compatibility Tool"
        log_error "or specify path with --dpct-path"
        echo ""
        echo "Install with:"
        echo "  apt-get install intel-oneapi-dpcpp-ct"
        echo "  # or"
        echo "  source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi

    log_info "Using dpct: $DPCT_PATH"

    # Create output directory
    mkdir -p "$OUTPUT_ROOT"

    # Mode 1: Use compilation database
    if [[ -n "$BUILD_DIR" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "Would convert using database from: $BUILD_DIR"
            exit 0
        fi

        if convert_with_database "$DPCT_PATH" "$BUILD_DIR" "$REPO_ROOT" "$OUTPUT_ROOT" "${EXTRA_ARGS[@]}"; then
            generate_sycl_cmake "$OUTPUT_ROOT"
            log_info "Conversion complete!"
            exit 0
        else
            log_error "Conversion failed"
            exit 1
        fi
    fi

    # Mode 2: Convert specific files or discover files
    local cuda_files=()
    if [[ ${#FILES[@]} -gt 0 ]]; then
        for f in "${FILES[@]}"; do
            cuda_files+=("$(realpath "$f")")
        done
    else
        mapfile -t cuda_files < <(find_cuda_files "$REPO_ROOT" "$AMALGAMATED_ONLY")
    fi

    if [[ ${#cuda_files[@]} -eq 0 ]]; then
        log_warn "No CUDA files found to convert."
        exit 0
    fi

    log_info "Found ${#cuda_files[@]} file(s) to convert:"
    local count=0
    for f in "${cuda_files[@]}"; do
        if [[ $count -lt 20 ]]; then
            echo "  ${f#$REPO_ROOT/}"
        fi
        count=$((count + 1))
    done
    if [[ $count -gt 20 ]]; then
        echo "  ... and $((count - 20)) more"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run - no conversion performed."
        exit 0
    fi

    # Convert files
    log_info "Converting files..."
    local success=0
    local failed=0

    for f in "${cuda_files[@]}"; do
        if convert_file "$DPCT_PATH" "$f" "$REPO_ROOT" "$OUTPUT_ROOT" "${EXTRA_ARGS[@]}"; then
            success=$((success + 1))
        else
            failed=$((failed + 1))
        fi
    done

    log_info "Conversion complete: $success succeeded, $failed failed"

    if [[ $success -gt 0 ]]; then
        generate_sycl_cmake "$OUTPUT_ROOT"
    fi

    [[ $failed -eq 0 ]]
}

main "$@"
