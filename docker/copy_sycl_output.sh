#!/usr/bin/env bash
#
# Copy SYCL-converted files back to the Gadgetron source tree
#
# Usage:
#     ./docker/copy_sycl_output.sh [--sycl-root /path/to/sycl_output] [--repo-root /path/to/gadgetron]
#     ./docker/copy_sycl_output.sh --dry-run

set -euo pipefail

# Default values
SYCL_ROOT=""
REPO_ROOT=""
DRY_RUN=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

print_usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Copy SYCL-converted files from sycl_output back to the Gadgetron source tree.

Options:
    --sycl-root DIR    Source directory with converted files (default: /install/sycl_output)
    --repo-root DIR    Gadgetron repository root (default: auto-detect)
    --dry-run          Show what would be copied without copying
    -h, --help         Show this help message

Examples:
    $(basename "$0")
    $(basename "$0") --sycl-root /build/sycl_output --repo-root /gadgetron
    $(basename "$0") --dry-run
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sycl-root)
                SYCL_ROOT="$2"
                shift 2
                ;;
            --repo-root)
                REPO_ROOT="$2"
                shift 2
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

    # Determine sycl output root
    if [[ -z "$SYCL_ROOT" ]]; then
        SYCL_ROOT="/install/sycl_output"
    fi

    if [[ ! -d "$SYCL_ROOT" ]]; then
        log_error "SYCL output directory not found: $SYCL_ROOT"
        exit 1
    fi

    SYCL_ROOT="$(realpath "$SYCL_ROOT")"

    log_info "SYCL output root: $SYCL_ROOT"
    log_info "Repository root: $REPO_ROOT"

    # Find all .dp.cpp and .dp.hpp files
    local files=()
    mapfile -t files < <(find "$SYCL_ROOT" -type f \( -name '*.dp.cpp' -o -name '*.dp.hpp' -o -name '*.h' \) 2>/dev/null | sort)

    if [[ ${#files[@]} -eq 0 ]]; then
        log_warn "No converted files found in $SYCL_ROOT"
        exit 0
    fi

    log_info "Found ${#files[@]} file(s) to copy"

    local copied=0
    local skipped=0

    for src in "${files[@]}"; do
        # Get relative path from sycl root
        local rel_path="${src#$SYCL_ROOT/}"
        local dest="$REPO_ROOT/$rel_path"
        local dest_dir="$(dirname "$dest")"

        # Show what we're doing
        echo "  $rel_path"

        if [[ "$DRY_RUN" == "true" ]]; then
            copied=$((copied + 1))
            continue
        fi

        # Create destination directory if needed
        if [[ ! -d "$dest_dir" ]]; then
            mkdir -p "$dest_dir"
        fi

        # Copy the file
        if cp "$src" "$dest"; then
            copied=$((copied + 1))
        else
            log_error "Failed to copy: $rel_path"
            skipped=$((skipped + 1))
        fi
    done

    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run: would copy $copied file(s)"
    else
        log_info "Copied $copied file(s), skipped $skipped"
    fi
}

main "$@"
