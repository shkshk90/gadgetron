#!/usr/bin/env python3
"""
Gadgetron Amalgamation Script

Discovers executable targets from CMakeLists.txt files and creates amalgamated
source files. Each amalgamated file is self-contained: all repo headers are
expanded inline, all corresponding .cpp/.cu implementation files are included,
and only standard/external #include directives remain.

Usage:
    python3 docker/amalgamate.py [--repo-root /path/to/gadgetron]
"""

import os
import re
import sys
import argparse
from pathlib import Path
from collections import OrderedDict, defaultdict


SKIP_DIRS = frozenset({
    'build', '.git', 'bin', '.devcontainer', '.github', 'doc',
    'conda', 'docker', 'chroot',
})

# For target discovery, we include 'test' so test executables are found
SKIP_DIRS_TARGET_DISCOVERY = frozenset({
    'build', '.git', 'bin', '.devcontainer', '.github', 'doc',
    'conda', 'docker', 'chroot',
})

# Generated header templates and their default content
GENERATED_HEADERS = {
    'core_defines.h': """\
#pragma once
#define GADGETRON_CUDA_IS_AVAILABLE 0
#define GADGETRON_SYCL_IS_AVAILABLE 0
#ifndef __CUDACC__
#if !defined(__host__)
#define __host__
#endif
#if !defined(__device__)
#define __device__
#endif
#define __inline__ inline
#endif
""",
    'gadgetron_config.h': """\
#ifndef GADGETRON_CONFIG_H
#define GADGETRON_CONFIG_H
#define GADGETRON_VERSION_MAJOR 4
#define GADGETRON_VERSION_MINOR 7
#define GADGETRON_VERSION_PATCH 2
#define GADGETRON_VERSION_STRING "4.7.2"
#define GADGETRON_CONFIG_PATH "share/gadgetron/config"
#define GADGETRON_PYTHON_PATH "share/gadgetron/python"
#define GADGETRON_GIT_SHA1_HASH "amalgamated"
#define GADGETRON_CUDA_NVCC_FLAGS ""
#define GADGETRON_VAR_DIR "/var/lib/gadgetron/"
#endif
""",
    'gadgetron_sha1.h': """\
#pragma once
#define GADGETRON_SHA1 "amalgamated"
""",
}

# Targets that can't be auto-discovered (e.g., use CMake variables for sources).
# Format: { 'target_name': { 'cmake_dir': 'relative/path', 'sources': [...], 'is_cuda': bool } }
MANUAL_TARGETS = {
    'test_all': {
        'cmake_dir': 'test',
        'output_name': 'tests_amalgamated',
        'is_cuda': True,
        'sources': [
            'test/tests.cpp',
            # 'test/hoNDArray_elemwise_test.cpp',
            # 'test/hoNDArray_blas_test.cpp',
            # 'test/hoNDArray_utils_test.cpp',
            # 'test/hoNDArray_reductions_test.cpp',
            # 'test/read_writer_test.cpp',
            # 'test/hoNDFFT_test.cpp',
            # 'test/hoNFFT_test.cpp',
            # 'test/hoNDWavelet_test.cpp',
            # 'test/curveFitting_test.cpp',
            # 'test/image_morphology_test.cpp',
            # 'test/IsmrmrdContextVariables_test.cpp',
            # 'test/StorageSpaces_test.cpp',
            # 'test/pattern_recognition_test.cpp',
            # 'test/cmr_mapping_test.cpp',
            # 'test/hoNDArray_linalg_test.cpp',
            # 'test/core_test.cpp',
            # 'test/core_primitive_io_test.cpp',
            # 'test/threadpool_test.cpp',
            # 'test/from_string_test.cpp',
            # 'test/hoNDArrayView_test.cpp',
            # 'test/ChannelAlgorithmsTest.cpp',
            # 'test/cmr_strain_test.cpp',
            # 'test/cmr_thickening_test.cpp',
            # 'test/cmr_analytical_strain_test.cpp',
            # 'test/hoSDC_test.cpp',
            # 'test/nhlbi_compression_tests.cpp',
            # 'test/mri_core_stream_test.cpp',
            # 'test/gadgets/setup_gadget.h',
            # 'test/gadgets/AcquisitionAccumulateTrigget_test.cpp',
            # 'test/gadgets/FlagTriggerParsing_test.cpp',
            # CUDA tests
            'test/cuNDArray_elemwise_test.cpp',
            'test/cuNDArray_operators_test.cpp',
            'test/cuNDArray_blas_test.cpp',
            'test/cuNDArray_utils_test.cpp',
            'test/cuVector_td_test_kernels.h',
            'test/cuVector_td_test_kernels.cu',
            'test/cuNDFFT_test.cpp',
            'test/cuSDC_test.cpp',
            'test/cuNFFT_test.cpp',
        ],
    },
}


# ------------------------------------------------------------------
# CMake target discovery
# ------------------------------------------------------------------

def discover_targets(repo_root):
    """
    Walk all CMakeLists.txt files under *repo_root* and extract every
    ``add_executable(name ...)`` and ``cuda_add_executable(name ...)``.

    Returns a sorted list of dicts:
        { 'name': str, 'sources': [str], 'cmake_dir': str, 'is_cuda': bool }
    where *sources* are absolute resolved paths.
    """
    targets = []
    repo = str(Path(repo_root).resolve())

    for root, dirs, files in os.walk(repo):
        # Skip unwanted directories (but include 'test' for target discovery)
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS_TARGET_DISCOVERY)
        if 'CMakeLists.txt' not in files:
            continue

        cmake_path = os.path.join(root, 'CMakeLists.txt')
        try:
            with open(cmake_path, 'r') as fh:
                text = fh.read()
        except OSError:
            continue

        # Strip CMake line comments before searching for targets
        text_no_comments = re.sub(r'#[^\n]*', '', text)

        # Match add_executable(...) and cuda_add_executable(...)
        pattern = re.compile(
            r'(cuda_)?add_executable\s*\(\s*(\w+)(.*?)\)',
            re.DOTALL,
        )
        for m in pattern.finditer(text_no_comments):
            is_cuda = m.group(1) is not None
            target_name = m.group(2)
            body = m.group(3)
            # Strip CMake variable references
            body = re.sub(r'\$\{[^}]*\}', '', body)
            tokens = body.split()
            # Filter out CMake keywords
            keywords = {'SHARED', 'STATIC', 'MODULE', 'WIN32', 'MACOSX_BUNDLE'}
            source_files = []
            for tok in tokens:
                if tok in keywords:
                    continue
                # Skip previously generated amalgamated sources
                if '_amalgamated.' in tok:
                    continue
                if tok.endswith(('.cpp', '.cu', '.c', '.cxx', '.h', '.hpp', '.cuh', '.hxx')):
                    full = os.path.realpath(os.path.join(root, tok))
                    if os.path.isfile(full):
                        source_files.append(full)
            if source_files:
                targets.append({
                    'name': target_name,
                    'sources': source_files,
                    'cmake_dir': root,
                    'is_cuda': is_cuda,
                })

    # Sort by target name for determinism
    targets.sort(key=lambda t: t['name'])
    return targets


# ------------------------------------------------------------------
# Amalgamator class
# ------------------------------------------------------------------

class Amalgamator:
    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()
        self.header_map = defaultdict(list)   # filename -> [full_paths]
        self.source_map = defaultdict(list)   # filename -> [full_paths]
        self.all_headers = set()
        self.all_sources = set()
        self._build_file_index()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_file_index(self):
        """Walk the repo and index every header and source file."""
        for root, dirs, files in os.walk(str(self.repo_root)):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                full = os.path.realpath(os.path.join(root, f))
                if f.endswith(('.h', '.hpp', '.cuh', '.hxx')):
                    self.header_map[f].append(full)
                    self.all_headers.add(full)
                elif f.endswith(('.cpp', '.cu')):
                    self.source_map[f].append(full)
                    self.all_sources.add(full)

    # ------------------------------------------------------------------
    # Include resolution
    # ------------------------------------------------------------------

    def _resolve_include(self, inc_name, current_file):
        """
        Resolve ``#include "inc_name"`` to a real path inside the repo.
        Returns None when the header is not part of the repo.
        """
        cur_dir = os.path.dirname(current_file)

        # 1. Relative to the including file
        rel = os.path.realpath(os.path.join(cur_dir, inc_name))
        if os.path.isfile(rel) and (rel in self.all_headers or rel in self.all_sources):
            return rel

        # 2. By basename
        basename = os.path.basename(inc_name)
        candidates = self.header_map.get(basename, [])
        if not candidates:
            candidates = self.source_map.get(basename, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # 3. Same directory first
        for c in candidates:
            if os.path.dirname(c) == cur_dir:
                return c

        # 4. Match trailing path components
        normed = inc_name.replace('\\', '/')
        for c in candidates:
            if c.replace('\\', '/').endswith(normed):
                return c

        # 5. Best common prefix with current file
        best = candidates[0]
        best_len = 0
        for c in candidates:
            common = len(os.path.commonpath([c, current_file]))
            if common > best_len:
                best_len = common
                best = c
        return best

    # ------------------------------------------------------------------
    # Find implementation files for a header
    # ------------------------------------------------------------------

    # Headers whose implementations live in differently-named source files.
    _EXTRA_IMPL_MAP = {
        'NHLBICompression.h': [
            'CompressedFloatBuffer.cpp',
            'CompressedFloatBufferAvx2.cpp',
            'CompressedFloatBufferSse41.cpp',
        ],
        # NFFTOperator impl files are in cpu/ and gpu/ subdirectories
        'NFFTOperator.h': [
            'hoNFFTOperator.cpp',
            'cuNFFTOperator.cpp',
        ],
        'NFFTOperator.hpp': [
            'hoNFFTOperator.cpp',
            'cuNFFTOperator.cpp',
        ],
    }

    def _find_impl_files(self, header_path):
        """Return list of .cpp/.cu files that implement a given header."""
        base = os.path.splitext(os.path.basename(header_path))[0]
        hdir = os.path.dirname(header_path)
        result = []
        for ext in ('.cpp', '.cu'):
            sname = base + ext
            cands = self.source_map.get(sname, [])
            same = [c for c in cands if os.path.dirname(c) == hdir]
            result.extend(same if same else cands)

        # Check for extra impl files that don't follow the basename convention
        hname = os.path.basename(header_path)
        for extra in self._EXTRA_IMPL_MAP.get(hname, []):
            cands = self.source_map.get(extra, [])
            same = [c for c in cands if os.path.dirname(c) == hdir]
            result.extend(same if same else cands)

        return result

    # ------------------------------------------------------------------
    # Include guard stripping
    # ------------------------------------------------------------------

    # Regex to match #pragma once (with optional whitespace variations)
    _PRAGMA_ONCE_RE = re.compile(r'^#\s*pragma\s+once\s*$')

    @classmethod
    def _strip_include_guards(cls, lines):
        """
        Strip include guards from file lines:
        - Remove all ``#pragma once`` lines
        - Detect ``#ifndef MACRO`` / ``#define MACRO`` at file start +
          matching ``#endif`` at file end → remove those 3 lines
        - Keep all other ``#ifdef``/``#ifndef``/``#endif`` (conditional compilation)
        """
        # Remove #pragma once lines
        filtered = []
        for line in lines:
            stripped = line.strip()
            if cls._PRAGMA_ONCE_RE.match(stripped):
                continue
            filtered.append(line)

        # Detect traditional include guards: #ifndef MACRO at start, #define MACRO next, #endif at end
        if len(filtered) >= 3:
            # Find first non-empty line
            first_idx = None
            for i, line in enumerate(filtered):
                if line.strip():
                    first_idx = i
                    break

            if first_idx is not None and first_idx + 1 < len(filtered):
                first_line = filtered[first_idx].strip()
                second_line = filtered[first_idx + 1].strip()

                ifndef_match = re.match(r'^#\s*ifndef\s+(\w+)$', first_line)
                if ifndef_match:
                    guard_macro = ifndef_match.group(1)
                    define_match = re.match(r'^#\s*define\s+' + re.escape(guard_macro) + r'\s*$', second_line)
                    if define_match:
                        # Find the last #endif
                        last_endif_idx = None
                        for i in range(len(filtered) - 1, -1, -1):
                            if filtered[i].strip().startswith('#endif'):
                                last_endif_idx = i
                                break

                        if last_endif_idx is not None and last_endif_idx > first_idx + 1:
                            # Remove the three guard lines
                            result = []
                            for i, line in enumerate(filtered):
                                if i == first_idx or i == first_idx + 1 or i == last_endif_idx:
                                    continue
                                result.append(line)
                            return result

        return filtered

    # ------------------------------------------------------------------
    # Test typedef deduplication
    # ------------------------------------------------------------------

    # Patterns for test type aliases that conflict when amalgamated
    _TEST_TYPEDEF_RE = re.compile(
        r'^(\s*typedef\s+Types<[^>]*>\s*)(realImplementations|cplxImplementations)(\s*;.*)$'
    )

    @classmethod
    def _make_test_typedefs_unique(cls, lines, source_file):
        """
        Rename realImplementations/cplxImplementations typedefs to be unique
        per source file, and update all references to them.
        """
        # Generate a unique suffix from the source filename
        base = os.path.splitext(os.path.basename(source_file))[0]
        # Sanitize: replace non-alphanumeric with underscore
        suffix = re.sub(r'[^a-zA-Z0-9]', '_', base)

        # First pass: find which typedefs are defined in this file
        typedefs_found = set()
        for line in lines:
            m = cls._TEST_TYPEDEF_RE.match(line)
            if m:
                typedefs_found.add(m.group(2))

        if not typedefs_found:
            return lines

        # Second pass: rename definitions and all references
        result = []
        for line in lines:
            new_line = line
            for typedef_name in typedefs_found:
                unique_name = f'{typedef_name}_{suffix}'
                # Replace the typedef definition
                new_line = re.sub(
                    r'^(\s*typedef\s+Types<[^>]*>\s*)' + typedef_name + r'(\s*;)',
                    r'\g<1>' + unique_name + r'\g<2>',
                    new_line
                )
                # Replace references (word boundary match)
                new_line = re.sub(
                    r'\b' + typedef_name + r'\b',
                    unique_name,
                    new_line
                )
            result.append(new_line)

        return result

    # ------------------------------------------------------------------
    # Main entry detection
    # ------------------------------------------------------------------

    _MAIN_RE = re.compile(r'\b(?:int|void)\s+main\s*\(')

    def _has_main(self, filepath):
        try:
            with open(filepath, 'r', errors='replace') as fh:
                return bool(self._MAIN_RE.search(fh.read()))
        except OSError:
            return False

    # ------------------------------------------------------------------
    # Include regex patterns
    # ------------------------------------------------------------------

    _INC_QUOTED = re.compile(r'\s*#\s*include\s+"([^"]+)"')
    _INC_ANGLE  = re.compile(r'\s*#\s*include\s+<([^>]+)>')

    # ------------------------------------------------------------------
    # Core amalgamation
    # ------------------------------------------------------------------

    def amalgamate(self, target):
        """
        Build an amalgamated source string for a target dict.
        Returns (content_string, has_cu_expansion).
        """
        source_files = target['sources']

        processed     = set()
        system_inc    = OrderedDict()   # preserves first-seen order; will sort at end
        header_blocks = []              # expanded repo header content
        impl_blocks   = []              # implementation file content
        impl_queue    = []              # implementation files discovered
        has_cu        = target['is_cuda']

        # Regex for preprocessor conditional directives
        _COND_OPEN  = re.compile(r'^\s*#\s*(?:if|ifdef|ifndef)\b')
        _COND_CLOSE = re.compile(r'^\s*#\s*endif\b')

        def _process_file(filepath, is_impl=False, inline_into=None,
                          conditional_context=False):
            """Process a single file: expand repo includes, collect system includes.

            When *inline_into* is not None it must be a list; the file's
            content is appended there instead of into header_blocks /
            impl_blocks.  Repo headers are always inlined into the
            current block to preserve correct dependency ordering.

            *conditional_context* is True when the caller expanded
            this file from inside a preprocessor conditional – in that
            case external ``#include`` directives are kept in-place
            rather than hoisted to the global system-include section.
            """
            nonlocal has_cu
            filepath = os.path.realpath(filepath)
            if filepath in processed:
                return
            processed.add(filepath)

            if filepath.endswith('.cu'):
                has_cu = True

            # Handle generated headers that don't exist on disk
            basename = os.path.basename(filepath)
            if basename in GENERATED_HEADERS and not os.path.isfile(filepath):
                rel = basename + ' (generated)'
                gen = [f'\n// ===== Begin: {rel} =====\n']
                gen.append(GENERATED_HEADERS[basename])
                gen.append(f'// ===== End: {rel} =====\n\n')
                if inline_into is not None:
                    inline_into.extend(gen)
                else:
                    header_blocks.append(''.join(gen))
                return

            try:
                with open(filepath, 'r', errors='replace') as fh:
                    lines = fh.readlines()
            except OSError:
                return

            # Strip UTF-8 BOM from the first line if present
            if lines and lines[0].startswith('\ufeff'):
                lines[0] = lines[0][1:]

            # Strip include guards from repo content
            lines = self._strip_include_guards(lines)

            # Make test typedefs unique per source file to avoid conflicts
            lines = self._make_test_typedefs_unique(lines, filepath)

            rel = os.path.relpath(filepath, self.repo_root)
            block = [f'\n// ===== Begin: {rel} =====\n']

            # Track preprocessor conditional nesting depth so that
            # #include directives inside #if / #ifdef / #ifndef blocks
            # are inlined into the current block rather than hoisted to
            # the top of the amalgamated file.
            cond_depth = 0

            for line in lines:
                # --- Track conditional depth ---
                if _COND_OPEN.match(line):
                    cond_depth += 1
                    block.append(line)
                    continue
                if _COND_CLOSE.match(line):
                    cond_depth -= 1
                    block.append(line)
                    continue

                # Are we in a conditional context?  Either this
                # file has a nonzero cond_depth, or the caller
                # told us we are inside a preprocessor conditional.
                in_conditional = (cond_depth > 0
                                  or conditional_context)

                # --- #include "…" ---
                mq = self._INC_QUOTED.match(line)
                if mq:
                    inc_name = mq.group(1)
                    resolved = self._resolve_include(inc_name, filepath)
                    if resolved and (resolved in self.all_headers
                                     or resolved in self.all_sources):
                        # Always inline repo headers into the current
                        # block to preserve correct dependency ordering.
                        _process_file(resolved, is_impl=False,
                                      inline_into=block,
                                      conditional_context=in_conditional)
                        # Queue implementation files for this header
                        if resolved in self.all_headers:
                            for impl in self._find_impl_files(resolved):
                                if (impl not in processed
                                        and not self._has_main(impl)):
                                    impl_queue.append(impl)
                        continue  # line replaced by expansion
                    # Check generated header names
                    inc_base = os.path.basename(inc_name)
                    if inc_base in GENERATED_HEADERS:
                        fake_path = os.path.join('/generated', inc_base)
                        _process_file(fake_path, is_impl=False,
                                      inline_into=block,
                                      conditional_context=in_conditional)
                        continue
                    # Not a repo header → external include.
                    # If inside a conditional, keep in-place so the
                    # guard is preserved (e.g. #ifdef USE_OMP /
                    # #include "omp.h" / #endif).
                    if in_conditional:
                        block.append(line)
                    else:
                        system_inc[line.rstrip()] = True
                    continue

                # --- #include <…> ---
                ma = self._INC_ANGLE.match(line)
                if ma:
                    inc_name = ma.group(1)
                    # Check if the angle-bracket include is actually
                    # a repo header (some files use <Header.h> style).
                    resolved = self._resolve_include(inc_name, filepath)
                    if resolved and (resolved in self.all_headers
                                     or resolved in self.all_sources):
                        _process_file(resolved, is_impl=False,
                                      inline_into=block,
                                      conditional_context=in_conditional)
                        if resolved in self.all_headers:
                            for impl in self._find_impl_files(resolved):
                                if (impl not in processed
                                        and not self._has_main(impl)):
                                    impl_queue.append(impl)
                        continue
                    # Check generated header names
                    inc_base = os.path.basename(inc_name)
                    if inc_base in GENERATED_HEADERS:
                        fake_path = os.path.join('/generated', inc_base)
                        _process_file(fake_path, is_impl=False,
                                      inline_into=block,
                                      conditional_context=in_conditional)
                        continue
                    # External / system include
                    if in_conditional:
                        block.append(line)
                    else:
                        system_inc[line.rstrip()] = True
                    continue

                block.append(line)

            block.append(f'// ===== End: {rel} =====\n\n')

            if inline_into is not None:
                inline_into.extend(block)
            elif is_impl:
                impl_blocks.append(''.join(block))
            else:
                header_blocks.append(''.join(block))

        # -- Step 1: Process all target source files ----------------------
        for src in source_files:
            _process_file(src, is_impl=True)

        # -- Step 2: Iteratively process implementation files -------------
        MAX_ROUNDS = 50
        for _ in range(MAX_ROUNDS):
            if not impl_queue:
                break
            batch = list(impl_queue)
            impl_queue.clear()
            for impl in batch:
                if impl not in processed and not self._has_main(impl):
                    _process_file(impl, is_impl=True)

        # -- Step 3: Assemble output --------------------------------------
        out = []
        out.append('// ============================================================\n')
        out.append(f'// Amalgamated file for target: {target["name"]}\n')
        out.append('// Auto-generated by docker/amalgamate.py\n')
        out.append('// DO NOT EDIT – changes will be overwritten.\n')
        out.append('// ============================================================\n\n')

        # External / system includes (deduplicated, sorted alphabetically)
        sorted_includes = sorted(system_inc.keys(),
                                 key=lambda s: re.sub(r'[^a-zA-Z0-9/._]', '', s).lower())
        for inc_line in sorted_includes:
            out.append(inc_line + '\n')
        out.append('\n')

        # Expanded repo header content (in first-encounter order)
        out.extend(header_blocks)

        # Implementation file content (in discovery order)
        out.extend(impl_blocks)

        return ''.join(out), has_cu

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def run(self):
        """Discover targets from CMakeLists.txt and amalgamate each one."""
        targets = discover_targets(str(self.repo_root))

        # Add manual targets that can't be auto-discovered
        for name, info in MANUAL_TARGETS.items():
            sources = []
            for src in info['sources']:
                full = os.path.realpath(os.path.join(str(self.repo_root), src))
                if os.path.isfile(full):
                    sources.append(full)
            if sources:
                targets.append({
                    'name': name,
                    'sources': sources,
                    'cmake_dir': os.path.join(str(self.repo_root), info['cmake_dir']),
                    'is_cuda': info.get('is_cuda', False),
                    'output_name': info.get('output_name'),
                })

        print(f"Found {len(targets)} executable target(s):")
        for t in targets:
            src_list = ', '.join(os.path.basename(s) for s in t['sources'])
            print(f"  {t['name']}  ({src_list})")
        print()

        results = []
        for target in targets:
            # Determine output filename
            if target.get('output_name'):
                out_base = target['output_name']
            else:
                first_src = target['sources'][0]
                src_base = os.path.splitext(os.path.basename(first_src))[0]
                out_base = f'{src_base}_amalgamated'

            print(f"Amalgamating target: {target['name']}")

            content, has_cu = self.amalgamate(target)

            ext = '.cu' if has_cu else '.cpp'
            out_path = os.path.join(target['cmake_dir'], f'{out_base}{ext}')

            with open(out_path, 'w') as fh:
                fh.write(content)

            rel_out = os.path.relpath(out_path, self.repo_root)
            n_lines = content.count('\n')
            print(f"  -> {rel_out}  ({n_lines} lines, {len(content)} bytes)")
            results.append({
                'target': target['name'],
                'output': out_path,
                'rel_output': rel_out,
                'has_cu': has_cu,
            })

        print(f"\nDone. Generated {len(results)} amalgamated file(s).")
        return results


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create amalgamated source files for every Gadgetron executable target.')
    parser.add_argument(
        '--repo-root', default=None,
        help='Repository root (default: parent of the docker/ directory)')
    args = parser.parse_args()

    if args.repo_root:
        repo_root = os.path.abspath(args.repo_root)
    else:
        # Auto-detect: this script lives in docker/
        repo_root = str(Path(__file__).resolve().parent.parent)

    print(f"Repository root : {repo_root}")
    print("Building file index ...")

    amal = Amalgamator(repo_root)
    print(f"Indexed {len(amal.all_headers)} header files, "
          f"{len(amal.all_sources)} source files.\n")

    amal.run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
