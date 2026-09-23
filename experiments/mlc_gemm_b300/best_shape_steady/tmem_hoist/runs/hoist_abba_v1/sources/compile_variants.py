"""CPU-only compile and SASS gate for the fixed-shape TMEM-base experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

INPUT = Path('/opt/tmem_hoist/input')
STEM = 'aligned148_k8192__wide'
NVCC_FLAGS = ['--cubin', '-O3', '-arch=sm_103a', '--std=c++17',
              '--expt-relaxed-constexpr', '--expt-extended-lambda',
              '--use_fast_math', '--ptxas-options=-v,--register-usage-level=10']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv, output, timeout=180):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True,
                            timeout=timeout)
    output.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'{argv[0]} returned {result.returncode}; see {output}')
    return result.stdout, result.stderr


def resource_usage(log):
    registers = re.search(r'Used (\d+) registers', log)
    stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', log)
    return {'registers': int(registers[1]) if registers else None,
            'stack_bytes': int(stack[1]) if stack else None,
            'spill_store_bytes': int(stack[2]) if stack else None,
            'spill_load_bytes': int(stack[3]) if stack else None}


def inspect_sass(text):
    """Find the smallest backward branch enclosing all static MMA instructions.

    This fixed kernel has one MMA K-loop, ptxas-unrolled twice. Conservatively
    fail the gate if cuobjdump's instruction/branch representation changes.
    """
    instructions = []
    for line in text.splitlines():
        match = re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;', line)
        if match:
            instructions.append((int(match[1], 16), match[2]))
    mmas = [(pc, op) for pc, op in instructions if 'UTCHMMA.' in op]
    loops = []
    if mmas:
        for pc, op in instructions:
            branch = re.search(r'\bBRA(?:\.[A-Z]+)?\s+(?:!?UP\d+,\s*)?(0x[0-9a-fA-F]+)', op)
            if branch:
                dest = int(branch[1], 16)
                if dest < pc and all(dest <= addr <= pc for addr, _ in mmas):
                    loops.append((pc - dest, dest, pc))
    if not loops:
        return {'identified': False, 'mma_count': len(mmas),
                'reason': 'No backward-branch interval encloses the MMA K-loop'}
    _, start, end = min(loops)
    loop = [(pc, op) for pc, op in instructions if start <= pc <= end]
    loads = [(pc, op) for pc, op in loop if re.search(r'\bLDS(?:\.|\s)', op)]
    return {'identified': True, 'mma_count': len(mmas),
            'loop_start': hex(start), 'loop_end': hex(end),
            'shared_loads_in_mma_loop': [{'pc': hex(pc), 'instruction': op} for pc, op in loads],
            'shared_load_count_in_mma_loop': len(loads),
            'loop_instruction_count': len(loop),
            'loop_listing': '\n'.join(f'{pc:#06x} {op}' for pc, op in loop)}


def tir_invariants(script):
    expected = ['T.cta_id([148])', 'T.cta_id_in_cluster([2, 1])',
                'T.warpgroup_id([3])', 'T.warp_id_in_wg([4])', 'T.lane_id([32])',
                'T.int64(214016)', '(3, 2, 128, 64)', '(3, 128, 64)', '(2, 128, 128)',
                'T.Buffer((2048, 8192)', 'T.Buffer((9472, 8192)', 'T.Buffer((2048, 9472)']
    return {value: value in script for value in expected}


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    summary = {'success': False, 'gpu_ready': False,
               'started_at': datetime.now(timezone.utc).isoformat(),
               'shape': [2048, 9472, 8192], 'grid_ctas': 148,
               'cluster_ctas': 2, 'threads_per_cta': 384,
               'dynamic_shared_memory_bytes': 214016, 'variants': {},
               'note': 'No GPU is used by this process; GPU correctness remains untested.'}
    try:
        import tvm
        import kernels_hoisted
        summary['tvm_version'] = tvm.__version__
        if tvm.__version__ != '0.26.0':
            raise RuntimeError('Expected frozen apache-tvm==0.26.0 environment')
        summary['factory_sha256'] = sha(Path(kernels_hoisted.__file__))
        summary['input_sha256'] = {path.name: sha(path) for path in INPUT.iterdir() if path.is_file()}
        kernel = kernels_hoisted.hgemm_v9_wide(2048, 9472, 8192, sm_count=148)
        tir = kernel.script() + '\n'
        (output / 'hoisted.tirx.py').write_text(tir)
        summary['tir_invariants'] = tir_invariants(tir)
        summary['baseline_tir_invariants'] = tir_invariants((INPUT / (STEM + '.tirx.py')).read_text())
        target = tvm.target.Target({'kind': 'cuda', 'arch': 'sm_103a'}).with_host('llvm')
        with target:
            executable = tvm.compile(tvm.IRModule({'main': kernel}), target=target, tir_pipeline='tirx')
        cuda = executable.mod.imports[0].inspect_source('cuda')
        (output / 'hoisted.cu').write_text(cuda)
        executable.export_library(str(output / 'hoisted.so'))
        for suffix in ('.cu', '.so', '.tirx.py', '.compile.json'):
            shutil.copyfile(INPUT / (STEM + suffix), output / ('baseline' + suffix))
        for name in ('baseline', 'hoisted'):
            command = ['nvcc'] + NVCC_FLAGS + [str(output / (name + '.cu')),
                                              '-o', str(output / (name + '.cubin'))]
            stdout, stderr = run(command, output / (name + '.ptxas.txt'))
            sass, _ = run(['cuobjdump', '--dump-sass', output / (name + '.cubin')],
                          output / (name + '.sass'))
            inspection = inspect_sass(sass)
            (output / (name + '.mma_loop.sass')).write_text(inspection.pop('loop_listing', '') + '\n')
            summary['variants'][name] = {'nvcc_command': command, 'resource_usage': resource_usage(stdout + stderr),
                                         'sass_analysis': inspection,
                                         'sha256': {suffix: sha(output / (name + suffix))
                                                    for suffix in ('.cu', '.cubin', '.so', '.sass')}}

        # Validate the generated CUDA uses a named scalar, initialized before
        # the persistent loop, and every MMA's first argument uses that scalar.
        lines = cuda.splitlines()
        declarations = [(i, line) for i, line in enumerate(lines)
                        if re.search(r'\b(?:uint|unsigned int)\s+mma_tmem_base\s*=', line)]
        mma_calls = [(i, line) for i, line in enumerate(lines)
                     if re.match(r'\s*ptx_tcgen05_mma_cta_2_kind_f16_SS\(', line)]
        declaration_outside = False
        if len(declarations) == 1 and mma_calls:
            decl_i = declarations[0][0]
            declaration_outside = (decl_i < mma_calls[0][0]
                                   and any('while (1)' in line for line in lines[decl_i:mma_calls[0][0]]))
        scalar_args = bool(mma_calls) and all('mma_tmem_base' in line.split(',', 1)[0] for _, line in mma_calls)
        base_sass = summary['variants']['baseline']['sass_analysis']
        new_sass = summary['variants']['hoisted']['sass_analysis']
        gate = {
            'launch_and_shared_memory_unchanged': all(summary['tir_invariants'].values()) and all(summary['baseline_tir_invariants'].values()),
            'scalar_declared_before_persistent_loop': declaration_outside,
            'all_generated_mma_calls_use_scalar_base': scalar_args,
            'same_four_generated_mma_calls': len(mma_calls) == 4,
            'baseline_mma_loop_has_two_lds': base_sass.get('identified') and base_sass.get('shared_load_count_in_mma_loop') == 2,
            'candidate_mma_loop_has_zero_lds': new_sass.get('identified') and new_sass.get('shared_load_count_in_mma_loop') == 0,
            'same_eight_static_mma_instructions': base_sass.get('mma_count') == new_sass.get('mma_count') == 8,
            'candidate_has_no_spills': all(summary['variants']['hoisted']['resource_usage'].get(key) == 0
                                          for key in ('spill_store_bytes', 'spill_load_bytes')),
        }
        summary.update(success=True, gpu_ready=all(gate.values()), gate=gate,
                       gate_failures=[name for name, passed in gate.items() if not passed])
    except BaseException:
        summary['error'] = traceback.format_exc()
        raise
    finally:
        summary['seconds'] = time.monotonic() - started
        write(output / 'build_summary.json', summary)
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
