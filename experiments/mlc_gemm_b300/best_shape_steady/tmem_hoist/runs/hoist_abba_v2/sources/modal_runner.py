"""One new TMEM-base-hoist experiment, with separate CPU and GPU gates."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import modal

HERE = Path(__file__).resolve().parent
ROOT = Path('/opt/tmem_hoist')
VOLUME_ROOT = Path('/results')
VOLUME_NAME = 'mlc-b300-tmem-hoist'
FILES = ('modal_runner.py', 'kernels_hoisted.py', 'compile_variants.py', 'benchmark.py')
STEM = 'aligned148_k8192__wide'
SECTIONS = ['SpeedOfLight', 'ComputeWorkloadAnalysis', 'MemoryWorkloadAnalysis',
            'MemoryWorkloadAnalysis_Tables', 'SchedulerStats', 'WarpStateStats',
            'InstructionStats', 'LaunchStats', 'Occupancy', 'SourceCounters']

if modal.is_local():
    if os.environ.get('MODAL_PROFILE') != 'simidawhu':
        raise RuntimeError('Set MODAL_PROFILE=simidawhu explicitly')
    spec = importlib.util.spec_from_file_location('hoist_image', HERE.parent / 'base_image.py')
    image_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(image_module)
    image = image_module.image.apt_install('cuda-nsight-compute-13-1')
    original = HERE.parent / 'ncu_gui/runs/20260920T105923Z_78d47693/input'
    INPUTS = {f'input/{STEM}{suffix}': original / (STEM + suffix)
              for suffix in ('.so', '.cu', '.tirx.py', '.compile.json')}
    INPUTS['input/kernels_tuned.py'] = original / 'kernels_tuned.py'
    INPUTS['input/base_image.py'] = HERE.parent / 'base_image.py'
    INPUTS.update({name: HERE / name for name in FILES})
    for name, path in INPUTS.items():
        image = image.add_local_file(path, str(ROOT / name))
else:
    image = None

app = modal.App('mlc-b300-tmem-hoist')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run_path(run_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    return VOLUME_ROOT / run_id


def check_manifest(manifest):
    for name, sha in manifest.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError('Source mismatch: ' + name)


def collect(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


def command(argv, path, timeout):
    import signal
    import subprocess
    import threading
    import time
    started = time.monotonic()
    env = os.environ.copy()
    env['NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT'] = '1'
    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, start_new_session=True)
    def read():
        with path.open('w') as output:
            for line in process.stdout:
                output.write(line)
                output.flush()
                if path.name in ('compile.log', 'benchmark.log', 'profile.log'):
                    print(line, end='', flush=True)
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
    reader.join(timeout=10)
    return {'argv': list(map(str, argv)), 'returncode': process.returncode,
            'timed_out': timed_out, 'seconds': time.monotonic() - started,
            'log': path.name}


@app.function(image=image, cpu=4, memory=16384, timeout=900, retries=0,
              max_containers=1, scaledown_window=2, volumes={str(VOLUME_ROOT): volume})
def cpu_build(run_id, manifest):
    import sys
    import traceback
    volume.reload()
    out = run_path(run_id)
    if out.exists():
        raise RuntimeError('Run exists; retrieve instead of repeating')
    out.mkdir()
    status = {'state': 'building', 'started_at': now(), 'manifest': manifest, 'commands': []}
    try:
        check_manifest(manifest)
        for name in manifest:
            dest = out / 'sources' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
        write_json(out / 'status.json', status)
        volume.commit()
        record = command([sys.executable, '-u', str(ROOT / 'compile_variants.py'),
                          '--output', str(out / 'build')], out / 'compile.log', 780)
        status['commands'].append(record)
        if record['returncode'] != 0:
            raise RuntimeError('CPU compilation failed; no GPU started')
        summary = json.loads((out / 'build/build_summary.json').read_text())
        if not summary.get('success') or not summary.get('gpu_ready'):
            raise RuntimeError('SASS/build gate not satisfied; no GPU started')
        status.update(state='built', finished_at=now())
    except BaseException as exc:
        status.update(state='build_failed', error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
    write_json(out / 'status.json', status)
    volume.commit()
    return {'status': status, 'files': collect(out)}


@app.function(image=image, gpu='B300', cpu=4, memory=32768, timeout=900, retries=0,
              max_containers=1, scaledown_window=2, volumes={str(VOLUME_ROOT): volume})
def gpu_run(run_id, manifest):
    import sys
    import traceback
    volume.reload()
    out = run_path(run_id)
    status = json.loads((out / 'status.json').read_text())
    if status['state'] != 'built' or status['manifest'] != manifest:
        raise RuntimeError('Refusing repeated GPU execution or changed source snapshot')
    check_manifest(manifest)
    status.update(state='running', gpu_started_at=now())
    write_json(out / 'status.json', status)
    volume.commit()
    try:
        status['commands'].append(command(['nvidia-smi'], out / 'nvidia_smi.txt', 20))
        record = command([sys.executable, '-u', str(ROOT / 'benchmark.py'), '--run-dir', str(out)],
                         out / 'benchmark.log', 480)
        status['commands'].append(record)
        write_json(out / 'status.json', status)
        volume.commit()
        if record['returncode'] != 0:
            raise RuntimeError('Benchmark/validation failed; no profiler launched')
        status['state'] = 'profiling_candidate'
        write_json(out / 'status.json', status)
        volume.commit()
        ncu = shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
        status['commands'].append(command([ncu, '--version'], out / 'ncu_version.txt', 20))
        report = out / 'hoisted_M2048_N9472_K8192.ncu-rep'
        select = [token for section in SECTIONS for token in ('--section', section)]
        argv = [ncu] + select + ['--metrics',
            'sm__cycles_active.avg.pct_of_peak_sustained_elapsed,'
            'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,'
            'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active',
            '--target-processes', 'all', '--nvtx', '--nvtx-include', 'tmem_hoist_candidate/',
            '--kernel-name-base', 'function', '--kernel-name', 'regex:^kernel_kernel$',
            '--launch-count', '1', '--replay-mode', 'kernel', '--cache-control', 'none',
            '--clock-control', 'none', '--import-source', 'yes', '--export', str(report),
            sys.executable, '-u', str(ROOT / 'benchmark.py'), '--run-dir', str(out),
            '--profile-variant', 'hoisted']
        record = command(argv, out / 'profile.log', 300)
        status['commands'].append(record)
        if record['returncode'] != 0 or not report.exists():
            raise RuntimeError('Candidate NCU failed; completed benchmark is retained')
        for page, name, extra in [('details', 'details.txt', []),
                                  ('raw', 'metrics.csv', ['--csv', '--print-units', 'base'])]:
            record = command([ncu, '--import', str(report), '--page', page] + extra,
                             out / name, 30)
            status['commands'].append(record)
            if record['returncode'] != 0:
                raise RuntimeError('NCU export failed: ' + page)
        status.update(state='complete', finished_at=now(),
                      report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc(), finished_at=now())
    write_json(out / 'status.json', status)
    volume.commit()
    return {'status': status, 'files': collect(out)}


@app.function(image=image, cpu=1, timeout=120, retries=0, volumes={str(VOLUME_ROOT): volume})
def retrieve(run_id):
    volume.reload()
    out = run_path(run_id)
    return {'status': json.loads((out / 'status.json').read_text()), 'files': collect(out)}


@app.local_entrypoint()
def main(run_dir: str, phase: str = 'build'):
    out = Path(run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    invocation = out / 'invocation.json'
    manifest = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in INPUTS.items()}
    if phase == 'build':
        if invocation.exists():
            raise RuntimeError('Build already dispatched; use retrieve')
        info = {'run_id': datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8],
                'volume': VOLUME_NAME, 'manifest': manifest, 'created_at': now()}
        write_json(invocation, info)
        call = cpu_build.spawn(info['run_id'], manifest)
        info['build_call_id'] = call.object_id
    elif phase in ('gpu', 'retrieve'):
        info = json.loads(invocation.read_text())
        if phase == 'gpu':
            if info['manifest'] != manifest:
                raise RuntimeError('Local sources changed after CPU gate')
            if 'gpu_dispatch_started_at' in info:
                raise RuntimeError('GPU already dispatched; retrieve only')
            info['gpu_dispatch_started_at'] = now()
            write_json(invocation, info)
            call = gpu_run.spawn(info['run_id'], manifest)
            info['gpu_call_id'] = call.object_id
        else:
            call = retrieve.spawn(info['run_id'])
    else:
        raise ValueError('phase must be build, gpu, or retrieve')
    write_json(invocation, info)
    print('DISPATCH', json.dumps(info), flush=True)
    result = call.get()
    for name, data in result['files'].items():
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    write_json(out / (phase + '_return.json'), result['status'])
    print('COMPLETE', json.dumps(result['status']), flush=True)
