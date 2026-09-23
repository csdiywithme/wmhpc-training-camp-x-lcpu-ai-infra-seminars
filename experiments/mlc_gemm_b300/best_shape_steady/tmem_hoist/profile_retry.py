"""One bounded, reduced-counter retry of a failed NCU capture; no benchmark."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

import modal

HERE = Path(__file__).resolve().parent
runner_path = HERE / 'modal_runner.py' if modal.is_local() else Path('/opt/tmem_hoist/modal_runner.py')
spec = importlib.util.spec_from_file_location('hoist_runner', runner_path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
image = runner.image.add_local_file(__file__, '/opt/tmem_hoist/profile_retry.py') if modal.is_local() else None
app = modal.App('mlc-b300-tmem-hoist-profile-retry')
METRICS = ['gpu__time_duration.sum', 'smsp__sass_inst_executed_op_shared_ld.sum',
           'smsp__sass_inst_executed_op_shared_st.sum',
           'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
           'launch__registers_per_thread', 'launch__shared_mem_per_block_dynamic',
           'launch__grid_size', 'launch__block_size']


@app.function(image=image, gpu='B300', cpu=4, memory=16384, timeout=240,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={str(runner.VOLUME_ROOT): runner.volume})
def capture(run_id):
    runner.volume.reload()
    parent = runner.run_path(run_id)
    source_status = json.loads((parent / 'status.json').read_text())
    if source_status['state'] != 'failed' or 'Candidate NCU failed' not in source_status.get('error', ''):
        raise RuntimeError('Only the explicitly failed profile may use this bounded retry')
    bench = json.loads((parent / 'benchmark/results.json').read_text())
    if bench['status'] != 'complete':
        raise RuntimeError('Expected preserved completed benchmark')
    out = parent / 'profile_retry_hardware'
    if out.exists():
        raise RuntimeError('Retry already exists; do not profile again')
    (out / 'build').mkdir(parents=True)
    library = parent / 'build/hoisted.so'
    expected = bench['libraries']['hoisted']['sha256']
    if hashlib.sha256(library.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Candidate changed')
    shutil.copyfile(library, out / 'build/hoisted.so')
    runner.check_manifest(source_status['manifest'])
    shutil.copyfile('/opt/tmem_hoist/profile_retry.py', out / 'profile_retry.py')
    status = {'state': 'profiling', 'started_at': runner.now(), 'metrics': METRICS,
              'candidate_sha256': expected, 'commands': [], 'benchmark_repeated': False}
    runner.write_json(out / 'status.json', status)
    runner.volume.commit()
    try:
        status['commands'].append(runner.command(['nvidia-smi'], out / 'nvidia_smi.txt', 20))
        report = out / 'candidate_hardware.ncu-rep'
        ncu = shutil.which('ncu') or '/usr/local/cuda/bin/ncu'
        argv = [ncu, '--metrics', ','.join(METRICS), '--target-processes', 'all',
                '--nvtx', '--nvtx-include', 'tmem_hoist_candidate/',
                '--kernel-name-base', 'function', '--kernel-name', 'regex:^kernel_kernel$',
                '--launch-count', '1', '--replay-mode', 'kernel', '--cache-control', 'none',
                '--clock-control', 'none', '--export', str(report),
                sys.executable, '-u', '/opt/tmem_hoist/benchmark.py',
                '--run-dir', str(out), '--profile-variant', 'hoisted']
        record = runner.command(argv, out / 'profile.log', 150)
        status['commands'].append(record)
        if record['returncode'] != 0 or not report.exists():
            raise RuntimeError('Reduced-counter capture failed; no more automatic retries')
        target = json.loads((out / 'profile_candidate/profile_target.json').read_text())
        if target['status'] != 'complete' or not target['validation']['passed']:
            raise RuntimeError('Profiled output not validated')
        for page, name, extra in [('details', 'details.txt', []),
                                  ('raw', 'metrics.csv', ['--csv', '--print-units', 'base'])]:
            record = runner.command([ncu, '--import', str(report), '--page', page] + extra,
                                     out / name, 20)
            status['commands'].append(record)
            if record['returncode'] != 0:
                raise RuntimeError('Export failed')
        status.update(state='complete', report_sha256=hashlib.sha256(report.read_bytes()).hexdigest())
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
    status['finished_at'] = runner.now()
    runner.write_json(out / 'status.json', status)
    runner.volume.commit()
    return {'status': status, 'files': runner.collect(out)}


@app.local_entrypoint()
def main(run_dir: str):
    parent = Path(run_dir).resolve()
    info = json.loads((parent / 'invocation.json').read_text())
    out = parent / 'profile_retry_hardware'
    out.mkdir(exist_ok=True)
    marker = out / 'invocation.json'
    if marker.exists():
        raise RuntimeError('Already dispatched; retrieve only')
    runner.write_json(marker, {'run_id': info['run_id'], 'started_at': runner.now()})
    call = capture.spawn(info['run_id'])
    runner.write_json(marker, {'run_id': info['run_id'], 'call_id': call.object_id})
    result = call.get()
    for name, data in result['files'].items():
        path = out / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(data)
    print('PROFILE_RETRY', json.dumps(result['status']), flush=True)
