"""Bounded Nsight Compute diagnostics, separate from benchmark timings.

Example: python profile_ncu.py --request request.json --build /tmp/build \
    --output /tmp/results/ncu --results /tmp/results/results.json

Only the kernel inside an exact NVTX range is collected. Warmup, initialization,
the explicit cache flush, and load-time compilation are outside that range.
Missing tools/counter permission are recorded as unavailable, not successful.
"""

import argparse
import csv
import glob
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback


SOURCES = {
    "cli": "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html",
    "metrics": "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html",
    "permissions": "https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters",
    "installation": "https://docs.nvidia.com/cuda/archive/13.1.0/cuda-installation-guide-linux/index.html",
}

# Query the installed profiler/GPU before using these names. In particular,
# Tensor Core metric availability differs between Blackwell profiler versions.
METRIC_GROUPS = {
    "duration": ["gpu__time_duration.sum"],
    "sm_utilization": ["sm__throughput.avg.pct_of_peak_sustained_elapsed",
                       "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"],
    "tensor_utilization": ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
                           "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                           "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed"],
    "dram": ["dram__throughput.avg.pct_of_peak_sustained_elapsed",
             "dram__bytes_read.sum", "dram__bytes_write.sum", "dram__bytes.sum.per_second"],
    "l2": ["lts__t_sector_hit_rate.pct", "lts__throughput.avg.pct_of_peak_sustained_elapsed",
           "lts__t_bytes.sum"],
    "scheduler": ["smsp__warps_eligible.avg.per_cycle_active",
                  "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
                  "smsp__warp_issue_stalled_membar_per_warp_active.pct",
                  "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                  "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
                  "smsp__warp_issue_stalled_wait_per_warp_active.pct"],
    "occupancy": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "launch": ["launch__waves_per_multiprocessor", "launch__grid_size", "launch__block_size",
               "launch__registers_per_thread", "launch__shared_mem_per_block",
               "launch__occupancy_limit_shared_mem", "launch__occupancy_limit_registers"],
}
METRIC_PATTERN = re.compile(r"\b[a-z][a-z0-9]*__[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*\b")
DEADLINE = None


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def bounded_command(argv, log_path, timeout):
    """Keep raw output and kill the full profiler/application process group."""
    started = time.monotonic()
    if DEADLINE is not None:
        remaining = DEADLINE - started
        if remaining < 1:
            raise TimeoutError("Total Nsight profiling time budget exhausted")
        timeout = min(timeout, remaining)
    record = {"command": [str(x) for x in argv], "log": log_path.name, "timeout_seconds": timeout}
    environment = os.environ.copy()
    environment["NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT"] = "1"
    with log_path.open("w") as log:
        process = subprocess.Popen(record["command"], stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        try:
            record["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            record["timed_out"] = True
            record["returncode"] = None
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    record["seconds"] = time.monotonic() - started
    text = log_path.read_text(errors="replace")
    record["counter_permission_denied"] = ("ERR_NVGPUCTRPERM" in text or
                                          "does not have permission to access NVIDIA GPU Performance Counters" in text)
    return record, text


def find_ncu(explicit=None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    detected = shutil.which("ncu")
    if detected:
        candidates.append(detected)
    candidates.extend(["/usr/local/cuda/bin/ncu", "/usr/local/cuda-13.1/bin/ncu"])
    for pattern in ("/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda*/nsight-compute*/ncu",
                    "/usr/local/cuda*/NsightCompute*/ncu"):
        candidates.extend(sorted(glob.glob(pattern), reverse=True))
    candidates = list(dict.fromkeys(candidates))
    return next((p for p in candidates if Path(p).is_file() and os.access(p, os.X_OK)), None), candidates


def choose_targets(request, results_path, case_name=None, variant=None, include_cublas=False):
    cases = {case["name"]: case for case in request["cases"]}
    results = json.loads(results_path.read_text()) if results_path and results_path.exists() else {}
    if case_name:
        if case_name not in cases:
            raise ValueError(f"Unknown case {case_name}")
        selected = [(case_name, variant or "original")]
    else:
        def best(name, include_original=False):
            candidates = [candidate for candidate in cases.get(name, {}).get("variants", [])
                          if include_original or candidate not in ("original", "fenced")]
            measured = results.get("cases", {}).get(name, {}).get("measurements", {})
            return min(candidates, key=lambda candidate: measured.get(candidate, {}).get("median_ms", float("inf")),
                       default="full_early")
        selected = [("square128", "original"), ("rect144", "original"),
                    ("aligned148", "original"), ("aligned148", best("aligned148")),
                    ("tail152", "original"), ("aligned592", best("aligned592", include_original=True))]
    if include_cublas:
        selected.extend((name, "cublas") for name in dict.fromkeys(name for name, _ in selected))
    selected = list(dict.fromkeys(selected))
    targets, skipped = [], []
    for name, candidate in selected:
        if name not in cases or (candidate != "cublas" and candidate not in cases[name]["variants"]):
            skipped.append({"case": name, "variant": candidate, "reason": "not requested/built"})
            continue
        checked = results.get("cases", {}).get(name, {}).get("validation", {}).get(candidate, [])
        if checked and not all(record.get("passed", False) for record in checked):
            skipped.append({"case": name, "variant": candidate, "reason": "benchmark validation failed"})
            continue
        targets.append({"case": name, "variant": candidate, "shape": cases[name]["shape"]})
    return targets, skipped


def query_metrics(ncu, output, summary):
    desired = [metric for group in METRIC_GROUPS.values() for metric in group]
    base, base_text = bounded_command(
        [ncu, "--query-metrics", "--query-metrics-mode", "base"], output / "metrics_base.txt", 45)
    summary["metric_queries"].append(base)
    if base["returncode"] != 0 or base["counter_permission_denied"]:
        return [], base
    available_bases = set(METRIC_PATTERN.findall(base_text))
    wanted_bases = sorted({metric.split(".")[0] for metric in desired
                           if metric.split(".")[0] in available_bases and not metric.startswith("launch__")})
    available = set()
    if wanted_bases:
        suffix, suffix_text = bounded_command(
            [ncu, "--query-metrics", "--query-metrics-mode", "suffix", "--metrics", ",".join(wanted_bases)],
            output / "metrics_suffixes.txt", 45)
        summary["metric_queries"].append(suffix)
        if suffix["returncode"] == 0:
            available.update(METRIC_PATTERN.findall(suffix_text))
    launch, launch_text = bounded_command(
        [ncu, "--query-metrics", "--query-metrics-collection", "launch"], output / "metrics_launch.txt", 45)
    summary["metric_queries"].append(launch)
    if launch["returncode"] == 0:
        available.update(METRIC_PATTERN.findall(launch_text))
    selected = [metric for metric in desired if metric in available]
    summary["metric_selection"] = {
        "requested": METRIC_GROUPS, "selected": selected,
        "unavailable": [metric for metric in desired if metric not in available],
    }
    return selected, None


def kernel_names(build, stem):
    source = build / (stem + ".cu")
    if not source.exists():
        return []
    text = re.sub(r"__launch_bounds__\([^)]*\)", "", source.read_text())
    return sorted(set(re.findall(r"__global__\s+void\s+([A-Za-z_]\w*)\s*\(", text)))


def parse_metrics(csv_text):
    lines = csv_text.splitlines()
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
            rows = [row for row in reader if row.get("Metric Name") and row.get("Metric Value") is not None]
            return rows
    # Raw-page CSV in current ncu is wide: metric names in the header,
    # then a units row, then one row per profiled kernel.
    for index, line in enumerate(lines):
        if line.startswith('"ID",'):
            reader = csv.DictReader(io.StringIO("\n".join(lines[index:])))
            units = next(reader, {})
            rows = []
            for raw in reader:
                if not raw.get("ID"):
                    continue
                for name, value in raw.items():
                    if name and METRIC_PATTERN.fullmatch(name):
                        rows.append({"ID": raw["ID"], "Kernel Name": raw.get("Kernel Name", ""),
                                     "Metric Name": name, "Metric Unit": units.get(name, ""),
                                     "Metric Value": value})
            return rows
    return []


def kernel_only(args, request):
    import torch
    import tvm
    import tvm_ffi
    import tvm.support.nvcc

    case = next(case for case in request["cases"] if case["name"] == args.case)
    m, n, k = case["shape"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.cuda.set_stream(torch.cuda.default_stream())
    torch.manual_seed(request.get("seeds", [0, 1, 2])[-1])
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16)
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    with tvm_ffi.use_torch_stream():
        if args.variant == "cublas":
            launch = lambda: torch.mm(a, b.T, out=output)
        else:
            library = args.build / f"{args.case}__{args.variant}.so"
            with tvm.target.Target({"kind": "cuda", "arch": request["arch"]}):
                module = tvm.runtime.load_module(str(library))
            function = module.get_function("main", query_imports=True)
            launch = lambda: function(a, b, output)
        for _ in range(5):
            launch()
        flush.zero_()
        torch.cuda.synchronize()
        print("PROFILE_TARGET", args.case, args.variant, list(case["shape"]), flush=True)
        torch.cuda.nvtx.range_push("profile_target")
        try:
            launch()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    print("PROFILE_TARGET_COMPLETE", args.case, args.variant, flush=True)


def main():
    global DEADLINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--ncu")
    parser.add_argument("--case")
    parser.add_argument("--variant")
    parser.add_argument("--include-cublas", action="store_true")
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--timeout", type=int, default=90, help="Per target timeout in seconds")
    parser.add_argument("--total-timeout", type=int, default=600, help="Total subprocess time budget in seconds")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    if args.kernel_only:
        if not args.case or not args.variant:
            parser.error("--kernel-only requires --case and --variant")
        kernel_only(args, request)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.results or args.output / "results.json"
    if not results_path.exists() and args.results is None:
        results_path = args.output.parent / "results.json"
    DEADLINE = time.monotonic() + args.total_timeout
    summary = {
        "status": "starting", "sources": SOURCES, "metric_queries": [], "profiles": [],
        "protocol": {
            "purpose": "Diagnostic counters; benchmark.py remains the authoritative performance timing",
            "selection": "square128/rect144/aligned148/tail152 original; best aligned148 candidate; best aligned592",
            "filter": "One launch inside exact NVTX push/pop range profile_target/",
            "warmup_calls": 5, "explicit_prelaunch_cache_flush_bytes": 256 * 1024 * 1024,
            "replay": "kernel", "cache_control": "all", "clock_control": "none",
            "warning": "Profiler replay/serialization changes execution; ncu duration is not benchmark latency",
            "case_timeout_seconds": args.timeout,
            "total_timeout_seconds": args.total_timeout,
        },
    }

    def checkpoint():
        write_json(args.output / "ncu_summary.json", summary)

    checkpoint()
    try:
        targets, skipped = choose_targets(request, results_path, args.case, args.variant, args.include_cublas)
        summary.update(targets=targets, skipped=skipped, benchmark_results=str(results_path))
        ncu, candidates = find_ncu(args.ncu)
        summary.update(ncu=ncu, executable_candidates=candidates)
        if ncu is None:
            summary.update(status="unavailable", reason="ncu executable not installed or not executable",
                           installation_hint="CUDA 13.1 Ubuntu image: apt install cuda-nsight-compute-13-1")
            checkpoint()
            print("NCU_UNAVAILABLE", summary["reason"], flush=True)
            return
        version, _ = bounded_command([ncu, "--version"], args.output / "version.txt", 20)
        summary["version"] = version
        metrics, query_failure = query_metrics(ncu, args.output, summary)
        if query_failure or not any(not metric.startswith("launch__") for metric in metrics):
            summary.update(status="unavailable", reason="No usable hardware metrics discovered",
                           metric_query_failure=query_failure)
            checkpoint()
            print("NCU_UNAVAILABLE", summary["reason"], flush=True)
            return
        summary["status"] = "running"
        checkpoint()
        for target in targets:
            if DEADLINE - time.monotonic() < 30:
                summary["stop_reason"] = "Less than 30 seconds remains in total profiler time budget"
                break
            stem = target["case"] + "__" + target["variant"]
            report = args.output / (stem + ".ncu-rep")
            command = [ncu, "--target-processes", "all", "--nvtx", "--nvtx-include", "profile_target/",
                       "--launch-count", "1", "--replay-mode", "kernel", "--cache-control", "all",
                       "--clock-control", "none", "--metrics", ",".join(metrics),
                       "--force-overwrite", "--export", str(report)]
            names = kernel_names(args.build, stem) if target["variant"] != "cublas" else []
            if names:
                command.extend(["--kernel-name-base", "function", "--kernel-name",
                                "regex:^(?:" + "|".join(re.escape(name) for name in names) + ")$"])
            command.extend([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--kernel-only", "--request", str(args.request.resolve()),
                            "--build", str(args.build.resolve()), "--output", str(args.output.resolve()),
                            "--case", target["case"], "--variant", target["variant"]])
            print("NCU_START", stem, "metrics", len(metrics), flush=True)
            record = dict(target, status="running", kernel_names=names)
            summary["profiles"].append(record)
            checkpoint()
            run, run_text = bounded_command(command, args.output / (stem + ".profile.log"), args.timeout)
            record["execution"] = run
            if run["counter_permission_denied"]:
                record.update(status="unavailable", reason="ERR_NVGPUCTRPERM: GPU performance counters denied")
                summary.update(status="unavailable", reason=record["reason"])
                checkpoint()
                print("NCU_UNAVAILABLE", stem, record["reason"], flush=True)
                return
            if run["returncode"] != 0 or not report.exists() or report.stat().st_size == 0:
                record.update(status="failed", reason="Profiler failed, timed out, or created no report",
                              log_tail=run_text[-4000:])
                checkpoint()
                print("NCU_FAILED", stem, record["reason"], flush=True)
                if run.get("timed_out"):
                    break
                continue
            record["report"] = report.name
            exported, csv_text = bounded_command(
                [ncu, "--import", str(report), "--page", "raw", "--csv", "--print-units", "base"],
                args.output / (stem + ".csv"), 45)
            record["csv_export"] = exported
            rows = parse_metrics(csv_text)
            record["metrics"] = rows
            kernel_ids = sorted({row.get("ID", "") for row in rows})
            record["profiled_kernel_ids"] = kernel_ids
            collected = [row for row in rows if row.get("Metric Name") in metrics
                         and row.get("Metric Value", "").lower() not in ("n/a", "nan", "", "not available")]
            valid_hardware = [row for row in collected if not row["Metric Name"].startswith("launch__")]
            if exported["returncode"] == 0 and valid_hardware and len(kernel_ids) == 1:
                record["status"] = "complete"
            else:
                record.update(status="failed", reason="CSV export did not contain one kernel with usable hardware metrics")
            checkpoint()
            print("NCU_RESULT", stem, record["status"], "rows", len(rows), flush=True)
        completed = sum(record["status"] == "complete" for record in summary["profiles"])
        summary["status"] = ("complete" if completed == len(targets) and targets
                             else "partial" if completed else "failed")
        checkpoint()
        print("NCU_COMPLETE", summary["status"], completed, "/", len(targets), flush=True)
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        checkpoint()
        raise


if __name__ == "__main__":
    main()
