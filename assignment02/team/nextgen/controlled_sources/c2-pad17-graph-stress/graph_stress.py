"""Bounded Graph diagnostic for the exact first complete-chain benchmark case.

The external runner owns the wall-clock timeout. Every blocking operation is
bracketed by an atomic, fsynced progress.json update so termination preserves the
last entered stage. Wall times locate a stall; they are not benchmark results.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import importlib
import json
import os
from pathlib import Path
import time
import traceback


class Progress:
    def __init__(self, output):
        self.path = Path(output) / "progress.json"
        self.started = time.monotonic()
        self.value = {
            "status": "RUNNING", "diagnostic": "first_benchmark_case_graph_stress_v1",
            "case": {"case_id": "tp1-b1-bf16", "tp": 1, "batch": 1,
                     "seq_lens": [8192], "num_kv_heads": 4, "storage": "bf16", "seed": 101},
            "calls_per_graph": [2, 8, 16, 64], "replays_per_graph": 2,
            "path_order": ["baseline", "merge_only", "tcgen05"],
            "scope": "Stage/stall diagnosis only; no benchmark latency or speedup claim",
            "events": [], "checks": {},
        }
        self.emit("diagnostic", "begin")

    def write(self):
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w") as handle:
            json.dump(self.value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def emit(self, stage, phase, **details):
        event = {"stage": stage, "phase": phase,
                 "utc": datetime.now(timezone.utc).isoformat(),
                 "elapsed_seconds": time.monotonic() - self.started, **details}
        self.value["current"] = event
        self.value["events"].append(event)
        self.write()
        print(json.dumps(event), flush=True)

    @contextmanager
    def stage(self, name, **details):
        self.emit(name, "begin", **details)
        started = time.monotonic()
        yield
        self.emit(name, "end", wall_seconds=time.monotonic() - started, **details)


def run(args, *, check, frozen_manifest):
    progress = Progress(args.output)
    stack_log = (Path(args.output) / "python-stack.log").open("w")
    faulthandler.dump_traceback_later(45, repeat=True, file=stack_log)
    try:
        with progress.stage("import.torch"):
            torch = importlib.import_module("torch")
        with progress.stage("import.baseline_workload"):
            base = importlib.import_module("baseline_workload")
        with progress.stage("import.candidate_workload"):
            old = importlib.import_module("candidate_workload")
        with progress.stage("import.candidate"):
            candidate = importlib.import_module("candidate")
        with progress.stage("import.adapter"):
            adapter = importlib.import_module("nextgen.adapter")
        with progress.stage("import.validation_suite"):
            suite = importlib.import_module("validation.suite")
        with progress.stage("frozen_manifest"):
            manifest = frozen_manifest(args.manifest)
        spec = suite.CaseSpec("tp1-b1-bf16", 4, (8192,), storage="bf16")
        bounds = manifest["thresholds"][spec.family]
        progress.value.update(frozen_protocol_digest=manifest["protocol_digest"],
                              frozen_family_bounds=bounds, family=spec.family)
        with progress.stage("input.make_case"):
            case = suite.make_case(spec, 101)
        with progress.stage("input.synchronize"):
            torch.cuda.synchronize()
        with progress.stage("gold.compute"):
            gold = suite.gold(case)
        with progress.stage("gold.synchronize"):
            torch.cuda.synchronize()

        # Preserve benchmark preparation order, including the upstream capture
        # adapter's eager launch and internal synchronization.
        base.sa.current_platform.is_arch_support_pdl = lambda: False
        with progress.stage("baseline.capture"):
            base_out, calls = base.capture(case)
        with progress.stage("baseline.make_launchers"):
            base_part, base_merge = [base.launcher(call) for call in calls]
        def baseline_chain():
            base_part()
            base_merge()
        with progress.stage("merge_only.selected_config"):
            config = candidate.selected_config(case, use_pdl=False)
        with progress.stage("merge_only.prepare"):
            old_fn, old_out, _, old_splits = old.prepare(case, config)
        with progress.stage("tcgen05.prepare"):
            prepared = adapter.prepare(case, splits=None, merge="feature")
        progress.value.update(actual_backend=prepared["actual_backend"],
                              compiled_variant_id=prepared["compiled_variant_id"],
                              splits=prepared["splits"], merge_only_splits=old_splits,
                              kv_conversion=prepared["kv_conversion"], pdl=False, fallback=False)
        with progress.stage("prepared.eager_enqueue"):
            prepared["chain"]()
        with progress.stage("prepared.eager_synchronize"):
            torch.cuda.synchronize()
        fns = {"baseline": baseline_chain, "merge_only": old_fn, "tcgen05": prepared["chain"]}
        values = {"baseline": base_out, "merge_only": old_out, "tcgen05": prepared["output"]}
        for name, value in values.items():
            with progress.stage(f"eager_gate.{name}"):
                progress.value["checks"][f"eager.{name}"] = check(case, value, gold, bounds)

        # Deliberately do not use run.capture(): it hides three warmup graph
        # replays and synchronization. Here each graph is replayed exactly twice.
        graphs = []
        for call_count in (2, 8, 16, 64):
            for name, fn in fns.items():
                label = f"graph{call_count}.{name}"
                with progress.stage(f"{label}.warmup_enqueue", eager_calls=5):
                    for _ in range(5):
                        fn()
                with progress.stage(f"{label}.warmup_synchronize"):
                    torch.cuda.synchronize()
                with progress.stage(f"{label}.pre_gate"):
                    progress.value["checks"][f"{label}.before"] = check(case, values[name], gold, bounds)
                with progress.stage(f"{label}.create_graph_and_stream"):
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                with progress.stage(f"{label}.capture"):
                    with torch.cuda.graph(graph, stream=stream):
                        progress.emit(f"{label}.capture", "entered")
                        for _ in range(call_count):
                            fn()
                        progress.emit(f"{label}.capture", "enqueued", calls=call_count)
                with progress.stage(f"{label}.capture_synchronize"):
                    torch.cuda.current_stream().wait_stream(stream)
                    torch.cuda.synchronize()
                graphs.append(graph)  # Keep captures/workspaces alive for all stages.
                for replay in (1, 2):
                    with progress.stage(f"{label}.replay{replay}.enqueue"):
                        graph.replay()
                    with progress.stage(f"{label}.replay{replay}.synchronize"):
                        torch.cuda.synchronize()
                    with progress.stage(f"{label}.replay{replay}.post_gate"):
                        progress.value["checks"][f"{label}.replay{replay}"] = check(case, values[name], gold, bounds)
        progress.value["status"] = "PASS"
        progress.emit("diagnostic", "end", graph_count=len(graphs), graph_replays=2 * len(graphs))
        return progress.value
    except Exception as error:
        progress.value.update(status="FAIL", failed_at=progress.value.get("current"),
                              error=str(error), traceback=traceback.format_exc())
        progress.emit("diagnostic", "failure")
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        stack_log.close()
