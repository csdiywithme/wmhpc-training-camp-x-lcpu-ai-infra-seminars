"""Remote payload; upstream snapshots remain read-only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path("/tmp/c1-artifacts")
OUT.mkdir(exist_ok=True)


def command(args, name, timeout=500):
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    result = dict(command=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    (OUT / (name + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(result, stdout=p.stdout[:12000], stderr=p.stderr[:6000],
                          full_stdout_chars=len(p.stdout), full_stderr_chars=len(p.stderr))), flush=True)
    return p


def official(heads):
    # Run the snapshot unchanged first. Preserve incompatibility if current pinned
    # FLA no longer accepts this older benchmark's keyword conventions.
    command([sys.executable, "/opt/FlashKDA/benchmarks/bench_fwd.py", "--mode", "all",
             "--H", str(heads), "--warmup", "30", "--iters", "200", "--repeats", "5"],
            "official-benchmark")
    # FLA a3edffc introduced explicit safe_gate: without it the old benchmark's
    # lower_bound keyword does not select FlashKDA's bounded sigmoid gate.
    source = Path("/opt/FlashKDA/benchmarks/bench_fwd.py").read_text()
    source = source.replace("lower_bound=LOWER_BOUND,\n            transpose_state_layout=True,",
                            "lower_bound=LOWER_BOUND,\n            safe_gate=True,\n            state_v_first=True,")
    corrected = OUT / "bench_fwd_safe_gate.py"
    corrected.write_text(source)
    command([sys.executable, str(corrected), "--mode", "all", "--H", str(heads),
             "--warmup", "30", "--iters", "200", "--repeats", "5"], "semantic-matched-benchmark")


def inputs(heads=12, lengths=(8192,), seed=0):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    shape = (1, sum(lengths), heads, 128)
    q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16() for _ in range(2)]
    v, g = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, device="cuda")
    a = torch.rand(heads, device="cuda")
    bias = torch.rand(heads, 128, device="cuda")
    h0 = torch.randn(len(lengths), heads, 128, 128, device="cuda").bfloat16()
    ht, out = torch.empty_like(h0), torch.empty_like(q)
    kwargs = dict(A_log=a, dt_bias=bias, lower_bound=-5.0, initial_state=h0, final_state=ht)
    if len(lengths) > 1:
        kwargs["cu_seqlens"] = torch.tensor([0] + list(__import__("itertools").accumulate(lengths)),
                                          device="cuda", dtype=torch.int64)
    return (q, k, v, g, beta, 128 ** -0.5, out), kwargs


def profile(heads, module_name="flash_kda"):
    import torch
    import importlib
    flash_kda = importlib.import_module(module_name)
    flash_kda_C = importlib.import_module(module_name + "_C")
    (OUT / "profile-module.json").write_text(json.dumps({"module": module_name}))
    if module_name != "flash_kda":
        baseline = importlib.import_module("flash_kda")
        validation = []
        for lengths in ((8192,), (1300, 547, 2048, 963, 271, 3063), (1024,) * 8):
            args, kw = inputs(heads, lengths)
            baseline.fwd(*args, **kw)
            expected_o, expected_h = args[-1].clone(), kw["final_state"].clone()
            flash_kda.fwd(*args, **kw)
            torch.cuda.synchronize()
            record = dict(heads=heads, lengths=lengths, output=error_stats(args[-1], expected_o),
                          state=error_stats(kw["final_state"], expected_h))
            validation.append(record)
            assert record["output"]["exact"] and record["state"]["exact"]
        (OUT / "large-shape-correctness.json").write_text(json.dumps(validation, indent=2))
    command(["cuobjdump", "--dump-sass", flash_kda_C.__file__], "sass", timeout=90)
    command(["cuobjdump", "--dump-resource-usage", flash_kda_C.__file__], "resources", timeout=90)
    payload = OUT / "profile-one.py"
    payload.write_text("import sys; sys.path.insert(0, '/opt')\nfrom run_experiments import inputs\nimport torch\nimport " + module_name + " as flash_kda\na,k=inputs(" + str(heads) + ")\nflash_kda.fwd(*a,**k)\ntorch.cuda.synchronize()\n")
    command(["ncu", "--clock-control", "none", "--cache-control", "all", "--set", "detailed",
             "--kernel-name-base", "function", "--kernel-name", "regex:.*_flash_kda_fwd_.*",
             "--launch-count", "2",
             "--export", str(OUT / "baseline"), sys.executable, str(payload)], "ncu", timeout=180)
    if (OUT / "baseline.ncu-rep").exists():
        command(["ncu", "--import", str(OUT / "baseline.ncu-rep"), "--page", "raw", "--csv"],
                "ncu-metrics", timeout=90)
    args, kwargs = inputs(heads)
    for _ in range(5):
        flash_kda.fwd(*args, **kwargs)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            flash_kda.fwd(*args, **kwargs)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(OUT / "torch-profile.json"))
    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


def challenge_profile(heads):
    profile(heads, "flash_kda_split2")


def tile(heads):
    p = command(["nvcc", "-std=c++17", "-O3", "-lineinfo", "--expt-relaxed-constexpr",
             "-gencode", "arch=compute_103a,code=sm_103a", "-I/opt/FlashKDA/cutlass/include",
             "/opt/tile_microbench.cu", "-o", "/tmp/tile_microbench"], "tile-build", timeout=180)
    if p.returncode:
        raise RuntimeError("Tile microbenchmark compilation failed")
    p = command(["/tmp/tile_microbench"], "tile-results", timeout=180)
    if p.returncode:
        raise RuntimeError("Tile microbenchmark failed")
    command(["cuobjdump", "--dump-sass", "/tmp/tile_microbench"], "tile-sass", timeout=60)


def numeric(heads):
    p = command([sys.executable, "/opt/chunk_numeric_gpu.py"], "numeric-run", timeout=650)
    if p.returncode:
        raise RuntimeError("Numeric microbenchmark failed; see numeric-run.json")


def error_stats(actual, expected):
    import torch
    a, e = actual.double(), expected.double()
    diff = a - e
    return dict(finite=bool(torch.isfinite(a).all()), exact=torch.equal(actual, expected),
                max_abs=diff.abs().max().item(), mean_abs=diff.abs().mean().item(),
                rmse=diff.square().mean().sqrt().item(),
                rmse_ratio=(diff.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-30)).item())


def precision(heads, candidate=None, cases=None, gates=("random", "weak_decay", "strong_decay")):
    import torch
    import torch.nn.functional as F
    import flash_kda
    from fla.ops.kda import chunk_kda
    sys.path.insert(0, "/opt/FlashKDA/tests")
    from torch_ref import torch_ref
    sys.path.insert(0, "/opt")
    from c1_naive import naive_recurrent_kda
    records = []
    # Naive reference promotes internally to FP32 even if FP64 tensors are passed.
    # Gate activation and q/k normalization are kept in FP32 for this independent
    # reference, so its difference includes quantization and approximation error.
    cases = cases or [(16,), (17,), (97,), (17, 33, 65), (1024,)]
    for seed in (0, 1):
        for lengths in cases:
            for gate in gates:
                args, kw = inputs(min(heads, 4), lengths, seed)
                if gate != "random":
                    args[3].fill_(-8 if gate == "weak_decay" else 8)
                    kw["dt_bias"].zero_()
                q, k, v, g, beta, scale, out = args
                flash_kda.fwd(*args, **kw)
                torch.cuda.synchronize()
                ref_o, ref_h = torch.empty_like(out), torch.empty_like(kw["final_state"])
                ref_kw = dict(kw, final_state=ref_h)
                torch_ref(q, k, v, g, beta, scale, ref_o, **ref_kw)
                record = dict(seed=seed, lengths=lengths, heads=q.shape[2], gate=gate,
                              official_output=error_stats(out, ref_o),
                              official_state=error_stats(kw["final_state"], ref_h))
                assert record["official_output"]["exact"] and record["official_state"]["exact"], record
                qs, ks = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
                activated_g = -5.0 * torch.sigmoid(kw["A_log"].exp()[None, None, :, None] *
                                                 (g.float() + kw["dt_bias"][None, None]))
                activated_beta = beta.float().sigmoid()
                naive_os, naive_hs = [], []
                start = 0
                for i, length in enumerate(lengths):
                    stop = start + length
                    no, nh = naive_recurrent_kda(qs[:, start:stop], ks[:, start:stop], v[:, start:stop].float(),
                        activated_g[:, start:stop], activated_beta[:, start:stop], scale=scale,
                        initial_state=kw["initial_state"][i:i+1].float().transpose(-1, -2).contiguous(),
                        output_final_state=True)
                    naive_os.append(no)
                    naive_hs.append(nh.transpose(-1, -2).contiguous())
                    start = stop
                no, nh = torch.cat(naive_os, dim=1), torch.cat(naive_hs, dim=0)
                record["naive_output"] = error_stats(out, no)
                record["naive_state"] = error_stats(kw["final_state"], nh)
                if sum(lengths) >= 8192:
                    record["naive_output_windows"] = [dict(start=start, end=min(start+1024, out.shape[1]),
                        **error_stats(out[:, start:start+1024], no[:, start:start+1024]))
                        for start in range(0, out.shape[1], 1024)]
                co, ch = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                    initial_state=kw["initial_state"].float(), output_final_state=True,
                    use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, state_v_first=True, safe_gate=True,
                    lower_bound=-5.0, A_log=kw["A_log"], dt_bias=kw["dt_bias"],
                    cu_seqlens=kw.get("cu_seqlens"))
                record["fla_chunk_output"] = error_stats(co, no)
                record["fla_chunk_state"] = error_stats(ch, nh)
                if candidate is not None:
                    trial_o, trial_h = torch.empty_like(out), torch.empty_like(kw["final_state"])
                    candidate.fwd(q, k, v, g, beta, scale, trial_o, **dict(kw, final_state=trial_h))
                    torch.cuda.synchronize()
                    record["candidate_output"] = error_stats(trial_o, out)
                    record["candidate_state"] = error_stats(trial_h, kw["final_state"])
                    assert record["candidate_output"]["exact"] and record["candidate_state"]["exact"], record
                records.append(record)
                (OUT / "precision.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)


def long_precision(heads):
    precision(heads, cases=[(8192,), (32768,)], gates=("weak_decay",))


def timings(fn, warmup=20, iters=100, repeats=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            start[i].record()
            fn()
            end[i].record()
        torch.cuda.synchronize()
        samples.extend(s.elapsed_time(e) for s, e in zip(start, end))
    return dict(mean_ms=sum(samples) / len(samples), min_ms=min(samples),
                max_ms=max(samples), samples_ms=samples)


def challenge(heads):
    import torch
    import flash_kda
    import flash_kda_split2 as candidate
    import flash_kda_split2_C
    command(["cuobjdump", "--dump-resource-usage", flash_kda_split2_C.__file__], "challenge-resources", timeout=90)
    precision(heads, candidate)
    # Cover the exact state interface independently of the small naive sweep.
    for state_dtype in (torch.bfloat16, torch.float32):
        for has_in, has_out in ((True, True), (True, False), (False, True), (False, False)):
            args, kw = inputs(heads=4, lengths=(4, 8, 12))
            kw["initial_state"] = kw["initial_state"].to(state_dtype) if has_in else None
            kw["final_state"] = kw["final_state"].to(state_dtype) if has_out else None
            flash_kda.fwd(*args, **kw)
            expected = args[-1].clone()
            expected_h = kw["final_state"].clone() if has_out else None
            candidate.fwd(*args, **kw)
            torch.cuda.synchronize()
            assert torch.equal(args[-1], expected)
            assert not has_out or torch.equal(kw["final_state"], expected_h)
    records = []
    for h in (12, 96):
        for lengths in ((8192,), (1300, 547, 2048, 963, 271, 3063), (1024,) * 8):
            args, kw = inputs(h, lengths)
            record = dict(heads=h, lengths=lengths,
                          baseline=timings(lambda: flash_kda.fwd(*args, **kw)),
                          split2=timings(lambda: candidate.fwd(*args, **kw)))
            record["speedup"] = record["baseline"]["mean_ms"] / record["split2"]["mean_ms"]
            records.append(record)
            (OUT / "challenge-timings.json").write_text(json.dumps(records, indent=2))
            print(json.dumps({k: v for k, v in record.items() if k not in ("baseline", "split2")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--heads", type=int, default=96)
    a = p.parse_args()
    globals()[a.mode](a.heads)
