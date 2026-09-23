"""Independent fixed-pin upstream CUTLASS comparison, without full vLLM.

Only the unused CuTe sparse-prefill adapter is replaced with an explicit error
stub. The C++ decode planner, kernel, reduction and vLLM decode wrapper remain
unchanged. This is an external baseline, not a submitted optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import torch

import baseline_workload as bw


def load_cutlass():
    sys.path.insert(0, "/opt/msa/python")
    unused = types.ModuleType("fmha_sm100.sparse_fmha_adapter")
    def reject(*args, **kwargs):
        raise RuntimeError("This isolated experiment supports C++ decode only; CuTe prefill was not loaded")
    unused.sparse_fmha = reject
    unused.sparse_fmha_plan = reject
    sys.modules[unused.__name__] = unused
    from fmha_sm100 import api
    for name in ("vllm.config", "vllm.config.attention", "vllm.third_party", "vllm.third_party.fmha_sm100"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.config.attention"].MiniMaxM3MSADecodeBackend = str
    sys.modules["vllm.third_party.fmha_sm100.api"] = api
    name = "isolated_msa_cutlass_sparse_decode"
    path = bw.ROOT / "vllm_msa_ref/msa_cutlass_sparse_decode.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def error(out, ref):
    diff = out.double() - ref
    return dict(finite=bool(torch.isfinite(out).all()), max_abs=float(diff.abs().max()),
                nrmse=float(diff.norm() / ref.norm().clamp_min(1e-30)))


def run_shape(module, batch, tp, output, pdl):
    case = bw.inputs(batch, tp, "fp8", pdl)
    q_scale = .25
    q_fp8 = (case["q"].float() / q_scale).to(torch.float8_e4m3fn)
    effective_case = dict(case, q=(q_fp8.float() * q_scale).to(torch.bfloat16))
    original_gold = bw.reference(case)
    effective_gold = bw.reference(effective_case)
    seq_cpu = case["seq_lens"].cpu()
    cache = module.MSACutlassDecodePlanCache()
    def prepare():
        return module.prepare_decode_metadata(case["block_table"], case["seq_lens"], seq_cpu, 1,
                                              num_q_heads=64 // tp, num_kv_heads=4 // tp,
                                              page_size=128, topk_blocks=16, plan_cache=cache)
    begin = time.perf_counter()
    metadata = prepare()
    torch.cuda.synchronize()
    plan_cold_s = time.perf_counter() - begin
    out = torch.empty_like(case["q"])
    def attention():
        module.msa_cutlass_sparse_decode(q_fp8, case["kv_cache"], case["topk_idx"], out, metadata,
                                        scale=case["sm_scale"], q_scale_float=q_scale,
                                        k_scale_float=.25, v_scale_float=.5)
    begin = time.perf_counter()
    attention()
    torch.cuda.synchronize()
    first_attention_s = time.perf_counter() - begin
    correctness = dict(against_effective_fp8_q=error(out, effective_gold),
                       against_original_bf16_q=error(out, original_gold))
    bw.dump(output / "correctness.json", correctness)
    if not correctness["against_effective_fp8_q"]["finite"]:
        raise AssertionError(correctness)
    def quantize():
        q_fp8.copy_((case["q"].float() / q_scale).to(torch.float8_e4m3fn))
    def full():
        quantize()
        prepare()
        attention()
    baseline_out, calls = bw.capture(case)
    partial, merge = map(bw.launcher, calls)
    def baseline_chain():
        partial()
        merge()
    effective_baseline_out, effective_calls = bw.capture(effective_case)
    ep, em = map(bw.launcher, effective_calls)
    def effective_baseline():
        ep()
        em()
    # CUDA activity trace establishes actual launch count and grid-independent
    # kernel duration. It is diagnostic, not the primary timing distribution.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA]) as prof:
        attention()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output / "activity.json"))
    row = dict(batch=batch, tp=tp, dtype="fp8", pdl_triton=pdl,
               query_scales=dict(q=.25, k=.25, v=.5), correctness=correctness,
               baseline_correctness=error(baseline_out, original_gold),
               effective_baseline_correctness=error(effective_baseline_out, effective_gold),
               cold_plan_seconds=plan_cold_s, first_attention_seconds=first_attention_s,
               attention_graph=bw.graph_time(attention), metadata_graph=bw.graph_time(prepare),
               q_quantize_graph=bw.graph_time(quantize), full_graph=bw.graph_time(full),
               triton_original_q_graph=bw.graph_time(baseline_chain),
               triton_effective_q_graph=bw.graph_time(effective_baseline),
               plan_scalars={k:v for k,v in metadata.plan[3].items() if isinstance(v, (int,float,bool,str,type(None)))},
               plan_tensors={k:dict(shape=list(v.shape),dtype=str(v.dtype),bytes=v.numel()*v.element_size())
                             for k,v in metadata.plan[3].items() if isinstance(v,torch.Tensor)})
    bw.dump(output / "measurement.json", row)
    print(json.dumps(dict(batch=batch,tp=tp,correctness=correctness,
                         cutlass_us=row["attention_graph"]["median_us"],
                         cutlass_full_us=row["full_graph"]["median_us"],
                         triton_us=row["triton_original_q_graph"]["median_us"])), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/c2-cutlass"))
    parser.add_argument("--batches", default="16,32,64")
    parser.add_argument("--tps", default="1,4")
    parser.add_argument("--pdl", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    module = load_cutlass()
    rows = []
    for tp in map(int,args.tps.split(",")):
        for batch in map(int,args.batches.split(",")):
            directory = args.output / f"tp{tp}-b{batch}"
            directory.mkdir(exist_ok=True)
            rows.append(run_shape(module,batch,tp,directory,args.pdl))
            bw.dump(args.output / "measurements.json", rows)
    print("C2_CUTLASS_DONE", flush=True)


if __name__ == "__main__":
    main()
