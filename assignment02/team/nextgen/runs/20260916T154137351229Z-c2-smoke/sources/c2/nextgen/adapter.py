"""Prepared complete-chain adapter; supports the existing frozen suite interface."""
from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import candidate


@lru_cache(maxsize=1)
def extension():
    directory = Path(os.environ.get("C2_NEXTGEN_BUILD", "/tmp/nextgen-build"))
    modules = sorted(directory.glob("c2_nextgen_ext*.so"))
    if len(modules) != 1:
        raise RuntimeError(f"Expected one precompiled c2_nextgen_ext module in {directory}, got {modules}; run build.py first")
    spec = importlib.util.spec_from_file_location("c2_nextgen_ext", modules[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_splits(case):
    target = max(1, min(case["topk"], 256 // max(1, case["q"].shape[0] * case["num_kv_heads"])))
    return 1 << (target.bit_length() - 1)


def prepare(case, *, splits=None, merge="feature"):
    ext = extension()
    q = case["q"]
    splits = default_splits(case) if splits is None else splits
    parts = torch.empty((splits, *q.shape), dtype=q.dtype, device=q.device)
    lse = torch.empty((splits, *q.shape[:2]), dtype=torch.float32, device=q.device)
    output = torch.empty_like(q)
    empty = torch.empty(0, dtype=torch.float32, device=q.device)
    ks = case.get("k_scale") if case.get("k_scale") is not None else empty
    vs = case.get("v_scale") if case.get("v_scale") is not None else empty
    def partial():
        ext.partial(q, case["kv_cache"], case["topk_idx"], case["block_table"],
                    case["seq_lens"], ks, vs, parts, lse, case["decode_query_len"],
                    splits, case["sm_scale"])
    if merge == "feature":
        tile = 128 if splits <= 8 else 64
        def finish():
            candidate.merge_features(parts, lse, output, tile=tile, warps=1, use_pdl=False)
    elif merge == "original":
        def finish():
            candidate._merge_topk_attn_out_kernel[(q.shape[0], q.shape[1])](
                parts, lse, output, q.shape[2], *parts.stride(), *lse.stride(),
                *output.stride(), NUM_TOPK_CHUNKS=splits, USE_PDL=False)
    else:
        raise ValueError(merge)
    def chain():
        partial()
        finish()
        return output
    return {"chain": chain, "partial": partial, "merge": finish, "output": output,
            "workspace": (parts, lse), "splits": splits, "merge_kind": merge,
            "actual_backend": "tcgen05_bf16_qk_and_pv_smem_thread_feed",
            "kv_conversion": "fp8_to_bf16_scaled" if case["kv_cache"].dtype == torch.float8_e4m3fn else "bf16"}


def run(case):
    """Adapter for validation/run_validation.py --adapter nextgen.adapter:run."""
    prepared = prepare(case)
    value = prepared["chain"]()
    audit = os.environ.get("C2_NEXTGEN_AUDIT")
    if audit:
        spec = case.get("spec")
        record = {"case_id": getattr(spec, "case_id", None), "seed": case.get("seed"),
                  "variant": case.get("variant"), "actual_backend": prepared["actual_backend"],
                  "kv_conversion": prepared["kv_conversion"], "splits": prepared["splits"],
                  "fallback": False, "meaning": "kernel calls enqueued; numeric completion checked by outer validator"}
        with Path(audit).open("a") as handle:
            handle.write(json.dumps(record) + "\n")
    return value
