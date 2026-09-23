"""C2 input construction, independent FP64 gold and reusable evaluation.

Import from the c2_msa_decode directory: from validation import suite.
The adapter interface is callable(case: dict) -> output Tensor. Baseline leaves
vendored sources untouched and explicitly disables PDL through the harness shim.
This suite validates the represented-input attention, not model quantization loss.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PAGE, DIM, TOPK, GROUP = 128, 128, 16, 16
CALIBRATION_SEEDS = (11, 29)
HELDOUT_SEEDS = (101, 307)
POLICY_VERSION = "c2-represented-input-v1"
POLICY = {
    "row_rms_denominator_floor": 0.01,
    "element_atol": 0.02,
    "element_rtol": 0.02,
    "absolute_error_floor": 0.002,
    "absolute_error_cap": 0.02,
    "row_nrmse_floor": 0.02,
    "row_nrmse_cap": 0.10,
    "global_nrmse_floor": 0.02,
    "global_nrmse_cap": 0.05,
    "baseline_multiplier": 2.0,
    "absolute_error_margin": 0.0005,
}


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    kv_heads: int = 4
    seq_lens: tuple[int, ...] = (2305,)
    decode_query_len: int = 1
    storage: str = "bf16"  # bf16, fp8_scalar, fp8_token
    pattern: str = "random"  # random, zero_query, constant_value, sharp, cancellation, causal_probe
    strided_topk: bool = True
    strided_scales: bool = True
    metamorphic: bool = False

    @property
    def family(self):
        # Padding alone has no numeric calibration; include it with random cases.
        return self.storage + ":" + self.pattern


def build_specs(tier="quick"):
    """A reproducible smoke matrix or a larger frozen acceptance domain."""
    if tier not in ("quick", "full"):
        raise ValueError("tier must be quick or full")
    quick = [
        CaseSpec("tp1_bf16_boundaries", seq_lens=(1, 127, 128, 129), pattern="zero_query"),
        CaseSpec("tp4_bf16_cross_page", 1, (129, 130, 257, 2050), 2),
        CaseSpec("tp1_fp8_scalar_tail", 4, (129, 257, 2049, 2305), storage="fp8_scalar"),
        CaseSpec("tp4_fp8_token_tail", 1, (129, 257, 2049, 2305), storage="fp8_token"),
        CaseSpec("tp1_fp8_token_remap", storage="fp8_token", metamorphic=True),
        CaseSpec("tp4_fp8_scalar_padding", 1, (0, 129, 0, 2049), storage="fp8_scalar"),
        CaseSpec("tp1_all_padding", seq_lens=(0, 0, 0, 0)),
        CaseSpec("tp4_bf16_sharp", 1, (2305,), pattern="sharp"),
        CaseSpec("tp1_fp8_token_average", seq_lens=(127, 128, 129, 257),
                 storage="fp8_token", pattern="zero_query"),
        CaseSpec("tp1_bf16_remap", metamorphic=True),
        CaseSpec("tp4_bf16_future_leak", 1, (129, 257), 4, pattern="causal_probe"),
        CaseSpec("tp1_fp8_token_future_leak", 4, (129, 257), 2,
                 storage="fp8_token", pattern="causal_probe"),
    ]
    if tier == "quick":
        return quick
    result = list(quick)
    for heads in (4, 1):
        for storage in ("bf16", "fp8_scalar", "fp8_token"):
            prefix = f"h{heads}_{storage}"
            for batch in (1, 4, 8, 16):
                result.append(CaseSpec(f"{prefix}_b{batch}_long", heads, (8192,) * batch,
                                       storage=storage))
            for dql in (2, 4):
                result.append(CaseSpec(f"{prefix}_dql{dql}_edges", heads,
                                       (129, 257, 2049, 2050), dql, storage))
            result.append(CaseSpec(f"{prefix}_mixed_padding", heads, (0, 1, 128, 0),
                                   storage=storage))
            for pattern in ("zero_query", "constant_value", "sharp", "cancellation"):
                result.append(CaseSpec(f"{prefix}_{pattern}", heads, (257, 2305),
                                       storage=storage, pattern=pattern))
    return result


def _strided_scale(x):
    backing = torch.empty(x.shape[0], x.shape[1] * 2, dtype=x.dtype, device=x.device)
    backing[:, ::2] = x
    backing[:, 1::2] = -123.0  # A wrong stride must not accidentally load a valid scale.
    return backing[:, ::2]


def make_case(spec, seed=11, device="cuda"):
    """Build finite BF16 Q and BF16/FP8 KV, with request-private shuffled pages.

    FP8 uses E4M3FN, positive FP32 scales, and bounded represented inputs.
    The returned keys also match harness/ref_sdpa.py where that harness applies.
    """
    if isinstance(spec, dict):
        spec = CaseSpec(**{**spec, "seq_lens": tuple(spec["seq_lens"])})
    if spec.kv_heads not in (1, 4) or spec.decode_query_len < 1:
        raise ValueError("This frozen domain supports 1/4 KV heads and positive dql")
    if spec.storage not in ("bf16", "fp8_scalar", "fp8_token"):
        raise ValueError(spec.storage)
    if not spec.seq_lens or any(s < 0 or 0 < s < spec.decode_query_len for s in spec.seq_lens):
        raise ValueError("Non-padding sequence lengths must contain all decode queries")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.tensor(spec.seq_lens, dtype=torch.int32)
    nblocks = [(s + PAGE - 1) // PAGE for s in spec.seq_lens]
    num_pages = max(1, sum(nblocks))
    table = torch.full((len(nblocks), max(1, max(nblocks))), -1, dtype=torch.int32)
    permutation = torch.randperm(num_pages, generator=generator)
    cursor = 0
    for request, count in enumerate(nblocks):
        table[request, :count] = permutation[cursor:cursor + count].to(torch.int32)
        cursor += count
    total_q = len(nblocks) * spec.decode_query_len
    hq = spec.kv_heads * GROUP
    query = torch.randn(total_q, hq, DIM, generator=generator) * 0.5
    values = torch.randn(num_pages, spec.kv_heads, PAGE, 2 * DIM, generator=generator) * 0.5
    if spec.pattern == "zero_query":
        query.zero_()
    elif spec.pattern == "sharp":
        query.mul_(16)
    elif spec.pattern == "constant_value":
        # Constant *effective* V is restored after quantization below.
        values[..., DIM:] = 0.75
    elif spec.pattern == "cancellation":
        query.zero_()
        signs = torch.where(torch.arange(PAGE) % 2 == 0, 1.0, -1.0)
        values[..., DIM:] = signs[None, None, :, None] * 0.5 + 0.0005
    elif spec.pattern == "causal_probe":
        query.zero_()
        values[..., DIM:] = 0
        for request, length in enumerate(spec.seq_lens):
            if length:
                # Only the final query may see this nonzero value. An off-by-one
                # causal leak is large compared with the fixed absolute cap.
                physical = int(table[request, (length - 1) // PAGE])
                values[physical, :, (length - 1) % PAGE, DIM:] = 8
    elif spec.pattern != "random":
        raise ValueError(spec.pattern)
    # Every valid prefix is unique and contains the current page. Ignored slots
    # alternate sentinels to expose an incorrect sentinel-based loop bound.
    indices = torch.full((spec.kv_heads, total_q, TOPK), -1, dtype=torch.int32)
    indices[..., 1::2] = -73
    for token in range(total_q):
        request, local = divmod(token, spec.decode_query_len)
        visible = max(spec.seq_lens[request] - spec.decode_query_len + local + 1, 0)
        blocks = (visible + PAGE - 1) // PAGE
        count = min(TOPK, blocks)
        for head in range(spec.kv_heads):
            if not count:
                continue
            chosen = torch.randperm(blocks, generator=generator)[:count]
            if not bool((chosen == blocks - 1).any()):
                chosen[0] = blocks - 1
            indices[head, token, :count] = chosen.to(torch.int32)
    if spec.strided_topk:
        indices = indices.permute(1, 0, 2).contiguous().permute(1, 0, 2)

    kscale = vscale = None
    if spec.storage == "fp8_scalar":
        kscale = torch.tensor(0.37, dtype=torch.float32)
        vscale = torch.tensor(0.63, dtype=torch.float32)
    elif spec.storage == "fp8_token":
        token = torch.arange(num_pages * PAGE, dtype=torch.float32)[None, :]
        head = torch.arange(spec.kv_heads, dtype=torch.float32)[:, None]
        kscale = 0.17 + 0.041 * head + 0.013 * (token % 17) + 0.019 * ((token // PAGE) % 7)
        vscale = 0.31 + 0.037 * head + 0.017 * (token % 13) + 0.023 * ((token // PAGE) % 5)
    if kscale is None:
        cache = values.to(torch.bfloat16)
    else:
        cache = torch.empty_like(values, dtype=torch.float8_e4m3fn)
        if kscale.numel() == 1:
            broadcast_k, broadcast_v = kscale, vscale
        else:
            broadcast_k = kscale.reshape(spec.kv_heads, num_pages, PAGE).permute(1, 0, 2)[..., None]
            broadcast_v = vscale.reshape(spec.kv_heads, num_pages, PAGE).permute(1, 0, 2)[..., None]
        cache[..., :DIM] = (values[..., :DIM] / broadcast_k).to(cache.dtype)
        cache[..., DIM:] = (values[..., DIM:] / broadcast_v).to(cache.dtype)
        if spec.pattern == "constant_value":
            # Use a constant scale so that this analytic property concerns
            # effective V rather than pre-quantization V.
            vscale.fill_(0.5)
            cache[..., DIM:] = torch.full_like(values[..., DIM:], 1.5).to(cache.dtype)
    target = torch.device(device)
    kscale = kscale.to(target) if kscale is not None else None
    vscale = vscale.to(target) if vscale is not None else None
    if spec.storage == "fp8_token" and spec.strided_scales:
        kscale, vscale = _strided_scale(kscale), _strided_scale(vscale)
    case = {
        "q": query.to(device=target, dtype=torch.bfloat16), "kv_cache": cache.to(target),
        "topk_idx": indices.to(target), "block_table": table.to(target),
        "seq_lens": lengths.to(target), "num_kv_heads": spec.kv_heads,
        "gqa_group_size": GROUP, "head_dim": DIM, "topk": TOPK,
        "decode_query_len": spec.decode_query_len, "sm_scale": DIM ** -0.5,
        "k_scale": kscale, "v_scale": vscale, "spec": spec, "seed": seed,
        "family": spec.family, "variant": "original",
    }
    case["active_rows"] = (case["seq_lens"] > 0).repeat_interleave(spec.decode_query_len)
    return case


@lru_cache(maxsize=1)
def _baseline_module():
    # Do not load an unrelated installed vLLM version. This shim explicitly sets
    # PDL=False; production Graph/PDL integration is a separate validation item.
    path = ROOT / "harness" / "vllm_shim.py"
    descriptor = importlib.util.spec_from_file_location("_c2_validation_shim", path)
    module = importlib.util.module_from_spec(descriptor)
    descriptor.loader.exec_module(module)
    return module.load_sparse_attn()


def baseline(case):
    output = torch.empty_like(case["q"])
    _baseline_module().minimax_m3_sparse_attn_decode(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
        case["seq_lens"], case["num_kv_heads"], case["sm_scale"], output,
        case["decode_query_len"], k_scale=case.get("k_scale"), v_scale=case.get("v_scale"),
    )
    return output


def _effective_page(case, page, head, count, raw_scale_gold=False):
    """Independent dequant: match FP32-scale multiply and Q-dtype rounding."""
    d = case["head_dim"]
    packed = case["kv_cache"][page, head, :count]
    is_fp8 = str(packed.dtype).startswith("torch.float8_")
    outputs = []
    for start, scale_name in ((0, "k_scale"), (d, "v_scale")):
        source = packed[:, start:start + d]
        scale = case.get(scale_name) if is_fp8 else None
        if scale is None:
            effective = source.to(torch.float64)
        else:
            if scale.numel() != 1:
                scale = scale[head, page * PAGE:page * PAGE + count, None]
            if raw_scale_gold:
                effective = source.to(torch.float64) * scale.to(torch.float64)
            else:
                # FP8 conversion is exact for the finite values in this domain;
                # retain the explicit first cast to document the kernel contract.
                effective = (source.to(case["q"].dtype).float() * scale.float()).to(case["q"].dtype).double()
        outputs.append(effective)
    return outputs


@torch.inference_mode()
def gold(case, *, raw_scale_gold=False):
    """Explicit gathered FP64 attention. Padding rows are zero plus active mask.

    Default: BF16-dequant represented-input gold. raw_scale_gold=True instead
    uses exact FP8 value * FP64 scale; it measures a different precision contract.
    No SDPA, candidate code, online recurrence, or split merge is used.
    """
    query = case["q"].double()
    output = torch.zeros_like(query)
    table = case["block_table"].detach().cpu().tolist()
    indices = case["topk_idx"].detach().cpu().tolist()
    lengths = case["seq_lens"].detach().cpu().tolist()
    dql, group = case["decode_query_len"], case["gqa_group_size"]
    for token in range(query.shape[0]):
        request, local = divmod(token, dql)
        visible = max(lengths[request] - dql + local + 1, 0)
        if not visible:
            continue
        blocks = (visible + PAGE - 1) // PAGE
        count = min(case["topk"], blocks)
        for head in range(case["num_kv_heads"]):
            selected = indices[head][token][:count]
            if len(set(selected)) != count or any(b < 0 or b >= blocks for b in selected):
                raise ValueError("Invalid or duplicate logical page in active top-k prefix")
            keys, values = [], []
            for logical in selected:
                physical = table[request][logical]
                if not 0 <= physical < case["kv_cache"].shape[0]:
                    raise ValueError("Invalid physical page in block table")
                n = min(PAGE, visible - logical * PAGE)
                key, value = _effective_page(case, physical, head, n, raw_scale_gold)
                keys.append(key)
                values.append(value)
            key, value = torch.cat(keys), torch.cat(values)
            qgroup = query[token, head * group:(head + 1) * group]
            logits = (qgroup @ key.T) * case["sm_scale"]
            weights = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
            weights = weights / weights.sum(dim=-1, keepdim=True)
            output[token, head * group:(head + 1) * group] = weights @ value
    return output


@torch.inference_mode()
def evaluate(case, output, reference=None):
    """Report active-row accuracy; padding NaNs are counted but not rejected."""
    if output.shape != case["q"].shape or output.dtype != case["q"].dtype:
        raise ValueError("Adapter output must have Q shape and dtype")
    if output.device != case["q"].device:
        raise ValueError("Adapter output must be on Q device")
    reference = gold(case) if reference is None else reference
    active = case.get("active_rows")
    if active is None:
        active = (case["seq_lens"] > 0).repeat_interleave(case["decode_query_len"])
    actual, expected = output[active].double(), reference[active].double()
    nonfinite = int((~torch.isfinite(actual)).sum().item())
    metrics = {
        "active_rows": int(active.sum().item()), "active_nonfinite": nonfinite,
        "padding_nonfinite": int((~torch.isfinite(output[~active])).sum().item()),
        "max_abs": 0.0, "rmse": 0.0, "global_nrmse": 0.0,
        "row_nrmse_max": 0.0, "row_nrmse_p50": 0.0, "row_nrmse_p95": 0.0,
        "row_nrmse_p99": 0.0, "elementwise_scaled_max": 0.0,
    }
    if not actual.numel():
        return metrics
    if not bool(torch.isfinite(expected).all()):
        raise ValueError("Gold has non-finite active output; input is outside this numeric domain")
    if nonfinite:
        return metrics
    error = actual - expected
    rms = error.square().mean(dim=-1).sqrt()
    ref_rms = expected.square().mean(dim=-1).sqrt()
    row = rms / ref_rms.clamp_min(POLICY["row_rms_denominator_floor"])
    flat = row.flatten()
    metrics.update(
        max_abs=error.abs().max().item(), rmse=error.square().mean().sqrt().item(),
        global_nrmse=(error.norm() / expected.norm().clamp_min(1e-12)).item(),
        row_nrmse_max=flat.max().item(),
        row_nrmse_p50=torch.quantile(flat, 0.50).item(),
        row_nrmse_p95=torch.quantile(flat, 0.95).item(),
        row_nrmse_p99=torch.quantile(flat, 0.99).item(),
        elementwise_scaled_max=(error.abs() / (POLICY["element_atol"] +
                               POLICY["element_rtol"] * expected.abs())).max().item(),
    )
    return metrics


def variants(case):
    """Return mathematically equivalent cases; future poisoning is dql=1 only."""
    result = []
    reordered = dict(case)
    reordered["topk_idx"] = case["topk_idx"].clone()
    lengths = case["seq_lens"].cpu().tolist()
    for token in range(case["q"].shape[0]):
        request, local = divmod(token, case["decode_query_len"])
        visible = max(lengths[request] - case["decode_query_len"] + local + 1, 0)
        count = min(case["topk"], (visible + PAGE - 1) // PAGE)
        reordered["topk_idx"][:, token, :count] = case["topk_idx"][:, token, :count].flip(-1)
    reordered["variant"] = "selected_order"
    result.append(reordered)
    remapped = dict(case)
    pages = case["kv_cache"].shape[0]
    order = torch.arange(pages - 1, -1, -1, device=case["q"].device)
    # Byte indexing works for all Torch FP8 backends and preserves encoded values.
    remapped["kv_cache"] = case["kv_cache"].view(torch.uint8)[order].contiguous().view(case["kv_cache"].dtype)
    remapped["block_table"] = torch.where(case["block_table"] >= 0,
                                          pages - 1 - case["block_table"], case["block_table"])
    for name in ("k_scale", "v_scale"):
        scale = case.get(name)
        if scale is not None and scale.numel() != 1:
            remapped[name] = scale.reshape(case["num_kv_heads"], pages, PAGE)[:, order].reshape_as(scale)
    remapped["variant"] = "physical_page_remap"
    result.append(remapped)
    if case["decode_query_len"] == 1:
        poisoned = dict(case)
        packed = case["kv_cache"].float()
        indices = case["topk_idx"].cpu().tolist()
        table = case["block_table"].cpu().tolist()
        # Per-request physical pages are disjoint in make_case. Do not use this
        # helper on an arbitrary case with shared physical pages across requests.
        for request, length in enumerate(lengths):
            blocks = (length + PAGE - 1) // PAGE
            for head in range(case["num_kv_heads"]):
                selected = set(indices[head][request][:min(TOPK, blocks)])
                for logical in range(blocks):
                    page = table[request][logical]
                    if logical not in selected:
                        packed[page, head] = float("nan")
                    else:
                        valid = min(PAGE, length - logical * PAGE)
                        packed[page, head, valid:] = float("nan")
        poisoned["kv_cache"] = packed.to(case["kv_cache"].dtype)
        poisoned["variant"] = "masked_data_poison"
        result.append(poisoned)
    return result


def source_hashes():
    paths = [Path(__file__), ROOT / "validation" / "ACCEPTANCE.md",
             ROOT / "validation" / "run_validation.py",
             ROOT / "vllm_msa_ref" / "sparse_attn.py", ROOT / "harness" / "vllm_shim.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def protocol(tier):
    return {"policy_version": POLICY_VERSION, "policy": POLICY,
            "calibration_seeds": CALIBRATION_SEEDS, "heldout_seeds": HELDOUT_SEEDS,
            "tier": tier, "specs": [asdict(s) for s in build_specs(tier)], "source_hashes": source_hashes()}


def protocol_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
