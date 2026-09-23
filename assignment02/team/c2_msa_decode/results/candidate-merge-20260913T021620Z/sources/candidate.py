# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Derived from the pinned d4da0c5 decode kernel; additions are the C2 experiment.
"""Sub-page split decode with configurable staging and standard online softmax.

The physical KV page is always 128 tokens. Arithmetic tiles may be 32/64/128.
The exported run(case) adapter uses a policy selected only after GPU experiments.
"""
import torch
import triton
import triton.language as tl

_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)

@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(
            16, triton.next_power_of_2(args["gqa_group_size"])
        ),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _subpage_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    kv_cache_ptr,  # main cache: [num_blocks, num_kv_heads, 128, 2*head_dim]
    k_scale_ptr,
    v_scale_ptr,
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kv_blk,
    stride_kv_h,
    stride_kv_pos,
    stride_kv_d,
    stride_ks_h,
    stride_ks_t,
    stride_vs_h,
    stride_vs_t,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # computation tile (32/64/128), physical page stays 128
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_FP8: tl.constexpr,  # fp8 KV cache: dequantize K/V to q.dtype on load
    KV_SCALE_MODE: tl.constexpr,  # 0: none, 1: scalar, 2: [kv_head, token]
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # split-K over the topk dimension: pid(0) folds (query-token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    tiles_per_page: tl.constexpr = 128 // BLOCK_SIZE_K
    total_tiles = max_topk * tiles_per_page
    chunk_size_topk = (total_tiles + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start_topk = pid_c * chunk_size_topk
    chunk_end_compiletime = chunk_start_topk + chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Valid block count from seq_len (no sentinel): min(topk, cdiv(kv_len, blk)).
    idx_base = t_ptr + pid_kh * stride_th + pid_b * stride_tn
    num_blocks = (kv_len + 127) // 128
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end_topk = tl.minimum(chunk_end_compiletime, real_topk * tiles_per_page)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qn + pid_h * stride_qh,
        shape=(gqa_group_size, head_dim),
        strides=(stride_qh, stride_qd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")

    for tile_index in tl.range(chunk_start_topk, chunk_end_topk):
        slot = tile_index // tiles_per_page
        tile_offset = (tile_index % tiles_per_page) * BLOCK_SIZE_K
        blk = tl.load(idx_base + slot * stride_tk).to(tl.int32)
        c = blk * 128 + tile_offset
        if c < kv_len:
            page = tl.load(bt_row + blk).to(tl.int64)
            pos = c + off_n
            pos_mask = pos < kv_len
            k = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk + tile_offset * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_n[None, :] * stride_kv_pos
                + off_d[:, None] * stride_kv_d,
                mask=d_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                k = k.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    k = (k * tl.load(k_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    k_scale = tl.load(
                        k_scale_ptr
                        + pid_kh * stride_ks_h
                        + (page * 128 + tile_offset + off_n) * stride_ks_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    k = (k * k_scale[None, :]).to(q.dtype)
            qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            qk += tl.dot(q, k) * sm_scale_log2e
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
            v = tl.load(
                kv_cache_ptr
                + page * stride_kv_blk + tile_offset * stride_kv_pos
                + pid_kh * stride_kv_h
                + off_n[:, None] * stride_kv_pos
                + (head_dim + off_d[None, :]) * stride_kv_d,
                mask=pos_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if USE_FP8:
                v = v.to(q.dtype)
                if KV_SCALE_MODE == 1:
                    v = (v * tl.load(v_scale_ptr)).to(q.dtype)
                elif KV_SCALE_MODE == 2:
                    v_scale = tl.load(
                        v_scale_ptr
                        + pid_kh * stride_vs_h
                        + (page * 128 + tile_offset + off_n) * stride_vs_t,
                        mask=pos_mask,
                        other=1.0,
                    )
                    v = (v * v_scale[:, None]).to(q.dtype)
            acc_o += tl.dot(p.to(v.dtype), v)
            m_i = m_ij
            l_i = l_i * alpha + l_ij

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    # Empty chunks for active rows must store zero output; otherwise the merge
    # can hit 0 * NaN. All-empty padded rows may still produce NaNs in merge.
    scale = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
    lse_i = tl.where(l_i > 0.0, m_i + tl.log2(l_i), float("-inf"))
    acc_o = acc_o * scale[:, None]
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(gqa_group_size, head_dim),
        strides=(stride_o_h, stride_o_d),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_D),
        order=(1, 0),
    )
    tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1))
    lse_ptrs = tl.make_block_ptr(
        base=lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h,
        shape=(gqa_group_size,),
        strides=(stride_l_h,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE_H,),
        order=(0,),
    )
    tl.store(lse_ptrs, lse_i.to(lse_ptr.dtype.element_ty), boundary_check=(0,))


@triton.heuristics(
    {"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])}
)
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    # NOTE: assume seq_lens is safe to load before gdc_wait()
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_o_b + pid_h * stride_o_h,
        shape=(NUM_TOPK_CHUNKS, head_dim),
        strides=(stride_o_c, stride_o_d),
        offsets=(0, 0),
        block_shape=(NUM_TOPK_CHUNKS, BLOCK_SIZE_D),
        order=(1, 0),
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, boundary_check=(0, 1), padding_option="zero")
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp2(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = (
        out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    )
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------
_KV_SCALE_NONE = 0
_KV_SCALE_SCALAR = 1
_KV_SCALE_PER_TOKEN_HEAD = 2


@triton.jit
def _merge_feature_tiles(
    partial, lse, output,
    stride_pc: tl.constexpr, stride_pb: tl.constexpr, stride_ph: tl.constexpr,
    stride_pd: tl.constexpr, stride_lc: tl.constexpr, stride_lb: tl.constexpr,
    stride_lh: tl.constexpr, stride_ob: tl.constexpr, stride_oh: tl.constexpr,
    stride_od: tl.constexpr, D: tl.constexpr, S: tl.constexpr,
    TILE_D: tl.constexpr, USE_PDL: tl.constexpr,
):
    # More independent output-feature CTAs, with a one-warp option that avoids
    # cross-warp shared-memory reductions observed in the baseline NCU profile.
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    split = tl.arange(0, S)
    dim = tile * TILE_D + tl.arange(0, TILE_D)
    logs = tl.load(lse + split * stride_lc + row * stride_lb + head * stride_lh)
    maximum = tl.max(logs, axis=0)
    weights = tl.exp2(logs - maximum)
    weights /= tl.sum(weights, axis=0)
    values = tl.load(partial + split[:, None] * stride_pc + row * stride_pb +
                     head * stride_ph + dim[None, :] * stride_pd,
                     mask=dim[None, :] < D, other=0).to(tl.float32)
    result = tl.sum(values * weights[:, None], axis=0)
    tl.store(output + row * stride_ob + head * stride_oh + dim * stride_od,
             result, mask=dim < D)


def merge_features(partial, lse, output, tile=32, warps=1, use_pdl=False):
    assert tile in (16, 32, 64, 128) and warps in (1, 2, 4)
    s, rows, heads, d = partial.shape
    return _merge_feature_tiles[(rows, heads, triton.cdiv(d, tile))](
        partial, lse, output, *partial.stride(), *lse.stride(), *output.stride(),
        D=d, S=s, TILE_D=tile, USE_PDL=use_pdl, num_warps=warps,
        **({"launch_pdl": True} if use_pdl else {}))


def _kv_scale_args(
    output: torch.Tensor,
    num_kv_heads: int,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int]:
    if k_scale is None and v_scale is None:
        return output, output, 0, 0, 0, 0, _KV_SCALE_NONE
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale must be both provided or both None")
    if k_scale.device != output.device or v_scale.device != output.device:
        raise ValueError("k_scale and v_scale must be on the same device as output")
    if k_scale.numel() == 1 and v_scale.numel() == 1:
        return k_scale, v_scale, 0, 0, 0, 0, _KV_SCALE_SCALAR
    if k_scale.dim() == 2 and v_scale.dim() == 2:
        if k_scale.shape[0] != num_kv_heads or v_scale.shape[0] != num_kv_heads:
            raise ValueError(
                "per-token/head KV scales must have shape "
                f"[{num_kv_heads}, max_kv_tokens]"
            )
        if k_scale.shape != v_scale.shape:
            raise ValueError("k_scale and v_scale must have matching shapes")
        return (
            k_scale,
            v_scale,
            k_scale.stride(0),
            k_scale.stride(1),
            v_scale.stride(0),
            v_scale.stride(1),
            _KV_SCALE_PER_TOKEN_HEAD,
        )
    raise ValueError(
        "MiniMax-M3 sparse attention supports scalar KV scales or "
        "[num_kv_heads, max_kv_tokens] per-token/head scales"
    )


@torch.no_grad()
def decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, num_kv_heads, 128, 2*head_dim]
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    *, token_tile: int = 64, splits: int | None = None, num_warps: int = 4,
    num_stages: int = 1, use_pdl: bool = False, workspace=None,
    merge_tile: int | None = None, merge_warps: int = 1,
) -> None:
    """GQA block-sparse attention for decode (split-K over the top-k blocks)."""
    total_q, num_heads, head_dim = q.shape
    assert total_q == seq_lens.shape[0] * decode_query_len
    max_topk = topk_idx.shape[-1]
    gqa_group_size = num_heads // num_kv_heads
    use_fp8 = kv_cache.dtype in _FP8_DTYPES
    (
        k_scale_arg,
        v_scale_arg,
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        kv_scale_mode,
    ) = (
        _kv_scale_args(output, num_kv_heads, k_scale, v_scale)
        if use_fp8
        else (
            output,
            output,
            0,
            0,
            0,
            0,
            _KV_SCALE_NONE,
        )
    )
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    # split-K over the selected blocks; chunk count is shape-constant (cuda graph).
    assert token_tile in (32, 64, 128)
    TARGET_GRID = 256
    total_tiles = max_topk * (128 // token_tile)
    target = max(1, min(total_tiles, TARGET_GRID // max(1, total_q * num_kv_heads)))
    num_topk_chunks = splits if splits is not None else 1 << (target.bit_length() - 1)
    assert num_topk_chunks > 0 and num_topk_chunks & (num_topk_chunks - 1) == 0
    if workspace is None:
        o_partial = torch.empty(
            num_topk_chunks, total_q, num_heads, head_dim, dtype=q.dtype, device=q.device
        )
        lse_partial = torch.empty(
            num_topk_chunks, total_q, num_heads, dtype=torch.float32, device=q.device
        )
    else:
        o_partial, lse_partial = workspace
        assert o_partial.shape == (num_topk_chunks, total_q, num_heads, head_dim)
        assert lse_partial.shape == (num_topk_chunks, total_q, num_heads)
    grid = (total_q * num_topk_chunks, num_kv_heads)
    _subpage_decode_kernel[grid](
        q,
        kv_cache,
        k_scale_arg,
        v_scale_arg,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        stride_ks_h,
        stride_ks_t,
        stride_vs_h,
        stride_vs_t,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=token_tile,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_FP8=use_fp8,
        KV_SCALE_MODE=kv_scale_mode,
        USE_PDL=use_pdl,
        num_warps=num_warps, num_stages=num_stages,
        **pdl_launch,
    )
    if merge_tile is not None:
        merge_features(o_partial, lse_partial, output, merge_tile, merge_warps, use_pdl)
        return
    merge_grid = (total_q, num_heads)
    _merge_topk_attn_out_kernel[merge_grid](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )


def run(case, config=None):
    config = config or dict(token_tile=64, num_warps=4, num_stages=1)
    output = torch.empty_like(case["q"])
    decode(case["q"], case["kv_cache"], case["topk_idx"], case["block_table"],
           case["seq_lens"], case["num_kv_heads"], case["sm_scale"], output,
           case["decode_query_len"], case.get("k_scale"), case.get("v_scale"), **config)
    return output
