"""Preliminary FlashKDA-only timings, following upstream CUDA-event timing.

This is not the full official benchmark or a correctness/FLA comparison.
"""
import json
import statistics

import torch
import torch.nn.functional as F
import flash_kda


@torch.inference_mode()
def run_case(heads, lengths):
    torch.manual_seed(0)
    shape = (1, sum(lengths), heads, 128)
    q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
            for _ in range(2)]
    v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
    a_log = torch.rand(heads, device="cuda")
    bias = torch.rand(heads, 128, device="cuda")
    h0 = torch.arange(len(lengths) * heads * 128 * 128, device="cuda", dtype=torch.float32)
    h0 = h0.reshape(len(lengths), heads, 128, 128).bfloat16()
    ht, out = torch.zeros_like(h0), torch.zeros_like(q)
    cu = (torch.tensor([0] + lengths, device="cuda", dtype=torch.int64).cumsum(0)
          if len(lengths) > 1 else None)

    def forward():
        # Match the public upstream API, including its workspace allocation.
        # Each invocation starts with the same h0; output state is not fed back.
        flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                      A_log=a_log, dt_bias=bias, lower_bound=-5.0,
                      initial_state=h0, final_state=ht, cu_seqlens=cu)

    for _ in range(30):
        forward()
    torch.cuda.synchronize()
    samples = []
    repeat_means = []
    for _ in range(5):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(200)]
        torch.cuda.synchronize()
        for start, end in zip(starts, ends):
            start.record()
            forward()
            end.record()
        torch.cuda.synchronize()
        batch = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        samples.extend(batch)
        repeat_means.append(statistics.mean(batch))
    finite = bool(torch.isfinite(out).all() and torch.isfinite(ht).all())
    record = dict(H=heads, D=128, seq_lens=lengths, state_dtype="bf16", seed=0,
                  warmup=30, iters=200, repeats=5, finite=finite,
                  mean_ms=statistics.mean(samples), median_ms=statistics.median(samples),
                  min_ms=min(samples), max_ms=max(samples), repeat_means_ms=repeat_means,
                  samples_ms=samples)
    print(json.dumps(record), flush=True)
    if not finite:
        raise RuntimeError("Nonfinite output/state; timings are not a valid baseline.")


if __name__ == "__main__":
    print(json.dumps(dict(torch=torch.__version__, cuda=torch.version.cuda,
                          gpu=torch.cuda.get_device_name(),
                          scope="preliminary FlashKDA-only; finite check is not correctness proof")),
          flush=True)
    for heads in (96, 12):
        for lengths in ([8192], [1024] * 8):
            run_case(heads, lengths)
    print("FLASHKDA_TIMING_DONE", flush=True)
