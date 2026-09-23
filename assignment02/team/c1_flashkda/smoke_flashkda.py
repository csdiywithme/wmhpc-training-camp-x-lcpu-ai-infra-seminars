"""Small exact comparison against the pinned upstream numerical reference."""
import sys

import torch
import torch.nn.functional as F
import flash_kda

sys.path.insert(0, "/opt/FlashKDA/tests")
from torch_ref import torch_ref


@torch.inference_mode()
def main():
    print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(), flush=True)
    for seed in (0, 1):
        torch.manual_seed(seed)
        shape = (1, 32, 2, 128)
        q, k = [F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
                for _ in range(2)]
        v, g = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        beta = torch.randn(shape[:-1], device="cuda", dtype=torch.bfloat16)
        a_log = torch.rand(2, device="cuda")
        bias = torch.rand(2, 128, device="cuda")
        h0 = torch.randn((1, 2, 128, 128), device="cuda", dtype=torch.bfloat16)
        out, ref = torch.zeros_like(q), torch.zeros_like(q)
        state, ref_state = torch.zeros_like(h0), torch.zeros_like(h0)
        kwargs = dict(A_log=a_log, dt_bias=bias, lower_bound=-5.0)
        flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                      initial_state=h0.clone(), final_state=state, **kwargs)
        torch.cuda.synchronize()
        torch_ref(q, k, v, g, beta, 128 ** -0.5, ref,
                  initial_state=h0.clone(), final_state=ref_state, **kwargs)
        for label, actual, expected in (("output", out, ref), ("state", state, ref_state)):
            exact = torch.equal(actual, expected)
            error = (actual.float() - expected.float()).abs().max().item()
            print(f"seed={seed} {label}: exact={exact} max_abs_error={error}", flush=True)
            if not torch.isfinite(actual).all() or not exact:
                raise AssertionError(f"seed={seed} {label} failed exact comparison")
    print("FLASHKDA_SMOKE_PASS (two small cases only)", flush=True)


if __name__ == "__main__":
    main()
