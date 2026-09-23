"""Build a separate value-column split experiment from the pinned FlashKDA clone.

Only the generated /opt/c1-value-split tree is modified. Each split CTA owns a
disjoint range of value columns throughout the sequence. K1, arithmetic order,
BF16 rounding points and each warp's 32 value columns stay unchanged. Input TMA
and state allocation are deliberately replicated in this first experiment;
profiling must decide whether increased grid parallelism covers that cost.
"""
import argparse
import difflib
from pathlib import Path
import shutil
import subprocess


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def generate(source: Path, target: Path, splits: int):
    assert splits in (1, 2, 4)
    assert not target.exists(), f"Refusing to overwrite {target}"
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git", "build", "*.egg-info"))
    changed = {}
    path = target / "csrc/smxx/fwd_launch.cu"
    original = path.read_text()
    modified = replace_once(original, "constexpr int kK2Threads = 32 * 2 + 128;",
                            f"constexpr int kK2Threads = 32 * 2 + 128 / {splits};")
    modified = replace_once(modified, "dim3 grid_k2(N, H);", f"dim3 grid_k2(N, H, {splits});")
    modified = replace_once(modified, "out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles",
                            "out_ptr, final_state_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles")
    changed[path.relative_to(target)] = (original, modified)

    path = target / "csrc/smxx/fwd_kernel2.cuh"
    original = path.read_text()
    modified = replace_once(original, "cutlass::bfloat16_t* out_raw_ptr,",
                            "cutlass::bfloat16_t* out_raw_ptr,\n    void* final_state_raw_ptr,")
    modified = replace_once(modified, "constexpr int kComputeThreads = 128;",
                            f"constexpr int kValueSplits = {splits};\n    constexpr int kComputeThreads = 128 / kValueSplits;")
    modified = replace_once(modified, "const int warp_id = compute_tid / 32;",
                            "const int warp_id = compute_tid / 32 + int(blockIdx.z) * (4 / kValueSplits);")
    begin = modified.index("    if (warp_role == WarpRole::STORE && lane_predicate) {")
    end = modified.index("    __syncthreads();\n#endif\n}", begin)
    # The elected lane participates in the one-consumer pipeline; all store-warp
    # lanes cooperatively write the selected value columns, then release the slot.
    # No CTA writes another split's state or output, including varlen tails.
    store = r'''    if (warp_role == WarpRole::STORE) {
        constexpr int kValues = D / kValueSplits;
        int value_begin = int(blockIdx.z) * kValues;
        int lane = threadIdx.x % 32;
        StorePipelineState out_read;
        for (int t = 0; t < t_tiles; ++t) {
            if (lane_predicate) store_pipeline.consumer_wait(out_read);
            __syncwarp();
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.output[stage].out.begin()), VOLayout{});
            for (int i = lane; i < actual_len * kValues; i += 32) {
                int row = i / kValues;
                int col = value_begin + i % kValues;
                int64_t base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                out_raw_ptr[base + col] = s_out(row, col);
            }
            __syncwarp();
            if (lane_predicate) store_pipeline.consumer_release(out_read);
            ++out_read;
        }
        if constexpr (HasStateOut) {
            Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            for (int i = lane; i < kValues * D; i += 32) {
                int value = value_begin + i / D;
                int key = i % D;
                int64_t offset = (int64_t(seq_idx) * H + head_idx) * D * D + value * D + key;
                if constexpr (StateFP32) {
                    static_cast<float*>(final_state_raw_ptr)[offset] = float(s_state(value, key));
                } else {
                    static_cast<BF16*>(final_state_raw_ptr)[offset] = s_state(value, key);
                }
            }
        }
    }

'''
    modified = modified[:begin] + store + modified[end:]
    changed[path.relative_to(target)] = (original, modified)

    path = target / "setup.py"
    original = path.read_text()
    modified = original.replace("flash_kda_C", f"flash_kda_split{splits}_C")
    # The copied tree has no .git; cutlass is copied from the pinned recursive clone.
    modified = modified.replace('subprocess.run(["git", "submodule", "update", "--init", "cutlass"])', "")
    modified = modified.replace("name='flash_kda',", f"name='flash_kda_split{splits}',")
    modified = modified.replace("packages=['flash_kda'],", f"packages=['flash_kda_split{splits}'],")
    changed[path.relative_to(target)] = (original, modified)
    for relative, (before, after) in changed.items():
        (target / relative).write_text(after)
    wrapper = target / "flash_kda" / "__init__.py"
    wrapper.write_text(wrapper.read_text().replace("from flash_kda_C", f"from flash_kda_split{splits}_C"))
    wrapper.parent.rename(target / f"flash_kda_split{splits}")
    diff = "".join("".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                             fromfile=f"a/{p}", tofile=f"b/{p}"))
                   for p, (a, b) in changed.items())
    (target / "challenge.patch").write_text(diff)
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("/opt/FlashKDA"))
    p.add_argument("--target", type=Path, default=Path("/opt/c1-value-split"))
    p.add_argument("--splits", type=int, default=2)
    p.add_argument("--build", action="store_true")
    a = p.parse_args()
    generate(a.source, a.target, a.splits)
    if a.build:
        subprocess.run(["python", "-m", "pip", "install", "-v", "--no-build-isolation", "--no-deps", str(a.target)],
                       cwd=a.target, check=True)
