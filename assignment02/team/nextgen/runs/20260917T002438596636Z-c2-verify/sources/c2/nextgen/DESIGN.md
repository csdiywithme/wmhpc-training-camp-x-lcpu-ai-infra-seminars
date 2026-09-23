# C2 next-generation experiment log

## 2026-09-16 — revision 1: full partial, explicit shared-memory feed

Status: source authored; no CUDA compilation or GPU correctness/performance claim
at authoring time. GPU execution is orchestrated separately by the root task.
Existing vendored sources, candidate.py, reports, and frozen protocol are intact.

The first complete kernel keeps the upstream split policy and BF16 partial /
FP32 log2-LSE interface. One CTA processes one query-row/KV-head/split, looking up
only valid top-k slots through the actual block table. It masks causal future
tokens before loading K/V (including NaN-poisoned valid addresses), and writes
zero partial / negative-infinity LSE for empty splits. All-padding final output
remains ignored by the existing contract. All tensor/scale strides are honored.

Each page performs K Q^T followed by V^T P^T: both are M128 N16 K128 BF16 tcgen05
operations with FP32 accumulators. Q is retained in shared memory, A is reused
between K and transposed V after completion, and P is explicitly cast to BF16.
Scores travel through TMEM -> registers -> shared -> SIMT softmax; output uses
the same per-page online scaling/log2-LSE recurrence as the original partial.
The first version intentionally makes these data movements visible. It is not
expected to have optimal performance and does not equate valid instruction
shapes with higher throughput. It uses no TMA, cluster, PDL, or native FP8 MMA.

Only the issuer warp advances the UMMA barrier phase. A CTA barrier protects
every shared-buffer reuse. Allocation is 32 TMEM columns; SM100 and SM103
architectures are selected explicitly during CPU compilation. Ordinary thread
loads establish a control to which TMA supply can later be compared. One-page
CTAs have no next page to prefetch; no cross-page pipeline is claimed.

FP8 E4M3FN KV uses the existing represented-input contract: FP8 -> BF16, multiply
FP32 scale, round BF16, then BF16 MMA. Scalar and physical-token/head scales with
arbitrary backing strides are supported. Query and probability remain BF16.
Native FP8 is a future, separately validated numeric experiment.

## Build and execution interface

CPU build (CUDA 13.1, torch 2.10, ninja, C++17, CUTLASS headers):

```sh
python nextgen/build.py --arch 103a --cutlass /opt/FlashKDA/cutlass --output /tmp/nextgen-build
```

Use `--arch 100a` for B200 or `--arch 100a,103a` for both. No GPU query is needed
to compile. All source hashes, build options, module path, and SASS are saved.

GPU smoke (single/multiple pages, causal DQL, empty splits, FP8 scalar/token):

```sh
python nextgen/run.py --mode smoke --build-dir /tmp/nextgen-build --output /tmp/nextgen-artifacts/smoke
python nextgen/run.py --mode verify --build-dir /tmp/nextgen-build --output /tmp/nextgen-artifacts/verify --manifest /path/to/frozen/calibration.json
python nextgen/run.py --mode bench --build-dir /tmp/nextgen-build --output /tmp/nextgen-artifacts/bench --tps 1,4 --batches 1,16 --seeds 101
```

Verification requires the existing manifest, defaulting to
`/opt/c2/validation/frozen_calibration.json`; this runner never recalibrates or
widens the frozen policy limits. Performance compares the actual unmodified baseline, the existing
merge-only candidate, and tcgen05+the same merge in randomized interleaved hot
CUDA Graph measurements. All buffers are preallocated. Both eager and actual
post-graph outputs are checked against independent FP64 attention outside timing.
Raw samples and all observed negative results are retained; no result is implied
before the GPU run. A `--splits` override is exploratory, not a heldout-tuned
production dispatcher. `--merge original` isolates the new partial with the
original merge. The default is the existing feature-tiled merge.

Every smoke/benchmark row records the actual tcgen05 backend and KV conversion;
frozen validation keeps a separate `adapter_calls.jsonl` with the same case/seed/
variant keys. There is no fallback implementation in this revision. The audit
records launches, while the untouched validator separately checks completion.

Static review follow-up: smoke uses the predeclared hard caps as an exploratory
gate; benchmark additionally validates the original manifest digest and uses
its frozen per-family thresholds for baseline, old merge-only and new outputs.
Neither smoke nor a limited benchmark is labeled full acceptance. A full PASS
requires the existing validation runner and its expected record count.

## Bounded profiler entry (added before the first GPU result)

The root runner may invoke `run.py --mode profile --tps 4 --batches 1
--storages bf16 --seeds 101 --profile-target tcgen05`. It requires exactly one
case and warms all three comparison paths outside the NVTX range. Choose target
`baseline`, `merge_only`, or `tcgen05`; the selected full two-kernel chain is then
enclosed in a push/pop range named `c2_nextgen_profile_<target>`. Default is one
chain replay; `--profile-replays` explicitly changes it. The root owns the NCU
command, replay/cache policy and timeout; an NVTX include filter can isolate the
named range. Before/after outputs use the same independent gold and frozen family
bounds. `profile.json` records launch count and route but no performance speedup.
NCU duration must never be mixed with the separate randomized hot-Graph benchmark.

## First CPU build feedback

The first real CUDA compilation failed before any GPU launch: CUDA 13.1 did not
make `CUDART_INF_F` visible through the included runtime headers. The compiler
reported exactly three undefined uses (empty-split LSE and softmax negative
infinity). Revision 2 adds the owning `<math_constants.h>` header; arithmetic,
indexing, synchronization, and launch policy are unchanged. The original failed
build log is preserved by the root runner in run `20260916T154654373706Z-c2-smoke`.
No CUDA correctness or performance result is inferred from this compile repair.

Benchmark measurement refinement: after graph capture/warmup, three whole-graph
event samples estimate each path. All paths then use the same replay count,
chosen so the fastest estimated path reaches 10 ms per paired sample (maximum
64 replays). Raw graph estimates, replay cap, actual chosen replay count, and
`calls_per_graph * replays_per_sample` denominator are recorded. This removes
the previous sub-millisecond sample intervals without changing the kernel or
numerical gate. If the cap prevents 10 ms, the recorded estimate makes that
limitation visible. Samples remain randomized/interleaved across all three paths.

Next decision after first GPU run: repair any semantic/compiler failure first;
then inspect full-chain latency and resource usage before adding TMA, changing
the MMA layout, or grouping more pages per CTA. In particular, an isolated MMA
win is insufficient to select this complete dataflow.
