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

The next real CPU build, run `20260916T154921959378Z-c2-smoke`, compiled and linked
successfully (approximately 42.33 seconds for the build subprocess, 38.77 seconds
in the build script's measured section). Ptxas reports 70 registers, zero
stack, zero spill loads/stores and one barrier for both the BF16 and FP8-storage
kernel instantiations. These are compilation/resource facts, not speed or
correctness results; the GPU smoke run follows separately. This revision remains
the explicit-thread-feed control (no TMA instruction is expected).

## First B300 GPU smoke

The same run completed all 15 planned cases (five input families times splits
1/4/16), each with eager and actual CUDA Graph output checks. All passed the
predeclared exploratory hard caps; the largest absolute error was approximately
0.00096827. Covered single/multiple pages, causal multi-query decode, empty
splits, mixed padding, FP8 scalar scales and strided physical-token/head scales.
The saved binary contains BF16 `UTCHMMA` instructions (64 static matches across
the binary's two instantiations), no `UTCQMMA` matches. This is not a dynamic
instruction count and not native FP8 MMA. No full frozen acceptance or speedup
is inferred from smoke. The root next runs the original frozen heldout protocol
using the same compiled binary before randomized three-path benchmarking.

## Frozen heldout acceptance

Run `20260916T155241079346Z-c2-verify` reused the successful binary and the
original frozen manifest. The untouched validation runner returned `PASS` with
168/168 baseline records and 168/168 candidate records, no failed records, and
protocol digest `430cf4da832b3fb3d2e3eeb2bff0c31959101a05703cc2ede89a1e10c4c54b02`.
The separate route audit contains 168 tcgen05 entries and no fallback entries.
Candidate maximum absolute error is 0.0092662716, maximum row NRMSE 0.0039739888,
maximum global NRMSE 0.0037378951, and active nonfinite count zero. Baseline's
maxima in this same heldout run match these aggregate values; this does not imply
the two full output tensors are bitwise identical for every case.
This is acceptance within the original represented-input domain, not production
PDL, other GPU architectures, compute-sanitizer, or model-quality certification.
The root now benchmarks all 16 TP/batch/storage shapes with two heldout seeds,
using seven randomized paired repeats per row as a bounded exploratory pass.

## First complete-chain performance result and follow-up variants

Run `20260916T155513744579Z-c2-bench` completed 32 shape/seed rows with seven
randomized paired repeats, all three complete chains on the same GPU and input.
The original dataflow is a large regression: geometric-mean speedup approximately
0.09399 relative to the original baseline and 0.08407 relative to the existing
merge-only candidate; baseline-relative range approximately 0.04386–0.14397.
Representative BF16 TP1 B1: original baseline 5.68 us, merge-only 5.06 us,
tcgen05 control 43.83 us. BF16 TP1 B16: 15.71 / 13.59 / 356.84 us. These are
complete-chain timings; instruction modernization alone delivered no gain.

Two measurable dataflow hypotheses now motivate isolated compile-time variants:

| Build variant | Global input traversal | Score/P temporary stride |
|---|---|---|
| `original` | Original MMA-partition ordinal | 16 FP32 elements |
| `coalesced` | Contiguous logical D across adjacent lanes | 16 |
| `coalesced_pad17` | Same as coalesced | 17 |

The coalesced variant handles Q, K and V, BF16 and FP8 storage, all valid pages
and causal tails. It keeps exactly the same effective-KV conversion/scale and
BF16 P representation. CuTe's right inverse maps a canonical logical index into
the existing shared MMA partition; V is globally read as (token,D), then placed
as (D,token) for PV. A CPU-only exhaustive coordinate check covers all 16,384 A
and 2,048 B entries after the real extension build, without touching a GPU.

The second variant changes only the score/P temporary row stride from 16 to 17.
For the current softmax loop's fixed head and consecutive token lanes, stride16
maps lanes onto only two of the 32 four-byte banks, while stride17 cycles through
all banks. This is a layout hypothesis, not a claimed measured stall fraction;
TMEM-copy and other accesses can respond differently and must also be checked.
Both QK and PV TMEM readback destinations use the chosen stride consistently.

Build with `--variant original|coalesced|coalesced_pad17`. The binary exports its
actual variant id, which drives the per-case route audit. The original variant
remains buildable; there is no automatic backend fallback or hidden dispatch.
Each new variant requires its own smoke/full frozen acceptance before speed
claims. The root is collecting a bounded NCU profile of the original control;
an initial clock-lock failure is a tooling result, not a kernel metric.

The corrected clock-unlocked profile, run `20260916T160153199885Z-c2-profile`,
reports original partial duration 81.728 us in NCU replay, 70,400 global-load
requests, and 546,880 actual versus 133,184 ideal shared wavefronts (413,696
excess). Dynamic shared is 57.600 KB; including the driver's 1.024 KB reservation,
per-block shared is 58.624 KB and the reported shared occupancy limit is three
blocks. No local spill is reported. These observations support investigating
thread-feed and shared layout; they do not isolate a unique cause. PC sampling
recorded 1,858 long-scoreboard and 135 short-scoreboard samples, which are not
wall-time percentages. NCU's derived `nway=353` aggregate is not interpreted as
an impossible literal 353-way conflict on 32 banks. Replay duration is kept
separate from the 43.83-us representative hot-Graph chain benchmark.

## Independent-input registration (not yet run)

`EXTRA_HELDOUT.json` pre-registers seeds 20260917/20261003, separately from
calibration 11/29 and now-public development/regression seeds 0/101/307. After the
final candidate and dispatch are frozen, `run.py --mode verify --extra-heldout`
runs 168 existing-domain records using the new seeds, plus 24 records from the
registered B2/DQL3 and B7/variable-length shapes across both KV-head partitions
and all three storage families. Expected total is 192 records per adapter.

This supplement reuses the existing validator's `run_matrix`, independent gold,
counting and `failures_for` logic, and the exact existing per-family thresholds.
It does not edit `suite.HELDOUT_SEEDS`, recalibrate, or widen the gate. The original
101/307 mode remains the fixed regression check with its separate `heldout.json`.
The supplementary mode writes `extra-heldout-freeze.json` before generating any
new test input, then baseline-first results to `extra-heldout.json` and routes to
`extra-adapter-calls.jsonl`. It records registration/source/binary/dispatch hashes.
If these inputs motivate another change, they have been consumed for development
and must not be described as unseen inputs for that later change.

## Coalesced smoke and padded-copy alignment failure

Run `20260916T162554805553Z-c2-smoke` completed all 15 coalesced-feed smoke cases.
This remains a smoke result until the full frozen gate completes. The first
`coalesced_pad17` attempt, run `20260916T162424950978Z-c2-smoke`, failed its first
GPU case with a misaligned-address error; it has no correctness/performance PASS.

The explicit 17-float row pitch advances by 68 bytes, so each row start is
guaranteed only four-byte alignment even though the array base is 128-byte
aligned. The generic `copy(reg,dst)` retains a 128-bit assumed-alignment policy,
and the failed binary still emits `STS.128`. The targeted repair changes only
the padded variant's two TMEM-register-to-shared writebacks (QK and PV) to
`copy(AutoVectorizingCopyWithAssumedAlignment<32>{}, reg, dst)`. The 16-float
original/coalesced branches keep the exact previous copy call. This narrows the
copy contract to the destination's actual guaranteed alignment; it changes no
attention arithmetic, scales, masks, split policy or numeric threshold. The
misaligned attempt remains archived, and the repaired variant requires a fresh
build and smoke before qualification.

## Separate release-assertion and TMEM-width ablations (not yet measured)

Two additional variants preserve the existing ids 0/1/2 and keep score stride16:

| ID / variant | Parent comparison | Single intended change |
|---|---|---|
| 3 / `coalesced_release` | id1 coalesced | Add `-DNDEBUG` to NVCC flags |
| 4 / `coalesced_wide` | id3 release | Replace TMEM `32dp32b1x` copy atom with `32dp32b16x` |

Neither variant includes pad17, TMA, different arithmetic, new split policies or
relaxed validation. `-DNDEBUG` studies compiled library assertion overhead; the
explicit host coordinate check, input `TORCH_CHECK`s and independent frozen
numeric gate remain required. A debug-versus-release speed difference is not
attributed to a better attention algorithm.

Before compiling id4, build.py reads the pinned CUTLASS `copy_sm100.hpp`, verifies
the exact `SM100_TMEM_LOAD_32dp32b16x` struct, 16 uint32 destination registers and
`tcgen05.ld.sync.aligned.32x32b.x16.b32` instruction, and saves the source hash and
primitive snippet. It refuses an absent or incompatible primitive. Widening
changes the TMEM copy partition and register lifetime, so its GPU correctness,
register allocation, spill behavior and complete-chain performance must be
remeasured. The binary's id and flags distinguish all paths in per-case audits.

## Coalesced full gate and performance outcome; original-feed wide control

The coalesced variant passed all 168 candidate records in the original full gate,
but the subsequent 32-row complete-chain benchmark regressed further: geometric
mean speedup 0.073612 versus the original baseline, compared with 0.093990 for
the first original-feed control. Globally contiguous logical traversal is not
selected as an improvement merely because its source accesses look favorable.
The global transaction/shared scatter/instruction tradeoff needs measured study.

Release and wide variants passed their 15-case smoke runs
`20260916T235940627966Z-c2-smoke` and `20260916T235940644602Z-c2-smoke`, respectively;
these do not yet constitute full acceptance or speed evidence.

New id5 `original_wide` isolates feed order relative to id4 `coalesced_wide`:
it restores the original Q/K/V traversal while retaining stride16, `-DNDEBUG`,
the audited x16 TMEM primitive, and all arithmetic/scale/split contracts. Existing
ids0–4 keep their previous semantics. This prevents the widened-read experiment
from being judged only through a traversal already observed to regress. No
performance or correctness result is claimed for id5 before its own execution.

## First-case Graph stall diagnostic

The repaired pad17 path passed its full 168-record gate and a one-chain eager
NCU run, but benchmark run `20260917T000250221149Z-c2-bench` did not finish its
first row before its 700-second external timeout. The remote Python process was CPU-active while
GPU utilization was zero. This does not establish either a GPU deadlock or a
slow 32-case benchmark; initialization, reference generation, JIT and graph
construction must be separated before attributing a cause.

`--mode smoke --graph-stress` invokes a separate diagnostic helper and preserves
the exact first benchmark case: `tp1-b1-bf16`, one 8192-token request, four KV
heads, seed101, default split selection, feature merge, PDL disabled. It ignores
the benchmark matrix selectors and rejects split/merge overrides. It brackets
imports, input/gold work, upstream launch capture, old/new preparation, eager
completion and all three paths' numerical gates. In increasing order, graphs
with 2/8/16/64 complete-chain calls are captured and replayed exactly twice per
path, with explicit synchronization and frozen-family checks. Five eager
warmups precede each capture; there are no hidden graph warmup replays.

Each stage begin/end is atomically written and fsynced to `progress.json`, with
an event history and the latest stage. Capture also records context-entry and
all-calls-enqueued markers, so graph finalization can be distinguished from
launch submission. A faulthandler watchdog writes all Python thread stacks to
`python-stack.log` every 45 seconds and is cancelled on normal/exception exit.
The external runner should impose a 240-second deadline
and preserve this file even if the child does not exit. Its CPU wall times are
diagnostic markers, not performance measurements. Passing these synchronized,
bounded replays does not validate the benchmark's longer unsynchronized replay
burst, and does not establish why the previous run stalled.

```sh
python nextgen/run.py --mode smoke --graph-stress --build-dir /tmp/nextgen-build --output /tmp/nextgen-artifacts
```

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
