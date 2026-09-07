# SM90 NVFP4 MegaMoE all-M — Weave implementation for H20

This implementation reproduces the H20-admitted all-M NVFP4 MegaMoE kernel in
Weave (the CAKE/Loom typed kernel IR) and evolves its schedule. It keeps the
production semantics behind `nvfp4_mega_moe` (EP8 routing, FP8 dispatch, packed
NVFP4 E2M1 weights with per-16 UE4M3 scales, FP4-to-FP8 decode before SM90
WGMMA, SwiGLU, L2, remote BF16 scatter, top-k combine, workspace state) and is
bit-exact against the parent on every validated row.

## Provenance

- Integration base: `all-m-opt@5ce3ab0` (this branch), plus the H20 admission
  commit that adds the 78-SM `(model, M)` bucket table to the all-M host
  selector.
- Parent kernels reproduced: the all-M small-M portfolio (dev-m dynamic + RS
  mode5, KF424 static-RS, dev-m dynamic + SS) as selected by the H20 table, and
  the `bigMopt` split L2 body for large M.
- Weave sources: `weave_h20/sm90_nvfp4_megamoe_h20_small_rs.py` (fused
  small-M provider, all ten material schedules) and
  `weave_h20/sm90_nvfp4_megamoe_h20_split_l2.py` (split L2), CAKE tree tag
  `h20-megamoe-weave-20260906`.
- Generated CUDA for every schedule: `weave_h20/generated/*.cu` with
  `weave_h20/generated/MANIFEST.json` (model, M, block_m, stages, launch
  geometry, sha256). These files are the exact NVRTC inputs of the receipts
  below.
- `weave_h20/validation/` holds the environment-gated phase mask used to run
  the production split L1 and the Weave L2 on one symmetric workspace during
  validation (not applied to the production host code).
- Hardware for every number here: one 8x NVIDIA H20-3e node (SM90, 78 SMs,
  NV18), eight ranks, Flash (`H=4096`, `I=2048`, `E=256`, `topk=6`) and Pro
  (`H=7168`, `I=3072`, `E=384`, `topk=6`).

## Runtime policy

The outer family decision is unchanged:

```text
rho = M * topk / local_experts

rho <= family_threshold (default 192) -> fused megakernel (dispatch + L1 + L2 + combine in one launch)
rho >  family_threshold               -> BN128 bigM split side (L1 kernel, then L2 kernel)
```

On the fused side one Weave kernel is built per material `(model, M)` bucket of
the H20 table. Every bucket launches 78 CTAs x 384 threads with 232448 bytes of
dynamic shared memory (1024 bytes are the Weave control page), no clusters,
PDL off. In the labels below, RS means a register A operand with a
shared-memory B operand for FP8 WGMMA; SS means both operands in shared memory.
SM90 has no FP4 tensor-core MMA: packed NVFP4 is decoded to E4M3 and the
tensor-core operation is FP8 WGMMA.

| Model | M   | H20 parent arm (baseline)   | block_m | operand form              | stages | active dispatch warps |
| ----- | --- | --------------------------- | ------- | ------------------------- | ------ | --------------------- |
| Flash | 8   | dev-m dynamic + RS mode5    | 8       | swap-AB RS, N8            | 6      | 1                     |
| Flash | 16  | dev-m dynamic + RS mode5    | 8       | swap-AB RS, N8            | 6      | 1                     |
| Flash | 32  | KF424 static-RS             | 8       | swap-AB RS, N8            | 6      | 2                     |
| Flash | 64  | dev-m dynamic + RS mode5    | 24      | swap-AB RS, N8/16/24      | 6      | 1                     |
| Flash | 128 | dev-m dynamic + SS          | 64      | SS, M64N128K32            | 3      | 2                     |
| Pro   | 8   | KF424 static-RS             | 8       | swap-AB RS, N8            | 6      | 1                     |
| Pro   | 16  | dev-m dynamic + RS mode5    | 8       | swap-AB RS, N8            | 6      | 1                     |
| Pro   | 32  | KF424 static-RS             | 8       | swap-AB RS, N8            | 6      | 2                     |
| Pro   | 64  | dev-m dynamic + RS mode5    | 24      | swap-AB RS, N8/16/24      | 6      | 1                     |
| Pro   | 128 | KF424 static-RS             | 24      | swap-AB RS, N8/16/24      | 6      | 1                     |

Every one of the ten buckets uses the dynamic task scheduler: loader warp 3
claims L1 and L2 tasks from the two global counters and publishes them through
the TaskInfo mailbox, exactly as the dev-m dynamic arms do. The parent's KF424
arm uses a static scheduler instead; the four KF424 buckets here keep KF424's
block_m, experts-per-wave and dispatch-warp settings but run them under the
dynamic scheduler, so no static-scheduler kernel exists in this implementation.
Nine buckets are swap-AB RS; only Flash M128 is SS. One more deviation is
deliberate and measured: every swap-AB bucket runs a 6-stage weight ring (the
parents use 3 or 4) because the rotated K loop keeps two stages resident.

On the split side the Weave L2 kernel replaces the production L2 behind the
unchanged production L1; the L1 body is not ported.

## Fused small-M kernel

### Warp roles

Twelve warps, `setmaxnreg` plan 48 / 64 / 208:

| Warps | Role                             | Regs | Duties                                                                                                                                                                                                                                 |
| ----- | -------------------------------- | ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0-1   | dispatch                         | 48   | source top-k token counting; ordered u64 per-expert offset claims; remote source-index scatter; workspace grid sync and stateful NVLink rendezvous; round-robin token / scale / top-k-weight pull into the local L1 pool (1-D TMA bulk copies, per-pool-block arrival counters); final workspace cleanup |
| 2     | activation loader                | 64   | consumes the task mailbox; polls the L1 arrival count (phase 1) or the L2 arrival mask (phase 2); issues the activation and per-128 activation-scale (SFA) TMA transactions per K block                                                  |
| 3     | weight loader and task producer  | 64   | caches the finalized expert receive counts; claims L1 / L2 tasks from the two global counters with the warm-up-wave rule; publishes the 32-byte TaskInfo record into a two-slot shared mailbox; issues the packed-weight TMA transaction per K block |
| 4-11  | math (two warpgroups)            | 208  | decode + WGMMA, L1 SwiGLU / FP8 epilogue, L2 BF16 remote scatter, top-k combine                                                                                                                                                         |

Named barriers: 0 (dispatch, 64 threads), 1 (dispatch + math handoff, 320
threads, unaligned), 2 (math, 256 threads). Mbarriers: `task_info_full` (one
producer arrival) and `task_info_empty` (eight math arrivals after the task's
first full stage is visible); `l1_stage_full` (two producer arrivals:
activation + SFA, weights) and `l1_stage_empty` (eight math arrivals); sixteen
combine load barriers (two slots per warp).

Implementation:

- `weave_h20/sm90_nvfp4_megamoe_h20_small_rs.py` (`_build_partial_dispatch_kernel`, roles `dispatch`, `loader`, `math`)

### Weight stream

One stage per BK = 128 holds 256 packed weight rows x 80 bytes (64 bytes of
E2M1 nibbles plus 16 bytes of UE4M3 scales for the eight 16-value groups), the
activation tile block_m x 128 FP8 bytes with the 128-byte TMA swizzle, and
block_m per-128 activation scales. Swap-AB buckets use six stages; the SS
bucket keeps three because its decoded FP8 tiles (32 KB per stage) do not fit
deeper.

### Decode and WGMMA layout (swap-AB buckets)

The product is transposed: the 256 weight rows are the WGMMA A operand (M =
four 64-row tiles; each warpgroup owns two, called halves) and the block_m
tokens are the B operand (N = 8, 16 or 24, chosen at run time from the task's
valid token count). Weights are decoded into registers (RS); activations are
read from shared memory by the tensor core.

Per lane, per half, per K32 slice:

1. The lane owns weight row `r` or `r+8` (lane bit 0) inside its warp's 16
   rows and packed word `word_sel` (lane bit 1) of the 16-byte slice chunk. It
   loads its two 32-bit packed words (K 0-7 and K 16-23 of that word) and, once
   per half, the row's two 32-bit scale words.
2. The two UE4M3 scale codes of the slice select two 8-byte rows of the 1 KB
   Mode2 lookup table in shared memory (128 codes x {low word, high word}),
   which maps the eight E2M1 magnitudes times the scale to FP8 bytes.
3. `nvfp4_mode2_lut_decode_word` (the production lop3 / prmt sequence) turns
   each packed word into eight FP8 bytes as two 32-bit halves. The lane keeps
   the half its fragment needs and ships the other to lane `^1` with one
   `shfl.xor` per word; the lane pair covers rows `r` and `r+8` of the same K
   columns, so every word is decoded exactly once.
4. The four registers are the RS A fragment of
   `wgmma.mma_async.m64nNk32.f32.e4m3.e4m3`; B is the activation stage tile
   `(0, k*32)` of shape (N, 32).

Four slices form one commit group per half. When the group retires, the
per-K-block activation scale is applied in registers
(`final += sfa[token] * accum`), which is the production accumulation order.

Implementation:

- `_emit_rs_half_decode`, `_emit_rs_half_mma`, `_emit_rs_half_accumulate`

### Rotated K loop

```text
wait full(k0); decode half0(k0)
for k:  mma half0(k)                                   # group G0
        decode half1(k); mma half1(k)                  # group G1, decode overlaps G0
        wait_group 1; accumulate half0(k)
        advance; if k+1 < n: wait full(k+1); decode half0(k+1)   # overlaps G1
        wait_group 0; accumulate half1(k)
        arrive empty(k)                                # two stages resident
```

Relative to the parent's serial stage (decode, WGMMA, drain, accumulate) this
keeps the tensor pipe busy during every decode and removes two per-stage
synchronizations: the 128-thread warpgroup named barrier before the empty
arrive (each warp's own `wgmma.wait_group 0` already orders its stage reads)
and the `fence.proxy.async` after the full-stage wait (the acquiring mbarrier
wait already orders the TMA-written stage against the generic loads). Both are
bit-exact.

Implementation:

- `_emit_rs_task_rotated`

### Non-swap SS bucket (Flash M128)

All eight math warps expand one packed N256 x K128 tile into a 128-byte
swizzled FP8 shared tile (one packed row per thread, same LUT decode, 16-byte
swizzled stores); each warpgroup then issues four `m64n128k32` shared/shared
WGMMA over its 128-column half. The stage is split into two K halves so the
second half's decode overlaps the first half's WGMMA. Here the 64 tokens are
the A operand and the weights the B operand.

Implementation:

- `_emit_nonswap_stage`

### Epilogues, scatter, combine, cleanup

- L1 epilogue (phase 1): per-expert global scale, clamp(+-10) fast-math SwiGLU
  (`gate * rcp(1 + exp(-gate)) * up`), per-route top-k weight, quad and
  cross-warp amax per token, UE8M0 power-of-two scale stored as the L2
  activation scale, E4M3 staging (block_m x 128 columns per task), one TMA
  store into the L2 activation pool, release-OR of the pool block's L2 arrival
  mask bit.
- L2 epilogue (phase 2): per-expert L2 scale, BF16 staging tile
  (block_m x 256, swizzled), then 16 lanes per token row write 16 bytes each
  into the destination rank's combine slot selected by the transported token
  metadata (rank, token, top-k slot) through the symmetric peer pointer table.
- Combine: after the epilogue grid sync and NVLink rendezvous, each math warp
  walks a token stride, reads the eight top-k slot markers, alternates two raw
  1-D TMA load slots over the valid symmetric contributions, accumulates in
  f32, packs BF16 into a third slot and issues one 1-D TMA store (hidden 4096:
  one chunk; hidden 7168: two chunks).
- Cleanup: the dispatch warps reset the workspace (CTA 0: global send counts
  and task counters; the others: their expert's receive counts and pool-block
  arrival state), then a grid sync and the final rendezvous.

Implementation:

- `_emit_l1_swap_epilogue`, `_emit_l1_nonswap_epilogue`, `_emit_l2_swap_remote_scatter`,
  `_emit_l2_nonswap_remote_scatter`, combine and cleanup sections of `_build_partial_dispatch_kernel`

### Common 128-byte workspace ABI

The parent's 128-byte workspace prefix is reused unchanged: bytes 0-27 grid /
NVLink state, 28-31 the L1 task counter, 32-35 the L2 task counter, padding to
128, then the expert counters. The Weave kernel and the parent therefore share
one symmetric buffer layout, which is what allows the paired same-process
comparisons and the split hybrid validation.

### Register policy

The Weave kernels are compiled with NVRTC for `sm_90a` with `--use_fast_math`
and the default register-usage level; the role budgets are set with
`setmaxnreg` (48 / 64 / 208 on the fused kernel, 112 / 192 on the split L2).
The fused kernels report 157 registers and no local-memory spills.

## Small-M performance versus the H20 parent

Physical 8x H20-3e, profile-off, same-process A/B/B/A with the parent arm of
each bucket, 5 blocks x 32 calls per observation, CUDA-event rank-MAX timing.
The measured/SOL column uses the weight-stream floor of the bucket (Flash
~105 us, Pro ~412 us at 4.8 TB/s).

| Model | M   | H20 parent arm  | parent p50 | Weave p50  | Paired speedup | Weave / SOL |
| ----- | --- | --------------- | ---------- | ---------- | -------------- | ----------- |
| Flash | 8   | dynamic+RS      | 252.9 us   | 235.0 us   | +7.6%          | 2.24x       |
| Flash | 16  | dynamic+RS      | 257.7 us   | 238.2 us   | +8.2%          | 2.27x       |
| Flash | 32  | KF424 static-RS | 282.6 us   | 239.7 us   | +17.9%         | 2.28x       |
| Flash | 64  | dynamic+RS      | 327.6 us   | 275.8 us   | +18.8%         | 2.63x       |
| Flash | 128 | dev-m dynamic SS| 429.2 us   | 428.8 us   | +0.1%          | 3.28x       |
| Pro   | 8   | KF424 static-RS | 853.4 us   | 752.0 us   | +13.5%         | 1.83x       |
| Pro   | 16  | dynamic+RS      | 952.2 us   | 840.6 us   | +13.3%         | 2.04x       |
| Pro   | 32  | KF424 static-RS | 941.5 us   | 852.1 us   | +10.5%         | 2.07x       |
| Pro   | 64  | dynamic+RS      | 1062.9 us  | 907.7 us   | +17.1%         | 2.20x       |
| Pro   | 128 | KF424 static-RS | 1147.4 us  | 1023.9 us  | +12.1%         | 2.44x       |

The ten-point, equally weighted geometric-mean speedup is **11.8%** (Weave /
parent 0.895). Every paired ratio has a tight per-bucket bootstrap interval;
the arm baselines themselves move by a few percent between runs, so the paired
ratio, not the absolute microseconds, is the receipt quantity.

### Correctness

The eight-rank gang evaluator ran the production `run_e2e` ABI on all ranks
against the frozen reference and the parent arm: objective 1.0, ten buckets,
bit-exact BF16 output on two independent workspace generations, every gate true
(artifact consensus, physical eight-rank launch, no hidden native fallback, no
Blackwell instructions, source bound). No non-equivalent transformation is
used: same input format, same FLOPs, same per-token accumulation order.

## Large-M split path

`weave_h20/sm90_nvfp4_megamoe_h20_split_l2.py` is an exact port of the
`bigMopt` split L2 body: 384 threads (warp 0 activation + SFA TMA, warp 1
packed-weight TMA, warps 2-3 a 64-thread in-place Mode2 decode team expanding
the 128 x 80-byte packed rows into the 128-byte-swizzled FP8 operand of the same
stage buffer behind a `dequant` mbarrier, warps 4-11 two M64N128 SS
warpgroups), register plan 112 / 192, six stages, a static grid-stride
linear-2 scheduler from per-lane cached expert counts, a warp-private 8 x 136
BF16 staging tile with 16-lane 16-byte remote scatter, NVLink rendezvous, the
top-k combine and the epilogue-owned workspace cleanup.

It was validated behind the production L1 on the same symmetric workspace
(`weave_h20/validation/` phase mask): bit-exact against the production L2 on
all eight ranks for the six large-M points, and the production L1 + L2 pair
run afterwards on the same workspace is bit-exact too (cleanup correct).
Paired timing of the L2 launch alone (events around the L2, three A/B/B/A
blocks x 8 iterations):

| Model | M    | production L2 | Weave L2   | L2 speedup | pair (L1 + L2) | pair / SOL |
| ----- | ---- | ------------- | ---------- | ---------- | -------------- | ---------- |
| Flash | 2048 | 924.4 us      | 915.3 us   | +1.0%      | 2517.2 us      | 1.20x      |
| Flash | 4096 | 1795.0 us     | 1769.8 us  | +1.4%      | 4919.5 us      | 1.18x      |
| Flash | 8192 | 3564.2 us     | 3518.2 us  | +1.3%      | 9784.7 us      | 1.17x      |
| Pro   | 2048 | 2211.3 us     | 2200.8 us  | +0.5%      | 6232.2 us      | 1.14x      |
| Pro   | 4096 | 4424.6 us     | 4383.7 us  | +0.9%      | 12360.7 us     | 1.13x      |
| Pro   | 8192 | 8799.4 us     | 8701.1 us  | +1.1%      | 24580.4 us     | 1.12x      |

The pair is within +-0.3% of the public-entry baseline on every point (the L2
is about 36% of the pair); the six large-M points run at 83-89% of the FP8
tensor peak, so the split side is inside its speed-of-light headroom and a
Weave L1 port can at best hold parity.

## Speed-of-light analysis and bottlenecks

Per-resource floors for Flash M8 (235 us), with the utilisation of the final
kernel from device-wide CUPTI PM sampling on the slowest rank:

| Resource         | Load per call                                              | H20-3e peak                        | Floor         | Utilisation                                                                     |
| ---------------- | ---------------------------------------------------------- | ---------------------------------- | ------------- | ------------------------------------------------------------------------------- |
| HBM              | 503 MB packed weights (+ small activation / scatter traffic) | 4.8 TB/s spec, 3.87 TB/s measured | 105 / 130 us  | ~45-50%                                                                         |
| SMEM / LSU pipe  | LUT gathers, packed-word loads, TMA writes, shuffles       | 128 B/clk/SM                       | ~108 us       | 46% of wavefronts; LSU loads 9.5%, stores 0%, bank conflicts ~21% of the loads |
| Issue slots      | ~342 SASS per math warp per K block                        | 1 inst/clk/scheduler               | ~120 us       | 51%                                                                             |
| FP8 tensor pipe  | 2.4 GFLOP                                                  | 296 TFLOPS                         | ~12 us        | 4.8%                                                                            |
| NVLink           | ~1 MB                                                      | 396 GB/s                           | ~1 us         | negligible                                                                      |

No unit is saturated. The K loop is about 92% of the kernel at ~0.64 us
(~1270 cycles) per K block; its critical chain is `LDS packed word -> LUT
gather -> lop3/prmt decode -> shfl exchange -> select -> wgmma`, about 90
cycles per K32 slice, with two math warps per scheduler to interleave. The
remaining ~35 us are protocol: the two grid syncs and the NVLink rendezvous
before the combine (12-15 us), task-boundary readiness waits (11 us) and the
epilogues (8 us). Pro buckets carry an additional 15-20% cross-rank routing
tail (the slowest rank bounds the call) that the parent shares.

The binding constraint is therefore the CUDA-core decode of NVFP4 weights
itself: every weight byte costs LUT gathers, lop3 / prmt and shuffles on an SM
whose CUDA-core throughput is small relative to its HBM bandwidth, and per-stage
synchronization is the only other lever that moved the kernel. The practical
ceiling of this design family is about 1.16-1.20x over the H20 parent on small
M (Flash M8 ~215-225 us, Pro M8 ~700-720 us). Closing the remaining ~1.8x to
the HBM floor needs a weight format with fewer CUDA-core operations per byte
(or pre-decoded weights), which is outside the same-input contract.

## Validation status

Done: ten-bucket eight-rank evaluator receipt (objective 1.0, bit-exact, all
gates true) on the committed default schedule; three-arm paired A/B/B/A
receipts (dynamic+RS, KF424, dev-m SS) with per-bucket bootstrap intervals; six
large-M points bit-exact behind the production L1 with the cleanup check;
headless NVRTC compile audit of every schedule (register plan, barrier
protocol, instruction sites); timestamp attribution and CUPTI PM sampling of
the default kernel.

Not done: Weave port of the split L1 body (the split side keeps the production
L1); H200 (132-SM) schedules, which this branch does not build.

## Usage

The Weave sources need the CAKE/Loom compiler (`loom` package, NVRTC, CUDA 13
runtime) to regenerate and launch:

```python
from loom.examples.weave import sm90_nvfp4_megamoe_h20_small_rs as small_rs
small_rs.compile_partial_weave()          # headless: generate + NVRTC + audit, all ten schedules
small_rs.run_e2e(sym_ptrs, rank, state_generation, l1_packed, l2_packed,
                 l1_global_scales, l2_global_scales, route_profile, m, h, y)   # production ABI, one rank
```

```python
from loom.examples.weave import sm90_nvfp4_megamoe_h20_split_l2 as l2
schedule = l2.select_split_l2_schedule(hidden, intermediate, num_experts)
launch = l2.prepare_split_l2_launch(schedule=schedule, local_base_ptr=..., peer_ptrs=..., rank=...,
                                    num_tokens=..., l2_packed_weights=..., l2_global_scales=..., y=...)
launch.prepared_launch.launch()           # after the production L1 (validation phase mask: l1 only)
```

Readers without the compiler can inspect `weave_h20/generated/*.cu`: each
file is the complete CUDA translation unit of one schedule as compiled for the
receipts (kernel symbols `kernel_small_rs_dispatch_fragment` and
`kernel_split_l2_fragment`, launch geometry in `MANIFEST.json`). Weights are
prepared once with `transform_nvfp4_weights_for_mega_moe_sm90` exactly as for
the parent; the Weave kernels consume the same symmetric workspace and the same
packed-weight and scale layouts.
