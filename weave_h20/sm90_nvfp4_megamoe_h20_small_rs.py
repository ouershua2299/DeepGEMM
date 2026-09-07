"""Pure H20 small-M RS MegaMoE Weave provider.

The production ``run_e2e`` ABI launches the source-exact Weave kernel.  The
Weave implementation below captures the source wrapper's ten material
Flash/Pro schedules and the fused
kernel's 384-thread role entry: dispatch warps 0-1 deallocate 48 registers,
loader warps 2-3 deallocate 64 registers for the effective current-provider
schedule, and math warps 4-11 allocate 208 registers.
It also captures the post-initialization dispatch phase: source top-k token
counting, ordered u64 per-expert offset claims, remote source-index scatter,
the exact contiguous workspace offsets, workspace grid synchronization,
remote status publication, the stateful NVLink rendezvous, round-robin source
rank selection, and the token/SF/top-k-weight pull into the local L1 pool.
It additionally retains the parent-shaped L1 mathematical slice for the
source's runtime-selected M64N8/N16/N24 register-source WGMMA arms: packed
Mode2 word-pair decode with independent per-16 UE4M3 scales, four live K32
fragments, every source operand fence, two 64-row weight halves, activation
scale accumulation, and static completion barriers for both math warpgroups.
The parent-shaped slice now also retains the unaligned 320-thread
dispatch/math handoff and loader-warp device-acquire wait for the L1 pool
block's u32 arrival count to equal ``valid_m``.
An additional parent-shaped fragment retains the source L1 producer/consumer
pipeline itself: loader warp 2 issues descriptor-bound activation and SFA TMA
transactions, loader warp 3 issues the packed-weight TMA transaction, the two
warps jointly complete one full barrier per stage, and one elected thread from
each of the eight math warps returns the stage through the empty barrier.  Both
material three- and four-stage schedules are represented with runtime stage
advance and reuse.
The swap-AB BLOCK_M=8/16/24 arms are now also retained as one integrated
parent-shaped TMA-to-RS consumer: the math warps wait on those exact full
barriers, consume the staged activation/SFA/packed-weight views through Mode2
register-source WGMMA, select static N8/N16/N24 instructions from runtime
``valid_m`` using the source bucket branches, synchronize each math warpgroup,
and only then return the empty stage to both loader warps.
For every material current-provider schedule, the dispatch fragment now also
retains the source weight-loader warp's scheduler acquisition state: finalized
u64 expert-count caching, total pool-block and warm-up-wave derivation, L1/L2
global task claims, native ``continue`` re-entry, the u32 L1-issued readiness
poll, warp-prefix task-to-expert ownership, and construction of the exact
32-byte task payload plus its derived K-block count.  This follows the dynamic
wrapper's effective ``kInterleaved = kCurrent`` schedule instead of mistaking
its raw ``CURRENT_INTERLEAVED`` request bit for the launched physical plan.
Consequently the retained current-provider loader plan is 64 registers on all
ten rows.  The dispatch fragment now fuses that claim state with the complete
two-stage TaskInfo transport: loader warp 3 waits before each claim, publishes
all eight words with a CTA-scoped fence, advances its independent cursor, and
publishes the final phase-zero invalid task exactly once.  Loader warp 2 and
all eight math warps independently consume until that terminator; only valid
tasks receive the source's eight math-warp empty-slot arrivals.
For all eight swap-AB rows, phase-one tasks in that real transported stream now
drive the persistent L1 body directly: loader warp 2 consumes the payload's
pool block and K extent for the acquire plus activation/SFA TMA coordinates,
loader warp 3 uses its expert/N-block/shape fields for packed-weight TMA, and
the math warps use valid-M for the runtime N8/N16/N24 RS choice.  The L1
full/empty cursor persists across task boundaries, and each math warp releases
the TaskInfo slot only after its first full L1 stage is visible.
Those same eight rows now continue through the complete source swap-AB L1
epilogue: per-expert scale and per-route top-k weighting, clamp-10 fast-math
SwiGLU, quad and eight-warp amax reduction, exact UE8M0 power-of-two scale
construction, E4M3 row-major staging, one 128-column TMA publication, and the
typed device-release OR of the corresponding L2 arrival bit.
Transported phase-2 tasks on those rows now wait for the complete L1 arrival
mask, bind the L2 activation, activation-scale, and packed-weight descriptors
to the same persistent pipeline, and execute the same four-slice Mode2 RS
WGMMA accumulation.  Their source epilogue scales by the per-expert L2 factor,
casts the register-source layout into a row-major BLOCK_M-by-256 BF16 shared
tile, synchronizes each math warpgroup, reloads one 16-byte vector per lane,
and scatters it to the symmetric peer selected by the three-word token source
metadata.  A final aligned 256-thread barrier protects shared-tile reuse.
After the final remote scatter, every retained row now also enters the source's
math-scoped pre-combine boundary: two epilogue grid synchronizations bracket
the typed workspace-stateful NVLink rendezvous, followed by the second
320-thread dispatch/math handoff.  Dispatch performs the production wrapper's
exact effective cleanup specialization after that handoff: CTA 0 clears every
global send count and both interleaved task counters, while CTAs 1-77 clear
their assigned local expert's receive counts plus the live L1-arrival and
L2-mask pool-block ranges.  The optional cumulative-statistics branch is
specialized away because the pinned wrapper passes ``nullptr``.  A dispatch
grid synchronization and the third typed rendezvous publish cleanup completion.
The math warps on every row continue past the pre-cleanup handoff through the source
combine schedule.  Each warp owns the source token stream, reads the eight
read-only top-k slot markers, alternates two raw 1-D TMA load stages over every
valid symmetric contribution, accumulates BF16 pairs in f32, packs a third
shared-memory slot, and issues the final BF16 1-D TMA output store.  Flash
hidden=4096 retains one full-hidden chunk; Pro hidden=7168 retains the source
two-chunk schedule so the register and shared-memory bounds stay exact.
Every swap-AB integrated L1 arm separately retains the same TaskInfo layout
and source release boundary around its one-task TMA-to-RS body.
The payload's pool block, expert, valid-M, N-block, shape-N, and shape-K fields
directly select the activation/SFA and packed-weight TMA coordinates and the
runtime RS shape, and the math warps return the mailbox slot only after the
first L1 full-stage wait makes the payload safe to recycle.
The two non-swap BLOCK_M=64 rows now consume transported phase-one tasks as
well.  Their loader warps stage source activation/SFA and packed weights, all
eight math warps expand each packed N256-by-K128 Mode2 tile into the source
FP8 N256-by-K128 shared tile, and the two math warpgroups issue four
M64N128K32 shared-source WGMMA groups.  The non-swap epilogue preserves the
source gate/up lane ownership, per-expert and per-route scales, clamp-10
SwiGLU, cross-warpgroup amax, UE8M0 scale construction, E4M3 staging, output
TMA store, and release-OR notification.  Its TaskInfo slot is recycled only
after the first full L1 stage, matching the swap-AB stream boundary.
Those two rows now also consume their transported phase-two descriptors: the
loader waits for the complete L1 arrival mask and stages the L2 activation,
scale, and packed weight, while both math warpgroups reuse the same four-slice
M64N128K32 shared-source body.  The source direct-scatter epilogue applies the
per-expert L2 scale, packs adjacent BF16 pairs from each row's register-owned
fragment, and stores 32-bit words directly into the symmetric rank/token/top-k
destination selected by the three-word metadata.  After the aligned epilogue
barrier, both rows join the retained cleanup/rendezvous and chunked top-k
combine tail.
All seven live tensor-map annotations now also retain the source descriptor
binder's 256-byte L2-promotion policy.  This metadata does not change the
generated CUDA instruction stream, but it is part of the runtime tensor-map
identity and must be present before the production launcher can reproduce the
native descriptor bindings.
The production family and both register-source helper families now declare the
exact 1024-byte E2M1/UE4M3 lookup table as one typed module-scope device
constant, with the authority payload hash pinned in this file.  The source's
first 64 threads each issue one typed 16-byte constant-to-shared copy; the LUT
is no longer an artificial kernel pointer argument.
The complete dispatch fragment no longer carries the compile-only sentinel
pointer used while earlier role fragments were being retained.  Every dispatch,
loader, and math role now stays live through its real source side effects, so
the sentinel stores would be an extra production ABI argument and observable
global writes absent from the pinned source.  Standalone helper fragments keep
their isolated compile sinks; they are not members of the production launch
family.
``compile_partial_weave`` generates and NVRTC-compiles every material dispatch
schedule, all three standalone RS shapes, every material L1 TMA pipeline arm,
and all four effective current-provider integrated swap-AB arms; it also
checks the constant-table payload/declaration, source-shaped uint4 copy, and
absence of a LUT pointer argument.  No fragment is launched.

The production host preparation path consumes the frozen ``run_e2e`` ABI,
validates its rank/model/shape identity, wraps the local raw symmetric base in
one ``ExternalDeviceAllocation``, and binds the production IR's seven tensor
maps through the canonical runtime: five exact symmetric-allocation slices use
``create_tma_from_external_slice`` and the two packed-weight views use
``create_tma_from_spec``.  The copied CUDA peer-pointer table binds both the
symmetric-memory peers and the otherwise-unused process-group flags parameter;
the compile audit proves that the generated kernel never reads the latter.

The production host path caches the generated SM90 CUDA and cubin per
material source schedule, loads the selected symbol on the current rank's
device, prepares the pinned 78-CTA by 384-thread launch with 232448 bytes of
dynamic shared memory and PDL disabled on the caller's current stream, and can
submit that prepared launch.  Its receipt contains the
worktree-bound provider hash, generated-source and cubin hashes, exact kernel
symbol, source family, and forbidden-instruction result.  The compile audit
exercises this cache and receipt construction for all ten rows without
launching a fragment.  Production ``run_e2e`` submits the same prepared launch,
and ``artifact_receipt`` fails closed until that submission has installed the
worktree-bound receipt.  External physical-eight correctness remains the
evaluator-owned gate.

Minimum architecture: SM90a.  The consumer preserves the pinned
small-RS fused schedule, EP8 routing/state protocol, Mode2 group-16 scaling,
register-source WGMMA, SwiGLU, L2, remote scatter, and top-k combine.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import NamedTuple

import loom
from loom.codegen.weave_ir import LaneIndexSource
from loom.weave.types import lm as LM


SOURCE_FAMILY = "small_rs"
PROVIDER_KIND = "weave"
# Source ``mbarrier.try_wait.parity`` suspendTimeHint (DeepGEMM ``ptx/tma.cuh``:
# ``"r"(0x989680)``). Restores the pre-rebase wait form; paired timing on H20
# showed it neutral (<0.6% on every row) but it is the source form.
MBARRIER_WAIT_SUSPEND_TICKS = 0x989680
# Schedule: overlap the second weight half's Mode2 decode with the first half's
# asynchronous WGMMA group inside one BK128 stage (wgmma.wait_group 1 then 0).
# Both halves accumulate into disjoint ``final_accum`` slots in the source
# order, so the result is bit-identical to the source-exact serial schedule.
# H20-3e paired timing 2026-09-05 (job 4151138): -5..-6% on every swap-AB row,
# geomean vs the dynamic+RS arm 0.976 -> 0.925, vs the 424 arm 1.029 -> 0.966.
# ``WEAVE_RS_HALF_OVERLAP=0`` restores the serial source schedule for A/B runs.
RS_STAGE_HALF_OVERLAP = os.environ.get("WEAVE_RS_HALF_OVERLAP", "1") == "1"
# Schedule: rotate the swap-AB K loop so half 0 of K-block k+1 is decoded while
# half 1 of K-block k is still in flight (two resident stages, so every swap-AB
# bucket runs a 6-stage ring). Bit-identical output. H20-3e paired timing
# 2026-09-05 (job 4151138): geomean vs the dynamic+RS arm 0.925 -> 0.897, vs the
# 424 arm 0.966 -> 0.942; with only 3 stages the loader starves and the same
# rotation loses 5-10%. ``WEAVE_RS_KBLOCK_ROTATE=0`` restores the per-K-block
# half-overlap schedule for A/B runs.
RS_KBLOCK_ROTATE = os.environ.get("WEAVE_RS_KBLOCK_ROTATE", "1") == "1"
# Schedule: in the non-swap SS stage, decode K 0-63 and issue its two WGMMA
# slices before decoding K 64-127, so the second decode half overlaps the first
# WGMMA group. Same accumulator and slice order -> bit-identical. H20 paired
# timing (Flash M128 vs the dev-m SS arm): 1.008 -> 1.000.
# ``WEAVE_SS_KHALF_OVERLAP=0`` restores the whole-stage source order.
SS_KHALF_OVERLAP = os.environ.get("WEAVE_SS_KHALF_OVERLAP", "1") == "1"
SS_K_HALVES = 2 if SS_KHALF_OVERLAP else 1
SS_QUADS_PER_HALF = 4 // SS_K_HALVES
REAL_WEAVE_CONSUMER = True

THREADS = 384
NUM_CTAS = 78
CLUSTER_DIMS = (1, 1, 1)
USE_PDL = False
SMEM_BYTES = 232448
LOOM_CONTROL_BYTES = 1024
TOP_K = 6
NUM_RANKS = 8
MAX_TOKENS_PER_RANK = 8448
DISPATCH_BARRIER_ID = 0
DISPATCH_THREADS = 64
DISPATCH_WITH_MATH_BARRIER_ID = 1
DISPATCH_WITH_MATH_THREADS = 320
EPILOGUE_FULL_BARRIER_ID = 2
EPILOGUE_THREADS = 256
EPILOGUE_WARPS = 8
DISPATCH_GRID_SYNC_INDEX = 0
EPILOGUE_GRID_SYNC_INDEX = 1
TOKENS_PER_WARP = 32 // TOP_K
WORKSPACE_SIGNAL_BYTES = 128
RS_BLOCK_K = 128
RS_WEIGHT_ROWS = 256
RS_WEIGHT_ROW_BYTES = 80
RS_ACT_ROWS = 24
RS_ACT_ROW_BYTES = 128
RS_LUT_WORDS = 256
RS_LUT_PAYLOAD_SHA256 = "9b0d0e0d9a2d42a807bac55590966e2f2c264e182979ed94b22f568e1c1e5afb"
# Exact little-endian u32 payload of the source's 128-row uint2
# kE2M1AndUe4m3ToFp8Lut.  Its 1024-byte identity is
# 9b0d0e0d9a2d42a807bac55590966e2f2c264e182979ed94b22f568e1c1e5afb.
RS_LUT_WORD_VALUES = (
    0x00000000, 0x00000000, 0x02010000, 0x06040302, 0x03020100, 0x0C080604, 0x04030200, 0x110C0906,
    0x06040200, 0x14100C08, 0x08050200, 0x17120F0A, 0x09060300, 0x1914110C, 0x0A070400, 0x1A16120E,
    0x0C080400, 0x1C181410, 0x0E090400, 0x1E191611, 0x0F0A0500, 0x1F1A1712, 0x100B0600, 0x201B1813,
    0x110C0600, 0x211C1914, 0x120D0600, 0x221D1A15, 0x120E0700, 0x221E1A16, 0x130F0800, 0x231F1B17,
    0x14100800, 0x24201C18, 0x16110900, 0x26211E19, 0x17120A00, 0x27221F1A, 0x18130B00, 0x2823201B,
    0x19140C00, 0x2924211C, 0x1A150D00, 0x2A25221D, 0x1A160E00, 0x2A26221E, 0x1B170F00, 0x2B27231F,
    0x1C181000, 0x2C282420, 0x1E191100, 0x2E292621, 0x1F1A1200, 0x2F2A2722, 0x201B1300, 0x302B2823,
    0x211C1400, 0x312C2924, 0x221D1500, 0x322D2A25, 0x221E1600, 0x322E2A26, 0x231F1700, 0x332F2B27,
    0x24201800, 0x34302C28, 0x26211900, 0x36312E29, 0x27221A00, 0x37322F2A, 0x28231B00, 0x3833302B,
    0x29241C00, 0x3934312C, 0x2A251D00, 0x3A35322D, 0x2A261E00, 0x3A36322E, 0x2B271F00, 0x3B37332F,
    0x2C282000, 0x3C383430, 0x2E292100, 0x3E393631, 0x2F2A2200, 0x3F3A3732, 0x302B2300, 0x403B3833,
    0x312C2400, 0x413C3934, 0x322D2500, 0x423D3A35, 0x322E2600, 0x423E3A36, 0x332F2700, 0x433F3B37,
    0x34302800, 0x44403C38, 0x36312900, 0x46413E39, 0x37322A00, 0x47423F3A, 0x38332B00, 0x4843403B,
    0x39342C00, 0x4944413C, 0x3A352D00, 0x4A45423D, 0x3A362E00, 0x4A46423E, 0x3B372F00, 0x4B47433F,
    0x3C383000, 0x4C484440, 0x3E393100, 0x4E494641, 0x3F3A3200, 0x4F4A4742, 0x403B3300, 0x504B4843,
    0x413C3400, 0x514C4944, 0x423D3500, 0x524D4A45, 0x423E3600, 0x524E4A46, 0x433F3700, 0x534F4B47,
    0x44403800, 0x54504C48, 0x46413900, 0x56514E49, 0x47423A00, 0x57524F4A, 0x48433B00, 0x5853504B,
    0x49443C00, 0x5954514C, 0x4A453D00, 0x5A55524D, 0x4A463E00, 0x5A56524E, 0x4B473F00, 0x5B57534F,
    0x4C484000, 0x5C585450, 0x4E494100, 0x5E595651, 0x4F4A4200, 0x5F5A5752, 0x504B4300, 0x605B5853,
    0x514C4400, 0x615C5954, 0x524D4500, 0x625D5A55, 0x524E4600, 0x625E5A56, 0x534F4700, 0x635F5B57,
    0x54504800, 0x64605C58, 0x56514900, 0x66615E59, 0x57524A00, 0x67625F5A, 0x58534B00, 0x6863605B,
    0x59544C00, 0x6964615C, 0x5A554D00, 0x6A65625D, 0x5A564E00, 0x6A66625E, 0x5B574F00, 0x6B67635F,
    0x5C585000, 0x6C686460, 0x5E595100, 0x6E696661, 0x5F5A5200, 0x6F6A6762, 0x605B5300, 0x706B6863,
    0x615C5400, 0x716C6964, 0x625D5500, 0x726D6A65, 0x625E5600, 0x726E6A66, 0x635F5700, 0x736F6B67,
    0x64605800, 0x74706C68, 0x66615900, 0x76716E69, 0x67625A00, 0x77726F6A, 0x68635B00, 0x7873706B,
    0x69645C00, 0x7974716C, 0x6A655D00, 0x7A75726D, 0x6A665E00, 0x7A76726E, 0x6B675F00, 0x7B77736F,
    0x6C686000, 0x7C787470, 0x6E696100, 0x7E797671, 0x6F6A6200, 0x7F7A7772, 0x706B6300, 0x7F7B7873,
    0x716C6400, 0x7F7C7974, 0x726D6500, 0x7F7D7A75, 0x726E6600, 0x7F7E7A76, 0x736F6700, 0x7F7F7B77,
    0x74706800, 0x7F7F7C78, 0x76716900, 0x7F7F7E79, 0x77726A00, 0x7F7F7F7A, 0x78736B00, 0x7F7F7F7B,
    0x79746C00, 0x7F7F7F7C, 0x7A756D00, 0x7F7F7F7D, 0x7A766E00, 0x7F7F7F7E, 0x7B776F00, 0x7F7F7F7F,
    0x7C787000, 0x7F7F7F7F, 0x7E797100, 0x7F7F7F7F, 0x7F7A7200, 0x7F7F7F7F, 0x7F7B7300, 0x7F7F7F7F,
    0x7F7C7400, 0x7F7F7F7F, 0x7F7D7500, 0x7F7F7F7F, 0x7F7E7600, 0x7F7F7F7F, 0x7F7E7600, 0x7F7F7F7F,
)
RS_LUT_COMPACT_WORD_VALUES = tuple(RS_LUT_WORD_VALUES[2 * i] for i in range(128)) + tuple(
    RS_LUT_WORD_VALUES[2 * j + 1] for j in range(8)
)
RS_WEIGHT_BYTES = RS_WEIGHT_ROWS * RS_WEIGHT_ROW_BYTES
RS_ACT_BYTES = RS_ACT_ROWS * RS_ACT_ROW_BYTES
RS_LUT_BYTES = RS_LUT_WORDS * 4
RS_SFA_BYTES = RS_ACT_ROWS * 4
RS_TMA_PACKED_B_BYTES = RS_WEIGHT_ROWS * RS_WEIGHT_ROW_BYTES
SS_DECODED_B_STAGE_BYTES = RS_WEIGHT_ROWS * RS_BLOCK_K
TASK_INFO_STAGES = 2
TASK_INFO_FIELDS = 8
TASK_INFO_BYTES = TASK_INFO_FIELDS * 4
RS_FRAGMENT_BYTES = (
    RS_WEIGHT_BYTES + RS_ACT_BYTES + RS_LUT_BYTES + RS_SFA_BYTES + 1023
) // 1024 * 1024


# Decode team for the swap-AB rows: the two dispatch warps (idle after the
# token/SF pulls) expand K-slice 3 (K 96..127) of every packed weight stage into
# a 32-byte-swizzled FP8 tile that the math warpgroups consume through one
# shared/shared WGMMA slice; the math warps decode only K-slices 0-2 into
# registers. Same per-token accumulation order (bit-exact with the RS path).
RS_TEAM_SLICE3 = os.environ.get("WEAVE_RS_TEAM_SLICE3", "0") == "1"
# RS decode packed-word fetch: one 16-byte load per K32 slice (lane pairs that
# share a row broadcast the same chunk; the two needed words are selected in
# registers) instead of two 32-bit loads at the 2-way-conflicted 80-byte row
# stride. Same words, bit-exact.
RS_PACKED_VEC4 = os.environ.get("WEAVE_RS_PACKED_VEC4", "0") == "1"
# Per-K-block warpgroup named barrier (bar 3/4, 128 threads) before the empty
# arrive in the rotated body. Each warp's own `wgmma.wait_group 0` already
# orders its stage reads, so the barrier is not needed for correctness; kept
# as a knob for A/B timing.
RS_STAGE_WG_SYNC = os.environ.get("WEAVE_RS_STAGE_WG_SYNC", "0") == "1"
# Generic/async proxy fence after each full-stage wait in the rotated body. The
# acquiring mbarrier wait already orders the TMA-written stage against the
# following generic loads, so the fence is a per-stage cost only.
RS_FULL_FENCE = os.environ.get("WEAVE_RS_FULL_FENCE", "0") == "1"
# Stage-level rotation: decode both halves of K-block k+1 (eight independent
# K32 slices) while K-block k's two WGMMA groups are in flight. Needs four
# fragment sets (64 registers), so it is limited to WGMMA N <= MAX_N; wider
# shapes keep the half-level rotation.
RS_STAGE_ROTATE = os.environ.get("WEAVE_RS_STAGE_ROTATE", "0") == "1"
# Wide math layout for the BLOCK_M=8 swap-AB rows: four math warpgroups (16
# warps, 640 threads per CTA), each owning one 64-row weight half per stage, so
# every scheduler interleaves four latency-bound decode streams instead of two.
RS_WIDE_MATH = os.environ.get("WEAVE_RS_WIDE_MATH", "0") == "1"
# Compact LUT for the RS decode: 4-byte low-word gathers plus arithmetic
# reconstruction of the high word (exact for codes 8..127; codes 0..7 read an
# eight-entry exception table). Same bytes as the full table, half the gather
# width.
RS_LUT_COMPACT = os.environ.get("WEAVE_RS_LUT_COMPACT", "0") == "1"
RS_LUT_COMPACT_WORDS = 136  # 128 low words + 8 exception high words
RS_LUT_COMPACT_BYTES = RS_LUT_COMPACT_WORDS * 4
RS_WIDE_MATH_WARPS = 16
RS_WIDE_MATH_REGISTERS = int(os.environ.get("WEAVE_RS_WIDE_MATH_REGISTERS", "112"))


def _wide_math_for(schedule) -> bool:
    return bool(
        RS_WIDE_MATH
        and schedule.swap_ab
        and schedule.block_m == 8
        and not (RS_TEAM_SLICE3 and schedule.swap_ab)
    )


def _math_warps_for(schedule) -> int:
    return RS_WIDE_MATH_WARPS if _wide_math_for(schedule) else EPILOGUE_WARPS


def _kernel_threads_for(schedule) -> int:
    return DISPATCH_THREADS + 64 + 32 * _math_warps_for(schedule)
RS_STAGE_ROTATE_MAX_N = int(os.environ.get("WEAVE_RS_STAGE_ROTATE_MAX_N", "16"))
RS_TEAM_SLICE_K0 = 96
RS_TEAM_DECODED_ROW_BYTES = 32
RS_TEAM_DECODED_STAGE_BYTES = RS_WEIGHT_ROWS * RS_TEAM_DECODED_ROW_BYTES
RS_TEAM_BARRIER_ID = 5
RS_TEAM_THREADS = 64
RS_TEAM_ROWS_PER_THREAD = RS_WEIGHT_ROWS // RS_TEAM_THREADS
RS_TEAM_REGISTERS = 80


class SmallRSSchedule(NamedTuple):
    """Material source schedule selected by the production wrapper."""

    model: str
    hidden: int
    intermediate: int
    experts: int
    m: int
    experts_per_wave: int
    block_m: int
    stages: int
    swap_ab: bool
    single_dispatch: bool
    ss_mode2: bool
    requested_rs_arm: bool
    requested_interleaved_arm: bool
    max_pool_tokens: int
    padded_sf_tokens: int


class SmallRSDispatchLayout(NamedTuple):
    """Exact source offsets through the complete symmetric allocation."""

    workspace_bytes: int
    grid_sync_offset: int
    rendezvous_counter_offset: int
    rendezvous_signal_offset: int
    l1_task_count_offset: int
    l2_task_count_offset: int
    expert_send_count_offset: int
    expert_recv_count_offset: int
    expert_recv_sum_offset: int
    l1_arrival_count_offset: int
    l2_arrival_mask_offset: int
    src_token_topk_offset: int
    token_src_metadata_offset: int
    input_token_offset: int
    input_sf_offset: int
    input_topk_idx_offset: int
    input_topk_weights_offset: int
    l1_token_offset: int
    l1_sf_offset: int
    l1_topk_weights_offset: int
    l2_token_offset: int
    l2_sf_offset: int
    combine_token_offset: int
    symmetric_bytes: int


class ExternalTmaBindingPlan(NamedTuple):
    """One source-owned 2-D tensor-map window in the symmetric allocation."""

    resource: str
    byte_offset: int
    dtype: str
    rows: int
    cols: int
    row_pitch_bytes: int


class WeightTmaBindingPlan(NamedTuple):
    """One packed-weight view consumed by ``create_tma_from_spec``."""

    resource: str
    rows: int
    cols: int


class PreparedSmallRSWeaveBindings(NamedTuple):
    """Launch-scoped host resources for the production Weave IR."""

    schedule: SmallRSSchedule
    kernel: object
    allocation: object
    peer_ptrs: object
    tensor_maps: dict[str, object]
    packed_args: list[object]
    keepalive: tuple[object, ...]
    bindings: dict[str, object]


class CompiledSmallRSWeaveArtifact(NamedTuple):
    """Generated and NVRTC-compiled artifact for one source row."""

    schedule: SmallRSSchedule
    arch: str
    kernel: object
    generated_source: str
    generated_source_sha256: str
    cubin: bytes
    cubin_sha256: str
    kernel_symbol: str
    forbidden_blackwell_absent: bool


class PreparedSmallRSWeaveLaunch(NamedTuple):
    """One fully bound launch and its strict artifact receipt."""

    bindings: PreparedSmallRSWeaveBindings
    artifact: CompiledSmallRSWeaveArtifact
    cuda_kernel: object
    prepared_launch: object
    stream: object
    artifact_receipt: dict[str, object]


_WEAVE_ARTIFACT_CACHE: dict[
    tuple[str, int, int], CompiledSmallRSWeaveArtifact
] = {}
_WEAVE_CUDA_KERNEL_CACHE: dict[tuple[int, str, str], object] = {}
_LAST_WEAVE_ARTIFACT_RECEIPT: dict[str, object] | None = None


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _dispatch_layout(schedule: SmallRSSchedule) -> SmallRSDispatchLayout:
    """Reproduce ``layout::Workspace`` and fused-body buffer slicing."""

    experts_per_rank = schedule.experts // NUM_RANKS
    max_recv_tokens_per_expert = NUM_RANKS * MAX_TOKENS_PER_RANK
    max_pool_blocks = schedule.max_pool_tokens // 8

    expert_send_count_offset = WORKSPACE_SIGNAL_BYTES
    expert_recv_count_offset = expert_send_count_offset + schedule.experts * 8
    expert_recv_sum_offset = expert_recv_count_offset + schedule.experts * 8
    l1_arrival_count_offset = expert_recv_sum_offset + experts_per_rank * 8
    l2_arrival_mask_offset = (
        l1_arrival_count_offset + _align(max_pool_blocks, 2) * 4
    )
    src_token_topk_offset = l2_arrival_mask_offset + max_pool_blocks * 8
    token_src_metadata_offset = (
        src_token_topk_offset
        + experts_per_rank
        * NUM_RANKS
        * max_recv_tokens_per_expert
        * 4
    )
    workspace_bytes = _align(
        token_src_metadata_offset + schedule.max_pool_tokens * 12,
        16,
    )

    input_token_offset = workspace_bytes
    input_sf_offset = input_token_offset + MAX_TOKENS_PER_RANK * schedule.hidden
    input_topk_idx_offset = (
        input_sf_offset + MAX_TOKENS_PER_RANK * (schedule.hidden // 32)
    )
    input_topk_weights_offset = (
        input_topk_idx_offset + MAX_TOKENS_PER_RANK * TOP_K * 8
    )
    l1_token_offset = (
        input_topk_weights_offset + MAX_TOKENS_PER_RANK * TOP_K * 4
    )
    l1_sf_offset = l1_token_offset + schedule.max_pool_tokens * schedule.hidden
    l1_topk_weights_offset = (
        l1_sf_offset
        + schedule.padded_sf_tokens * (schedule.hidden // 32)
    )
    l2_token_offset = l1_topk_weights_offset + schedule.max_pool_tokens * 4
    l2_sf_offset = (
        l2_token_offset + schedule.max_pool_tokens * schedule.intermediate
    )
    combine_token_offset = (
        l2_sf_offset
        + schedule.padded_sf_tokens * (schedule.intermediate // 16)
    )
    symmetric_bytes = (
        combine_token_offset
        + TOP_K * MAX_TOKENS_PER_RANK * schedule.hidden * 2
    )
    return SmallRSDispatchLayout(
        workspace_bytes=workspace_bytes,
        grid_sync_offset=0,
        rendezvous_counter_offset=16,
        rendezvous_signal_offset=20,
        l1_task_count_offset=28,
        l2_task_count_offset=32,
        expert_send_count_offset=expert_send_count_offset,
        expert_recv_count_offset=expert_recv_count_offset,
        expert_recv_sum_offset=expert_recv_sum_offset,
        l1_arrival_count_offset=l1_arrival_count_offset,
        l2_arrival_mask_offset=l2_arrival_mask_offset,
        src_token_topk_offset=src_token_topk_offset,
        token_src_metadata_offset=token_src_metadata_offset,
        input_token_offset=input_token_offset,
        input_sf_offset=input_sf_offset,
        input_topk_idx_offset=input_topk_idx_offset,
        input_topk_weights_offset=input_topk_weights_offset,
        l1_token_offset=l1_token_offset,
        l1_sf_offset=l1_sf_offset,
        l1_topk_weights_offset=l1_topk_weights_offset,
        l2_token_offset=l2_token_offset,
        l2_sf_offset=l2_sf_offset,
        combine_token_offset=combine_token_offset,
        symmetric_bytes=symmetric_bytes,
    )


PRODUCTION_TMA_RESOURCES = (
    "l1_activation",
    "l1_activation_scale",
    "l1_packed_weights",
    "l1_output",
    "l2_activation",
    "l2_activation_scale",
    "l2_packed_weights",
)


def _external_tma_binding_plan(
    schedule: SmallRSSchedule,
) -> tuple[ExternalTmaBindingPlan, ...]:
    """Resolve the five exact raw-state windows from ``descriptor_bind.hpp``."""

    layout = _dispatch_layout(schedule)
    return (
        ExternalTmaBindingPlan(
            "l1_activation",
            layout.l1_token_offset,
            "u8",
            schedule.max_pool_tokens,
            schedule.hidden,
            schedule.hidden,
        ),
        ExternalTmaBindingPlan(
            "l1_activation_scale",
            layout.l1_sf_offset,
            "f32",
            schedule.hidden // RS_BLOCK_K,
            schedule.padded_sf_tokens,
            schedule.padded_sf_tokens * 4,
        ),
        # The source encodes two descriptors over this exact byte range: the
        # L1 output store has no swizzle, while the L2 activation load uses the
        # descriptor-owned 128-byte swizzle.
        ExternalTmaBindingPlan(
            "l1_output",
            layout.l2_token_offset,
            "u8",
            schedule.max_pool_tokens,
            schedule.intermediate,
            schedule.intermediate,
        ),
        ExternalTmaBindingPlan(
            "l2_activation",
            layout.l2_token_offset,
            "u8",
            schedule.max_pool_tokens,
            schedule.intermediate,
            schedule.intermediate,
        ),
        ExternalTmaBindingPlan(
            "l2_activation_scale",
            layout.l2_sf_offset,
            "f32",
            schedule.intermediate // RS_BLOCK_K,
            schedule.padded_sf_tokens,
            schedule.padded_sf_tokens * 4,
        ),
    )


def _weight_tma_binding_plan(
    schedule: SmallRSSchedule,
) -> tuple[WeightTmaBindingPlan, ...]:
    """Resolve the two source packed-weight tensor views."""

    experts_per_rank = schedule.experts // NUM_RANKS
    l1_stride = schedule.hidden // RS_BLOCK_K * RS_WEIGHT_ROW_BYTES
    l2_stride = schedule.intermediate // RS_BLOCK_K * RS_WEIGHT_ROW_BYTES
    return (
        WeightTmaBindingPlan(
            "l1_packed_weights",
            experts_per_rank * 2 * schedule.intermediate,
            l1_stride,
        ),
        WeightTmaBindingPlan(
            "l2_packed_weights",
            experts_per_rank * schedule.hidden,
            l2_stride,
        ),
    )


# ``smallm_kernel_common.cuh::kf_dispatch_m`` is the authority for these ten
# entries. Rows are separate only where the source changes material schedule
# axes (EPW, block-M, stages, swap-AB, or dispatch strategy). One deliberate
# deviation: every swap-AB row runs a 6-stage L1 ring (the source buckets use 3
# or 4) because the rotated K loop keeps two stages resident and the weight
# stream is latency-bound with fewer slots in flight (H20 paired timing: 4
# stages 0.897, 6 stages 0.883, 8 stages 0.886 geomean vs the dynamic+RS arm);
# the SS row keeps its source depth (its decoded tiles do not fit more).
SMALL_RS_SCHEDULES = {
    (4096, 8): SmallRSSchedule(
        "flash",
        4096,
        2048,
        256,
        8,
        16,
        8,
        6,
        True,
        True,
        True,
        True,
        True,
        411648,
        6586368,
    ),
    (4096, 16): SmallRSSchedule(
        "flash",
        4096,
        2048,
        256,
        16,
        32,
        8,
        6,
        True,
        True,
        True,
        True,
        True,
        411648,
        6586368,
    ),
    (4096, 32): SmallRSSchedule(
        "flash",
        4096,
        2048,
        256,
        32,
        32,
        8,
        6,
        True,
        False,
        True,
        True,
        True,
        411648,
        6586368,
    ),
    (4096, 64): SmallRSSchedule(
        "flash",
        4096,
        2048,
        256,
        64,
        32,
        24,
        6,
        True,
        True,
        False,
        True,
        False,
        411648,
        6586368,
    ),
    (4096, 128): SmallRSSchedule(
        "flash",
        4096,
        2048,
        256,
        128,
        32,
        64,
        3,
        False,
        False,
        True,
        True,
        False,
        411648,
        6586368,
    ),
    (7168, 8): SmallRSSchedule(
        "pro",
        7168,
        3072,
        384,
        8,
        48,
        8,
        6,
        True,
        True,
        True,
        True,
        True,
        414720,
        6635520,
    ),
    (7168, 16): SmallRSSchedule(
        "pro",
        7168,
        3072,
        384,
        16,
        24,
        8,
        6,
        True,
        True,
        True,
        True,
        True,
        414720,
        6635520,
    ),
    (7168, 32): SmallRSSchedule(
        "pro",
        7168,
        3072,
        384,
        32,
        48,
        8,
        6,
        True,
        False,
        True,
        True,
        True,
        414720,
        6635520,
    ),
    (7168, 64): SmallRSSchedule(
        "pro",
        7168,
        3072,
        384,
        64,
        48,
        24,
        6,
        True,
        True,
        False,
        False,
        True,
        414720,
        6635520,
    ),
    (7168, 128): SmallRSSchedule(
        "pro",
        7168,
        3072,
        384,
        128,
        48,
        24,
        6,
        True,
        True,
        True,
        True,
        True,
        414720,
        6635520,
    ),
}


def select_small_rs_schedule(hidden: int, m: int) -> SmallRSSchedule:
    """Select exactly one source-backed material schedule."""

    try:
        schedule = SMALL_RS_SCHEDULES[(hidden, m)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported small-RS schedule hidden={hidden}, M={m}"
        ) from exc
    return schedule


def _uses_current_interleaved_scheduler() -> bool:
    """Return the wrapper's effective current-provider scheduler choice.

    ``smallm_kernel_common.cuh`` accepts a per-row
    ``CURRENT_INTERLEAVED`` request bit for auditability, but the delivered
    dynamic wrapper computes ``kInterleaved = kCurrent``.  This module retains
    the current provider, so all ten rows use the interleaved physical plan.
    """

    return True


@loom.weave_fragment
def _emit_compact_lut_row(lm, *, lutc_smem, scale, row):
    """Load one compact LUT low word and reconstruct the high word (source compact mode)."""

    lo_word: lm.u32[1]
    lm.smem_load_vec(dst=lo_word, src_addr=lutc_smem.addr + scale.i32() * 4, count=1)
    hi_word: lm.u32 = lm.prmt_b32(lo_word[0], lo_word[0], 0x3232) + 0x10100808
    sat: lm.u32 = hi_word & 0x80808080
    hi_word = (hi_word | (sat - (sat >> 7))) & (sat ^ 0xFFFFFFFF)
    any_subnormal: lm.i32 = lm.warp_vote(vote=lm.VOTE_ANY, predicate=scale < 8)
    if any_subnormal != 0:
        exc_word: lm.u32[1]
        lm.smem_load_vec(dst=exc_word, src_addr=lutc_smem.addr + 512 + (scale & 7).i32() * 4, count=1)
        if scale < 8:
            hi_word = exc_word[0]
    row[0] = lo_word[0]
    row[1] = hi_word


@loom.weave_fragment
def _emit_rs_half_decode(
    lm,
    *,
    half,
    packed_stage_addr,
    lut_smem,
    a_frags,
    num_slices=4,
    wide=False,
    lutc_smem=None,
):
    """Decode one 64-row Mode2 weight half into live K32 RS fragments.

    ``num_slices`` is 4 for the pure register-source path and 3 when the
    dispatch decode team supplies K-slice 3 through shared memory.
    """

    math_wg: lm.i32 = (lm.warp_id - 4) // 4
    warp_in_wg: lm.i32 = (lm.warp_id - 4) % 4
    row_in_warp: lm.i32 = lm.lane_id // 4
    word_sel: lm.i32 = (lm.lane_id >> 1) & 1
    # Plain integer arithmetic (a Python ternary would be traced into the IR).
    wg_rows = 128 - 64 * int(wide)
    half_rows = half * 64 * (1 - int(wide))
    decode_row: lm.i32 = (
        math_wg * wg_rows
        + half_rows
        + warp_in_wg * 16
        + row_in_warp
        + ((lm.lane_id & 1) << 3)
    )
    packed_row_addr: lm.i32 = (
        packed_stage_addr + decode_row * RS_WEIGHT_ROW_BYTES
    )
    scale_words: lm.u32[2]
    lm.smem_load_vec(
        dst=scale_words,
        src_addr=packed_row_addr + 64,
        count=2,
    )
    for k_slice in lm.range(0, num_slices, constexpr=True):
        packed_lo: lm.u32[1]
        packed_hi: lm.u32[1]
        if RS_PACKED_VEC4:
            packed_quad: lm.u32[4]
            lm.smem_load_vec(
                dst=packed_quad,
                src_addr=packed_row_addr + k_slice * 16,
                count=4,
            )
            packed_lo[0] = packed_quad[0]
            packed_hi[0] = packed_quad[2]
            if word_sel != 0:
                packed_lo[0] = packed_quad[1]
                packed_hi[0] = packed_quad[3]
        else:
            lm.smem_load_vec(
                dst=packed_lo,
                src_addr=packed_row_addr + k_slice * 16 + word_sel * 4,
                count=1,
            )
            lm.smem_load_vec(
                dst=packed_hi,
                src_addr=packed_row_addr + k_slice * 16 + 8 + word_sel * 4,
                count=1,
            )
        scale_word: lm.u32 = scale_words[0]
        if k_slice >= 2:
            scale_word = scale_words[1]
        scale_shift: lm.i32 = (k_slice & 1) * 16
        scale_lo: lm.u32 = (scale_word >> scale_shift) & 0x7F
        scale_hi: lm.u32 = (scale_word >> (scale_shift + 8)) & 0x7F
        lut_lo: lm.u32[2]
        lut_hi: lm.u32[2]
        use_compact_lut = int(RS_LUT_COMPACT) * int(lutc_smem is not None)  # plain ints: no traced BoolOp
        if use_compact_lut:
            _emit_compact_lut_row(lm, lutc_smem=lutc_smem, scale=scale_lo, row=lut_lo)
            _emit_compact_lut_row(lm, lutc_smem=lutc_smem, scale=scale_hi, row=lut_hi)
        else:
            lm.smem_load_vec(
                dst=lut_lo,
                src_addr=lut_smem.addr + scale_lo.i32() * 8,
                count=2,
            )
            lm.smem_load_vec(
                dst=lut_hi,
                src_addr=lut_smem.addr + scale_hi.i32() * 8,
                count=2,
            )
        decoded_lo_hi: lm.u32 = 0
        decoded_lo_lo: lm.u32 = 0
        decoded_hi_hi: lm.u32 = 0
        decoded_hi_lo: lm.u32 = 0
        lm.nvfp4_mode2_lut_decode_word(
            dst_hi=decoded_lo_hi,
            dst_lo=decoded_lo_lo,
            packed=packed_lo[0],
            lut_lo=lut_lo[0],
            lut_hi=lut_lo[1],
        )
        lm.nvfp4_mode2_lut_decode_word(
            dst_hi=decoded_hi_hi,
            dst_lo=decoded_hi_lo,
            packed=packed_hi[0],
            lut_lo=lut_hi[0],
            lut_hi=lut_hi[1],
        )
        keep_lo: lm.u32 = decoded_lo_hi
        ship_lo: lm.u32 = decoded_lo_lo
        keep_hi: lm.u32 = decoded_hi_hi
        ship_hi: lm.u32 = decoded_hi_lo
        if (lm.lane_id & 1) != 0:
            keep_lo = decoded_lo_lo
            ship_lo = decoded_lo_hi
            keep_hi = decoded_hi_lo
            ship_hi = decoded_hi_hi
        recv_lo: lm.u32 = lm.shfl_xor_sync(ship_lo, 1, dtype=lm.u32)
        recv_hi: lm.u32 = lm.shfl_xor_sync(ship_hi, 1, dtype=lm.u32)
        frag_base: lm.i32 = k_slice * 4
        if (lm.lane_id & 1) == 0:
            a_frags[frag_base + 0] = keep_lo
            a_frags[frag_base + 1] = recv_lo
            a_frags[frag_base + 2] = keep_hi
            a_frags[frag_base + 3] = recv_hi
        else:
            a_frags[frag_base + 0] = recv_lo
            a_frags[frag_base + 1] = keep_lo
            a_frags[frag_base + 2] = recv_hi
            a_frags[frag_base + 3] = keep_hi
        for frag_word in lm.range(0, 4, constexpr=True):
            lm.wgmma_fence_operand(a_frags[frag_base + frag_word])


@loom.weave_fragment
def _emit_rs_half_mma(
    lm,
    *,
    n_swap,
    act_stage,
    a_frags,
    swap_accum,
    team_slice3=False,
    decoded_smem=None,
    decoded_stage=None,
    decoded_row0=None,
):
    """Issue and commit one half's four WGMMA K32 slices.

    Slices 0-2 are register-source. Slice 3 is register-source too unless the
    dispatch decode team published it (``team_slice3``): then it is one
    shared/shared WGMMA over the 64-row, 32-byte-swizzled ``decoded_tile``.
    The K order is unchanged, so the accumulator is bit-identical.
    """

    accum_count = n_swap // 2
    swap_accum.fill_(0.0)
    for accum_idx in lm.range(0, accum_count, constexpr=True):
        lm.wgmma_fence_operand(swap_accum[accum_idx])
    lm.wgmma_fence()
    lm.mma(
        swap_accum,
        a_frags[0:4],
        act_stage.tile((0, 0), (n_swap, 32)),
        init=True,
    )
    lm.mma(
        swap_accum,
        a_frags[4:8],
        act_stage.tile((0, 32), (n_swap, 32)),
        init=False,
    )
    lm.mma(
        swap_accum,
        a_frags[8:12],
        act_stage.tile((0, 64), (n_swap, 32)),
        init=False,
    )
    if team_slice3:
        lm.mma(
            swap_accum,
            decoded_smem[decoded_stage].tile((decoded_row0, 0), (64, 32)),
            act_stage.tile((0, 96), (n_swap, 32)),
            init=False,
        )
    else:
        lm.mma(
            swap_accum,
            a_frags[12:16],
            act_stage.tile((0, 96), (n_swap, 32)),
            init=False,
        )
    lm.wgmma_commit_group()


@loom.weave_fragment
def _emit_rs_half_accumulate(
    lm,
    *,
    half,
    n_swap,
    sfa_stage_addr,
    valid_m,
    final_accum,
    swap_accum,
    a_frags,
):
    """Apply the per-token activation scale of one completed half."""

    accum_count = n_swap // 2
    token_chunks = n_swap // 8
    for accum_idx in lm.range(0, accum_count, constexpr=True):
        lm.wgmma_fence_operand(swap_accum[accum_idx])
    for frag_idx in lm.range(0, 16, constexpr=True):
        lm.wgmma_fence_operand(a_frags[frag_idx])
    col_idx: lm.i32 = lm.lane_id % 4
    for token_chunk in lm.range(0, token_chunks, constexpr=True):
        token_0: lm.i32 = token_chunk * 8 + col_idx * 2
        token_1: lm.i32 = token_0 + 1
        accum_offset: lm.i32 = half * 32 + token_chunk * 4
        if token_0 < valid_m:
            scale_0: lm.f32[1]
            lm.smem_load_vec(
                dst=scale_0,
                src_addr=sfa_stage_addr + token_0 * 4,
                count=1,
            )
            final_accum[accum_offset + 0] = (
                final_accum[accum_offset + 0]
                + scale_0[0] * swap_accum[token_chunk * 4 + 0]
            )
            final_accum[accum_offset + 2] = (
                final_accum[accum_offset + 2]
                + scale_0[0] * swap_accum[token_chunk * 4 + 2]
            )
        if token_1 < valid_m:
            scale_1: lm.f32[1]
            lm.smem_load_vec(
                dst=scale_1,
                src_addr=sfa_stage_addr + token_1 * 4,
                count=1,
            )
            final_accum[accum_offset + 1] = (
                final_accum[accum_offset + 1]
                + scale_1[0] * swap_accum[token_chunk * 4 + 1]
            )
            final_accum[accum_offset + 3] = (
                final_accum[accum_offset + 3]
                + scale_1[0] * swap_accum[token_chunk * 4 + 3]
            )


@loom.weave_fragment
def _emit_rs_task_rotated(
    lm,
    *,
    n_swap,
    num_k_blocks,
    stage_var,
    pipe,
    full_barrier,
    empty_barrier,
    task_info_empty,
    consumed_task_stage,
    act_smem,
    packed_smem,
    sfa_smem,
    lut_smem,
    valid_m,
    final_accum,
    team_slice3=False,
    decoded_smem=None,
    decoded_barrier=None,
    lutc_smem=None,
):
    """Consume one task's K-blocks with a one-half software rotation.

    Half 0 of K-block k+1 is decoded while half 1 of K-block k is still in
    flight, so the decode stream never waits for the WGMMA drain. Two stages
    are resident at the hand-off; the empty arrive for K-block k is issued
    only after its second group has retired. Accumulation order per token is
    the source order, so the result matches the serial schedule bit for bit.
    """

    accum_count = n_swap // 2
    num_rs_slices = 4 - int(team_slice3)
    a_frags_0: lm.u32[16]
    a_frags_1: lm.u32[16]
    swap_accum_0: lm.f32[accum_count]
    swap_accum_1: lm.f32[accum_count]

    lm.wait(full_barrier, stage=stage_var)
    with lm.elected_thread():
        lm.arrive(task_info_empty, stage=consumed_task_stage)
    if RS_FULL_FENCE:
        lm.fence_proxy_shared_cta()
    _emit_rs_half_decode(
        lm,
        half=0,
        packed_stage_addr=packed_smem.stage_addr(stage_var),
        lut_smem=lut_smem,
        a_frags=a_frags_0,
        num_slices=num_rs_slices,
        lutc_smem=lutc_smem,
    )
    for k_block in lm.range(0, num_k_blocks, unroll=1, dtype=lm.i32):
        cur_packed_addr: lm.i32 = packed_smem.stage_addr(stage_var)
        cur_sfa_addr: lm.i32 = sfa_smem.stage_addr(stage_var)
        cur_stage: lm.u32 = stage_var
        if team_slice3:
            # The team's shared/shared slice must be visible before this
            # stage's first WGMMA group is issued.
            lm.wait(decoded_barrier, stage=stage_var)
        _emit_rs_half_mma(
            lm,
            n_swap=n_swap,
            act_stage=act_smem[stage_var],
            a_frags=a_frags_0,
            swap_accum=swap_accum_0,
            team_slice3=team_slice3,
            decoded_smem=decoded_smem,
            decoded_stage=stage_var,
            decoded_row0=((lm.warp_id - 4) // 4) * 128,
        )
        _emit_rs_half_decode(
            lm,
            half=1,
            packed_stage_addr=cur_packed_addr,
            lut_smem=lut_smem,
            a_frags=a_frags_1,
            num_slices=num_rs_slices,
            lutc_smem=lutc_smem,
        )
        _emit_rs_half_mma(
            lm,
            n_swap=n_swap,
            act_stage=act_smem[stage_var],
            a_frags=a_frags_1,
            swap_accum=swap_accum_1,
            team_slice3=team_slice3,
            decoded_smem=decoded_smem,
            decoded_stage=stage_var,
            decoded_row0=((lm.warp_id - 4) // 4) * 128 + 64,
        )
        lm.wgmma_wait_group(n=1)
        _emit_rs_half_accumulate(
            lm,
            half=0,
            n_swap=n_swap,
            sfa_stage_addr=cur_sfa_addr,
            valid_m=valid_m,
            final_accum=final_accum,
            swap_accum=swap_accum_0,
            a_frags=a_frags_0,
        )
        lm.advance(stage_var, pipe)
        if k_block + 1 < num_k_blocks:
            lm.wait(full_barrier, stage=stage_var)
            if RS_FULL_FENCE:
                lm.fence_proxy_shared_cta()
            _emit_rs_half_decode(
                lm,
                half=0,
                packed_stage_addr=packed_smem.stage_addr(stage_var),
                lut_smem=lut_smem,
                a_frags=a_frags_0,
                num_slices=num_rs_slices,
                lutc_smem=lutc_smem,
            )
        lm.wgmma_wait_group(n=0)
        _emit_rs_half_accumulate(
            lm,
            half=1,
            n_swap=n_swap,
            sfa_stage_addr=cur_sfa_addr,
            valid_m=valid_m,
            final_accum=final_accum,
            swap_accum=swap_accum_1,
            a_frags=a_frags_1,
        )
        if RS_STAGE_WG_SYNC:
            if lm.warp_id < 8:
                lm.sync(bar_id=3, threads=128, aligned=True)
            else:
                lm.sync(bar_id=4, threads=128, aligned=True)
        with lm.elected_thread():
            lm.arrive(empty_barrier, stage=cur_stage)


@loom.weave_fragment
def _emit_rs_wide_block(
    lm,
    *,
    n_swap,
    stage_var,
    pipe,
    full_barrier,
    empty_barrier,
    act_smem,
    packed_smem,
    sfa_smem,
    lut_smem,
    valid_m,
    final_accum,
    cur_frags,
    next_frags,
    swap_accum,
    has_next,
):
    """One K block of a warpgroup's single 64-row half (wide layout)."""

    cur_sfa_addr: lm.i32 = sfa_smem.stage_addr(stage_var)
    cur_stage: lm.u32 = stage_var
    _emit_rs_half_mma(
        lm,
        n_swap=n_swap,
        act_stage=act_smem[stage_var],
        a_frags=cur_frags,
        swap_accum=swap_accum,
    )
    lm.advance(stage_var, pipe)
    if has_next:
        lm.wait(full_barrier, stage=stage_var)
        if RS_FULL_FENCE:
            lm.fence_proxy_shared_cta()
        _emit_rs_half_decode(
            lm,
            half=0,
            packed_stage_addr=packed_smem.stage_addr(stage_var),
            lut_smem=lut_smem,
            a_frags=next_frags,
            wide=True,
        )
    lm.wgmma_wait_group(n=0)
    _emit_rs_half_accumulate(
        lm,
        half=0,
        n_swap=n_swap,
        sfa_stage_addr=cur_sfa_addr,
        valid_m=valid_m,
        final_accum=final_accum,
        swap_accum=swap_accum,
        a_frags=cur_frags,
    )
    with lm.elected_thread():
        lm.arrive(empty_barrier, stage=cur_stage)


@loom.weave_fragment
def _emit_rs_task_wide(
    lm,
    *,
    n_swap,
    num_k_blocks,
    stage_var,
    pipe,
    full_barrier,
    empty_barrier,
    task_info_empty,
    consumed_task_stage,
    act_smem,
    packed_smem,
    sfa_smem,
    lut_smem,
    valid_m,
    final_accum,
):
    """Wide layout task body: one 64-row half per warpgroup, decode of K+1 under K's group."""

    accum_count = n_swap // 2
    a_frags: lm.u32[16]
    b_frags: lm.u32[16]
    swap_accum: lm.f32[accum_count]

    lm.wait(full_barrier, stage=stage_var)
    with lm.elected_thread():
        lm.arrive(task_info_empty, stage=consumed_task_stage)
    if RS_FULL_FENCE:
        lm.fence_proxy_shared_cta()
    _emit_rs_half_decode(
        lm,
        half=0,
        packed_stage_addr=packed_smem.stage_addr(stage_var),
        lut_smem=lut_smem,
        a_frags=a_frags,
        wide=True,
    )
    num_pairs: lm.i32 = num_k_blocks // 2
    for pair_idx in lm.range(0, num_pairs, unroll=1, dtype=lm.i32):
        _emit_rs_wide_block(
            lm,
            n_swap=n_swap,
            stage_var=stage_var,
            pipe=pipe,
            full_barrier=full_barrier,
            empty_barrier=empty_barrier,
            act_smem=act_smem,
            packed_smem=packed_smem,
            sfa_smem=sfa_smem,
            lut_smem=lut_smem,
            valid_m=valid_m,
            final_accum=final_accum,
            cur_frags=a_frags,
            next_frags=b_frags,
            swap_accum=swap_accum,
            has_next=True,
        )
        odd_has_next: lm.i32 = 0
        if pair_idx * 2 + 2 < num_k_blocks:
            odd_has_next = 1
        if odd_has_next != 0:
            _emit_rs_wide_block(
                lm,
                n_swap=n_swap,
                stage_var=stage_var,
                pipe=pipe,
                full_barrier=full_barrier,
                empty_barrier=empty_barrier,
                act_smem=act_smem,
                packed_smem=packed_smem,
                sfa_smem=sfa_smem,
                lut_smem=lut_smem,
                valid_m=valid_m,
                final_accum=final_accum,
                cur_frags=b_frags,
                next_frags=a_frags,
                swap_accum=swap_accum,
                has_next=True,
            )
        else:
            _emit_rs_wide_block(
                lm,
                n_swap=n_swap,
                stage_var=stage_var,
                pipe=pipe,
                full_barrier=full_barrier,
                empty_barrier=empty_barrier,
                act_smem=act_smem,
                packed_smem=packed_smem,
                sfa_smem=sfa_smem,
                lut_smem=lut_smem,
                valid_m=valid_m,
                final_accum=final_accum,
                cur_frags=b_frags,
                next_frags=a_frags,
                swap_accum=swap_accum,
                has_next=False,
            )
    if num_k_blocks % 2 != 0:
        _emit_rs_wide_block(
            lm,
            n_swap=n_swap,
            stage_var=stage_var,
            pipe=pipe,
            full_barrier=full_barrier,
            empty_barrier=empty_barrier,
            act_smem=act_smem,
            packed_smem=packed_smem,
            sfa_smem=sfa_smem,
            lut_smem=lut_smem,
            valid_m=valid_m,
            final_accum=final_accum,
            cur_frags=a_frags,
            next_frags=b_frags,
            swap_accum=swap_accum,
            has_next=False,
        )


@loom.weave_fragment
def _emit_l1_swap_epilogue_wide(
    lm,
    *,
    block_m,
    math_warps,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l1_global_scales,
    l1_topk_weights,
    l2_sf,
    l2_arrival_mask,
    l1_output,
    l1_output_smem,
    l1_amax_smem,
    padded_sf_tokens,
):
    """Swap-AB L1 epilogue for the wide layout: one 64-row half per warpgroup.

    Same arithmetic as ``_emit_l1_swap_epilogue`` (clamp-10 fast-math SwiGLU,
    per-route weight, quad/inter-group amax, UE8M0 scale, E4M3 staging, TMA
    store, release-OR); each warpgroup publishes 32 output columns and the
    per-token amax is reduced over ``math_warps`` shared slots.
    """

    math_threads = 32 * math_warps
    token_chunks = block_m // 8
    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    epilogue_thread_idx: lm.i32 = epilogue_warp_idx * 32 + lm.lane_id
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    wg_l1_out_n_idx: lm.i32 = epilogue_wg_idx * 32
    m_idx: lm.i32 = pool_block_idx * block_m

    global_scale_vec = lm.vec_load(
        l1_global_scales,
        index=local_expert_idx,
        count=1,
        dtype=lm.f32,
        dst_dtype=lm.f32,
        cache_hint="read_only",
    )
    l1_global_scale: lm.f32 = global_scale_vec[0]
    swap_v0: lm.f32[token_chunks]
    swap_v1: lm.f32[token_chunks]
    swap_v0.fill_(0.0)
    swap_v1.fill_(0.0)

    for token_chunk in lm.range(0, token_chunks, constexpr=True):
        token_0: lm.i32 = token_chunk * 8 + col_idx * 2
        token_1: lm.i32 = token_0 + 1
        v0_amax: lm.f32 = 0.0
        v1_amax: lm.f32 = 0.0
        accum_offset = token_chunk * 4
        if token_0 < valid_m:
            gate_0: lm.f32 = lm.min(final_accum[accum_offset + 0] * l1_global_scale, 10.0)
            up_0: lm.f32 = lm.max(lm.min(final_accum[accum_offset + 2] * l1_global_scale, 10.0), -10.0)
            weight_0_vec = lm.vec_load(l1_topk_weights, index=m_idx + token_0, count=1, dtype=lm.f32, dst_dtype=lm.f32)
            sigmoid_0: lm.f32 = lm.rcp(1.0 + lm.expf(-gate_0))
            value_0: lm.f32 = gate_0 * sigmoid_0 * up_0 * weight_0_vec[0]
            swap_v0[token_chunk] = value_0
            v0_amax = lm.max(v0_amax, lm.fabs(value_0))
        if token_1 < valid_m:
            gate_1: lm.f32 = lm.min(final_accum[accum_offset + 1] * l1_global_scale, 10.0)
            up_1: lm.f32 = lm.max(lm.min(final_accum[accum_offset + 3] * l1_global_scale, 10.0), -10.0)
            weight_1_vec = lm.vec_load(l1_topk_weights, index=m_idx + token_1, count=1, dtype=lm.f32, dst_dtype=lm.f32)
            sigmoid_1: lm.f32 = lm.rcp(1.0 + lm.expf(-gate_1))
            value_1: lm.f32 = gate_1 * sigmoid_1 * up_1 * weight_1_vec[0]
            swap_v1[token_chunk] = value_1
            v1_amax = lm.max(v1_amax, lm.fabs(value_1))
        peer_0: lm.f32 = lm.shfl_xor_sync(v0_amax, 4)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1: lm.f32 = lm.shfl_xor_sync(v1_amax, 4)
        v1_amax = lm.max(v1_amax, peer_1)
        peer_0 = lm.shfl_xor_sync(v0_amax, 8)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1 = lm.shfl_xor_sync(v1_amax, 8)
        v1_amax = lm.max(v1_amax, peer_1)
        peer_0 = lm.shfl_xor_sync(v0_amax, 16)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1 = lm.shfl_xor_sync(v1_amax, 16)
        v1_amax = lm.max(v1_amax, peer_1)
        if row_idx == 0:
            if token_0 < valid_m:
                l1_amax_smem[token_0 * math_warps + epilogue_warp_idx] = v0_amax
            if token_1 < valid_m:
                l1_amax_smem[token_1 * math_warps + epilogue_warp_idx] = v1_amax

    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)

    if epilogue_thread_idx < valid_m:
        amax: lm.f32 = 0.0
        for source_warp in lm.range(0, math_warps, constexpr=True):
            amax = lm.max(amax, l1_amax_smem[epilogue_thread_idx * math_warps + source_warp])
        scaled_amax: lm.f32 = amax * (1.0 / 448.0)
        scaled_bits: lm.u32 = 0
        lm.bit_cast_store(target=scaled_bits, src=scaled_amax, cast_type=lm.u32)
        exponent: lm.i32 = (scaled_bits >> 23).i32()
        mantissa: lm.u32 = scaled_bits & 0x7FFFFF
        scale_exponent: lm.i32 = exponent - 127
        if mantissa != 0:
            scale_exponent = scale_exponent + 1
        scale_bits: lm.u32 = (scale_exponent + 127).u32() << 23
        scale_inv_bits: lm.u32 = (-scale_exponent + 127).u32() << 23
        scale: lm.f32 = 0.0
        scale_inv: lm.f32 = 0.0
        lm.bit_cast_store(target=scale, src=scale_bits, cast_type=lm.f32)
        lm.bit_cast_store(target=scale_inv, src=scale_inv_bits, cast_type=lm.f32)
        sf_index: lm.i32 = n_block_idx * padded_sf_tokens + m_idx + epilogue_thread_idx
        lm.gmem_store(l2_sf, scale, index=sf_index, dtype=lm.f32, src_dtype=lm.f32)
        l1_amax_smem[epilogue_thread_idx * math_warps] = scale_inv

    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)

    for token_chunk in lm.range(0, token_chunks, constexpr=True):
        token_0: lm.i32 = token_chunk * 8 + col_idx * 2
        token_1: lm.i32 = token_0 + 1
        out_col: lm.i32 = wg_l1_out_n_idx + warp_idx_in_wg * 8 + row_idx
        quant_pair: lm.f32[4]
        quant_pair.fill_(0.0)
        if token_0 < valid_m:
            sf_inv_0: lm.f32 = l1_amax_smem[token_0 * math_warps]
            quant_pair[0] = swap_v0[token_chunk] * sf_inv_0
        if token_1 < valid_m:
            sf_inv_1: lm.f32 = l1_amax_smem[token_1 * math_warps]
            quant_pair[1] = swap_v1[token_chunk] * sf_inv_1
        packed_pair = quant_pair.fp8_e4m3()
        if token_0 < valid_m:
            l1_output_smem[token_0 * 128 + out_col] = packed_pair[0] & 0xFF
        if token_1 < valid_m:
            l1_output_smem[token_1 * 128 + out_col] = (packed_pair[0] >> 8) & 0xFF

    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)
    with lm.elected_thread(warp=4):
        lm.fence_proxy_shared_cta()
        lm.tma_store(l1_output, coords=(n_block_idx * 128, m_idx), src=l1_output_smem.addr)
        lm.bulk_commit()
    lm.sync_warp()
    lm.bulk_wait(n=0)
    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)
    with lm.elected_thread(warp=4):
        ready_bit: lm.u64 = 1
        ready_bit = ready_bit << n_block_idx
        lm.gmem_reduction_release(
            l2_arrival_mask, ready_bit, index=pool_block_idx, op=lm.REDUCE_OR, dtype=lm.u64, scope=lm.FENCE_DEVICE
        )
    lm.sync_warp()


@loom.weave_fragment
def _emit_l2_swap_remote_scatter_wide(
    lm,
    *,
    block_m,
    math_warps,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l2_global_scales,
    token_src_metadata,
    sym_buffer,
    combine_token_offset,
    hidden,
    l2_output_smem,
):
    """Swap-AB L2 BF16 staging and remote scatter for the wide layout (64 columns per warpgroup)."""

    math_threads = 32 * math_warps
    global_scale_vec = lm.vec_load(
        l2_global_scales, index=local_expert_idx, count=1, dtype=lm.f32, dst_dtype=lm.f32, cache_hint="read_only"
    )
    l2_global_scale: lm.f32 = global_scale_vec[0]
    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    r_0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    r_1: lm.i32 = r_0 + 8
    wg_n_idx: lm.i32 = epilogue_wg_idx * 64
    num_swap_token_chunks: lm.i32 = (valid_m + 7) // 8

    for token_chunk in lm.range(0, block_m // 8, constexpr=True):
        if token_chunk < num_swap_token_chunks:
            token_0: lm.i32 = token_chunk * 8 + col_idx * 2
            token_1: lm.i32 = token_0 + 1
            accum_offset = token_chunk * 4
            if token_0 < valid_m:
                l2_output_smem.swizzled_scalar_store(
                    row=token_0, col_bytes=(wg_n_idx + r_0) * 2,
                    value=final_accum[accum_offset + 0] * l2_global_scale, dtype=lm.bf16, row_stride_bytes=512,
                )
                l2_output_smem.swizzled_scalar_store(
                    row=token_0, col_bytes=(wg_n_idx + r_1) * 2,
                    value=final_accum[accum_offset + 2] * l2_global_scale, dtype=lm.bf16, row_stride_bytes=512,
                )
            if token_1 < valid_m:
                l2_output_smem.swizzled_scalar_store(
                    row=token_1, col_bytes=(wg_n_idx + r_0) * 2,
                    value=final_accum[accum_offset + 1] * l2_global_scale, dtype=lm.bf16, row_stride_bytes=512,
                )
                l2_output_smem.swizzled_scalar_store(
                    row=token_1, col_bytes=(wg_n_idx + r_1) * 2,
                    value=final_accum[accum_offset + 3] * l2_global_scale, dtype=lm.bf16, row_stride_bytes=512,
                )

    # Every warpgroup staged its own 64 columns; each warpgroup then scatters
    # those columns for all tokens (8 lanes x 16 bytes per token row).
    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)

    row_in_warp_block: lm.i32 = lm.lane_id // 8
    lane_in_row: lm.i32 = lm.lane_id % 8
    n_idx: lm.i32 = n_block_idx * 256 + wg_n_idx
    token: lm.i32 = warp_idx_in_wg * 4 + row_in_warp_block
    if token < valid_m:
        metadata_base: lm.i32 = (pool_block_idx * block_m + token) * 3
        dst_rank_word = lm.vec_load(token_src_metadata, index=metadata_base, count=1, dtype=lm.i32, dst_dtype=lm.i32)
        dst_token_word = lm.vec_load(token_src_metadata, index=metadata_base + 1, count=1, dtype=lm.i32, dst_dtype=lm.i32)
        dst_topk_word = lm.vec_load(token_src_metadata, index=metadata_base + 2, count=1, dtype=lm.i32, dst_dtype=lm.i32)
        dst_rank: lm.i32 = dst_rank_word[0]
        dst_token: lm.i32 = dst_token_word[0]
        dst_topk: lm.i32 = dst_topk_word[0]
        packed: lm.u32[4]
        lm.smem_load_vec(
            dst=packed, src_addr=l2_output_smem.addr + (token * 256 + wg_n_idx + lane_in_row * 8) * 2, count=4
        )
        peer_combine_u32 = lm.ptr_to(lm.byte_ptr(sym_buffer.peer_ptr(dst_rank)) + combine_token_offset, lm.u32)
        dst_word_offset: lm.u64 = (
            ((dst_topk.u64() * MAX_TOKENS_PER_RANK + dst_token.u64()) * hidden + n_idx.u64() + lane_in_row.u64() * 8) // 2
        )
        lm.gmem_store_vec(dst=peer_combine_u32 + dst_word_offset, src=packed)

    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=math_threads, aligned=True)


def _combine_token_start(lm, epilogue_warp_idx, num_tokens, wide):
    """Source eight-warp token partition; extra wide-layout warps get an empty range."""

    base = lm.bid * EPILOGUE_WARPS + epilogue_warp_idx
    if not wide:
        return base
    return lm.if_then_else(epilogue_warp_idx >= EPILOGUE_WARPS, base + num_tokens, base)


def _emit_swap_l1_epilogue(lm, *, wide, math_warps, **kwargs):
    if wide:
        _emit_l1_swap_epilogue_wide(lm, math_warps=math_warps, **kwargs)
    else:
        _emit_l1_swap_epilogue(lm, **kwargs)


def _emit_swap_l2_scatter(lm, *, wide, math_warps, **kwargs):
    if wide:
        _emit_l2_swap_remote_scatter_wide(lm, math_warps=math_warps, **kwargs)
    else:
        _emit_l2_swap_remote_scatter(lm, **kwargs)


@loom.weave_fragment
def _emit_rs_stage_block(
    lm,
    *,
    n_swap,
    stage_var,
    pipe,
    full_barrier,
    empty_barrier,
    act_smem,
    packed_smem,
    sfa_smem,
    lut_smem,
    valid_m,
    final_accum,
    cur_frags_0,
    cur_frags_1,
    next_frags_0,
    next_frags_1,
    swap_accum_0,
    swap_accum_1,
    has_next,
):
    """One K block of the stage-level rotation.

    Issues both halves of the current stage from ``cur_frags``, advances the
    cursor, decodes both halves of the next stage into ``next_frags`` (when
    ``has_next``), then drains the current groups, applies the activation
    scales and releases the current stage.
    """

    cur_sfa_addr: lm.i32 = sfa_smem.stage_addr(stage_var)
    cur_stage: lm.u32 = stage_var
    _emit_rs_half_mma(
        lm,
        n_swap=n_swap,
        act_stage=act_smem[stage_var],
        a_frags=cur_frags_0,
        swap_accum=swap_accum_0,
    )
    _emit_rs_half_mma(
        lm,
        n_swap=n_swap,
        act_stage=act_smem[stage_var],
        a_frags=cur_frags_1,
        swap_accum=swap_accum_1,
    )
    lm.advance(stage_var, pipe)
    if has_next:
        lm.wait(full_barrier, stage=stage_var)
        if RS_FULL_FENCE:
            lm.fence_proxy_shared_cta()
        _emit_rs_half_decode(
            lm,
            half=0,
            packed_stage_addr=packed_smem.stage_addr(stage_var),
            lut_smem=lut_smem,
            a_frags=next_frags_0,
        )
        _emit_rs_half_decode(
            lm,
            half=1,
            packed_stage_addr=packed_smem.stage_addr(stage_var),
            lut_smem=lut_smem,
            a_frags=next_frags_1,
        )
    lm.wgmma_wait_group(n=0)
    _emit_rs_half_accumulate(
        lm,
        half=0,
        n_swap=n_swap,
        sfa_stage_addr=cur_sfa_addr,
        valid_m=valid_m,
        final_accum=final_accum,
        swap_accum=swap_accum_0,
        a_frags=cur_frags_0,
    )
    _emit_rs_half_accumulate(
        lm,
        half=1,
        n_swap=n_swap,
        sfa_stage_addr=cur_sfa_addr,
        valid_m=valid_m,
        final_accum=final_accum,
        swap_accum=swap_accum_1,
        a_frags=cur_frags_1,
    )
    with lm.elected_thread():
        lm.arrive(empty_barrier, stage=cur_stage)


@loom.weave_fragment
def _emit_rs_task_stage_rotated(
    lm,
    *,
    n_swap,
    num_k_blocks,
    stage_var,
    pipe,
    full_barrier,
    empty_barrier,
    task_info_empty,
    consumed_task_stage,
    act_smem,
    packed_smem,
    sfa_smem,
    lut_smem,
    valid_m,
    final_accum,
):
    """Consume one task's K-blocks with a stage-level rotation (two fragment sets)."""

    accum_count = n_swap // 2
    a_frags_0: lm.u32[16]
    a_frags_1: lm.u32[16]
    b_frags_0: lm.u32[16]
    b_frags_1: lm.u32[16]
    swap_accum_0: lm.f32[accum_count]
    swap_accum_1: lm.f32[accum_count]

    lm.wait(full_barrier, stage=stage_var)
    with lm.elected_thread():
        lm.arrive(task_info_empty, stage=consumed_task_stage)
    if RS_FULL_FENCE:
        lm.fence_proxy_shared_cta()
    _emit_rs_half_decode(
        lm,
        half=0,
        packed_stage_addr=packed_smem.stage_addr(stage_var),
        lut_smem=lut_smem,
        a_frags=a_frags_0,
    )
    _emit_rs_half_decode(
        lm,
        half=1,
        packed_stage_addr=packed_smem.stage_addr(stage_var),
        lut_smem=lut_smem,
        a_frags=a_frags_1,
    )
    num_pairs: lm.i32 = num_k_blocks // 2
    for pair_idx in lm.range(0, num_pairs, unroll=1, dtype=lm.i32):
        # Even K block (fragments a), always followed by the odd block.
        _emit_rs_stage_block(
            lm,
            n_swap=n_swap,
            stage_var=stage_var,
            pipe=pipe,
            full_barrier=full_barrier,
            empty_barrier=empty_barrier,
            act_smem=act_smem,
            packed_smem=packed_smem,
            sfa_smem=sfa_smem,
            lut_smem=lut_smem,
            valid_m=valid_m,
            final_accum=final_accum,
            cur_frags_0=a_frags_0,
            cur_frags_1=a_frags_1,
            next_frags_0=b_frags_0,
            next_frags_1=b_frags_1,
            swap_accum_0=swap_accum_0,
            swap_accum_1=swap_accum_1,
            has_next=True,
        )
        # Odd K block (fragments b); the next block exists unless this is the
        # last pair of an even count.
        odd_has_next: lm.i32 = 0
        if pair_idx * 2 + 2 < num_k_blocks:
            odd_has_next = 1
        if odd_has_next != 0:
            _emit_rs_stage_block(
                lm,
                n_swap=n_swap,
                stage_var=stage_var,
                pipe=pipe,
                full_barrier=full_barrier,
                empty_barrier=empty_barrier,
                act_smem=act_smem,
                packed_smem=packed_smem,
                sfa_smem=sfa_smem,
                lut_smem=lut_smem,
                valid_m=valid_m,
                final_accum=final_accum,
                cur_frags_0=b_frags_0,
                cur_frags_1=b_frags_1,
                next_frags_0=a_frags_0,
                next_frags_1=a_frags_1,
                swap_accum_0=swap_accum_0,
                swap_accum_1=swap_accum_1,
                has_next=True,
            )
        else:
            _emit_rs_stage_block(
                lm,
                n_swap=n_swap,
                stage_var=stage_var,
                pipe=pipe,
                full_barrier=full_barrier,
                empty_barrier=empty_barrier,
                act_smem=act_smem,
                packed_smem=packed_smem,
                sfa_smem=sfa_smem,
                lut_smem=lut_smem,
                valid_m=valid_m,
                final_accum=final_accum,
                cur_frags_0=b_frags_0,
                cur_frags_1=b_frags_1,
                next_frags_0=a_frags_0,
                next_frags_1=a_frags_1,
                swap_accum_0=swap_accum_0,
                swap_accum_1=swap_accum_1,
                has_next=False,
            )
    # Odd K-block count: the trailing block was decoded into fragments a.
    if num_k_blocks % 2 != 0:
        _emit_rs_stage_block(
            lm,
            n_swap=n_swap,
            stage_var=stage_var,
            pipe=pipe,
            full_barrier=full_barrier,
            empty_barrier=empty_barrier,
            act_smem=act_smem,
            packed_smem=packed_smem,
            sfa_smem=sfa_smem,
            lut_smem=lut_smem,
            valid_m=valid_m,
            final_accum=final_accum,
            cur_frags_0=a_frags_0,
            cur_frags_1=a_frags_1,
            next_frags_0=b_frags_0,
            next_frags_1=b_frags_1,
            swap_accum_0=swap_accum_0,
            swap_accum_1=swap_accum_1,
            has_next=False,
        )


def _emit_rs_task(lm, *, n_swap, wide=False, **kwargs):
    """Select the wide, stage-level or half-level body for one task."""

    if wide:
        wide_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("team_slice3", "decoded_smem", "decoded_barrier", "lutc_smem")
        }
        if kwargs.get("team_slice3"):
            raise ValueError("the wide layout does not support the decode team")
        _emit_rs_task_wide(lm, n_swap=n_swap, **wide_kwargs)
    elif RS_STAGE_ROTATE and n_swap <= RS_STAGE_ROTATE_MAX_N:
        stage_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in ("team_slice3", "decoded_smem", "decoded_barrier", "lutc_smem")
        }
        if kwargs.get("team_slice3"):
            raise ValueError("the stage-level rotation does not support the decode team")
        _emit_rs_task_stage_rotated(lm, n_swap=n_swap, **stage_kwargs)
    else:
        _emit_rs_task_rotated(lm, n_swap=n_swap, **kwargs)


@loom.weave_fragment
def _emit_team_slice3_decode(
    lm,
    *,
    packed_stage_addr,
    decoded_smem,
    decoded_stage_addr,
    lut_smem,
    team_tid,
):
    """Expand K-slice 3 of four packed weight rows into the 32B-swizzled tile."""

    # Lane-consecutive rows: the 80-byte packed row stride and the swizzled
    # 32-byte decoded rows then spread over the banks (four consecutive rows
    # per thread would serialize every 16-byte store on one bank group).
    for row_idx in lm.range(0, RS_TEAM_ROWS_PER_THREAD, constexpr=True):
        row: lm.i32 = team_tid + row_idx * RS_TEAM_THREADS
        row_addr: lm.i32 = packed_stage_addr + row * RS_WEIGHT_ROW_BYTES
        packed: lm.u32[4]
        lm.smem_load_vec(dst=packed, src_addr=row_addr + 48, count=4)
        scale_word: lm.u32[1]
        lm.smem_load_vec(dst=scale_word, src_addr=row_addr + 68, count=1)
        scale_lo: lm.u32 = (scale_word[0] >> 16) & 0x7F
        scale_hi: lm.u32 = (scale_word[0] >> 24) & 0x7F
        lut_lo: lm.u32[2]
        lut_hi: lm.u32[2]
        lm.smem_load_vec(dst=lut_lo, src_addr=lut_smem.addr + scale_lo.i32() * 8, count=2)
        lm.smem_load_vec(dst=lut_hi, src_addr=lut_smem.addr + scale_hi.i32() * 8, count=2)
        w0_hi: lm.u32 = 0
        w0_lo: lm.u32 = 0
        w1_hi: lm.u32 = 0
        w1_lo: lm.u32 = 0
        w2_hi: lm.u32 = 0
        w2_lo: lm.u32 = 0
        w3_hi: lm.u32 = 0
        w3_lo: lm.u32 = 0
        lm.nvfp4_mode2_lut_decode_word(dst_hi=w0_hi, dst_lo=w0_lo, packed=packed[0], lut_lo=lut_lo[0], lut_hi=lut_lo[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=w1_hi, dst_lo=w1_lo, packed=packed[1], lut_lo=lut_lo[0], lut_hi=lut_lo[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=w2_hi, dst_lo=w2_lo, packed=packed[2], lut_lo=lut_hi[0], lut_hi=lut_hi[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=w3_hi, dst_lo=w3_lo, packed=packed[3], lut_lo=lut_hi[0], lut_hi=lut_hi[1])
        decoded_smem.swizzled_vec_store(
            row=row, col_bytes=0, values=(w0_hi, w0_lo, w1_hi, w1_lo), base_addr=decoded_stage_addr
        )
        decoded_smem.swizzled_vec_store(
            row=row, col_bytes=16, values=(w2_hi, w2_lo, w3_hi, w3_lo), base_addr=decoded_stage_addr
        )


@loom.weave_fragment
def _emit_team_slice3_loop(
    lm,
    *,
    task_info_full,
    task_info_empty,
    task_info_smem,
    task_info_pipe,
    full_barrier,
    decoded_barrier,
    pipe,
    packed_smem,
    decoded_smem,
    lut_smem,
):
    """Dispatch decode team: follow the task stream and publish K-slice 3 per stage.

    Both dispatch warps consume the TaskInfo mailbox (and return the slot, so
    the producer cannot recycle it before the team read it), wait for each
    stage's TMA arrival, expand K-slice 3 of all 256 packed rows, rendezvous on
    the team named barrier and arrive on the stage's ``decoded`` mbarrier.
    """

    team_task_stage: lm.u32 = 0
    team_stage: lm.u32 = 0
    team_tid: lm.i32 = lm.tid
    while True:
        lm.wait(task_info_full, stage=team_task_stage)
        team_task_words: lm.u32[8]
        lm.smem_load_vec(dst=team_task_words, src_addr=task_info_smem.stage_addr(team_task_stage), count=4)
        lm.smem_load_vec(
            dst=team_task_words,
            src_addr=task_info_smem.stage_addr(team_task_stage) + 16,
            count=4,
            dst_offset=4,
        )
        team_task_phase: lm.u32 = team_task_words[0]
        team_consumed_stage: lm.u32 = team_task_stage
        lm.advance(team_task_stage, task_info_pipe)
        if team_task_phase == 0:
            break
        team_num_k_blocks: lm.i32 = (team_task_words[7] // RS_BLOCK_K).i32()
        with lm.elected_thread():
            lm.arrive(task_info_empty, stage=team_consumed_stage)
        for _team_k in lm.range(0, team_num_k_blocks, unroll=1, dtype=lm.i32):
            lm.wait(full_barrier, stage=team_stage)
            lm.fence_proxy_shared_cta()
            _emit_team_slice3_decode(
                lm,
                packed_stage_addr=packed_smem.stage_addr(team_stage),
                decoded_smem=decoded_smem,
                decoded_stage_addr=decoded_smem.stage_addr(team_stage),
                lut_smem=lut_smem,
                team_tid=team_tid,
            )
            lm.fence_proxy_shared_cta()
            lm.sync(bar_id=RS_TEAM_BARRIER_ID, threads=RS_TEAM_THREADS, aligned=True)
            if team_tid == 0:
                lm.arrive(decoded_barrier, stage=team_stage)
            lm.advance(team_stage, pipe)


@loom.weave_fragment
def _emit_l1_rs_stage(
    lm,
    *,
    n_swap,
    act_stage,
    packed_stage_addr,
    lut_smem,
    sfa_stage_addr,
    valid_m,
    final_accum,
):
    """Emit both source Mode2 RS weight halves for one BK128 stage.

    ``n_swap`` is an unannotated trace-time integer, so callers may place this
    fragment beneath runtime ``valid_m`` branches while each WGMMA instruction
    retains one of the source's static M64N8/N16/N24K32 shapes.
    """

    accum_count = n_swap // 2
    token_chunks = n_swap // 8
    if RS_STAGE_HALF_OVERLAP:
        # Half 1 decodes while half 0's WGMMA group is in flight; the two
        # groups retire in order and land in disjoint final_accum slots.
        a_frags_0: lm.u32[16]
        a_frags_1: lm.u32[16]
        swap_accum_0: lm.f32[accum_count]
        swap_accum_1: lm.f32[accum_count]
        _emit_rs_half_decode(
            lm,
            half=0,
            packed_stage_addr=packed_stage_addr,
            lut_smem=lut_smem,
            a_frags=a_frags_0,
        )
        _emit_rs_half_mma(
            lm,
            n_swap=n_swap,
            act_stage=act_stage,
            a_frags=a_frags_0,
            swap_accum=swap_accum_0,
        )
        _emit_rs_half_decode(
            lm,
            half=1,
            packed_stage_addr=packed_stage_addr,
            lut_smem=lut_smem,
            a_frags=a_frags_1,
        )
        _emit_rs_half_mma(
            lm,
            n_swap=n_swap,
            act_stage=act_stage,
            a_frags=a_frags_1,
            swap_accum=swap_accum_1,
        )
        lm.wgmma_wait_group(n=1)
        _emit_rs_half_accumulate(
            lm,
            half=0,
            n_swap=n_swap,
            sfa_stage_addr=sfa_stage_addr,
            valid_m=valid_m,
            final_accum=final_accum,
            swap_accum=swap_accum_0,
            a_frags=a_frags_0,
        )
        lm.wgmma_wait_group(n=0)
        _emit_rs_half_accumulate(
            lm,
            half=1,
            n_swap=n_swap,
            sfa_stage_addr=sfa_stage_addr,
            valid_m=valid_m,
            final_accum=final_accum,
            swap_accum=swap_accum_1,
            a_frags=a_frags_1,
        )
        return
    math_wg: lm.i32 = (lm.warp_id - 4) // 4
    warp_in_wg: lm.i32 = (lm.warp_id - 4) % 4
    row_in_warp: lm.i32 = lm.lane_id // 4
    word_sel: lm.i32 = (lm.lane_id >> 1) & 1

    for half in lm.range(0, 2, constexpr=True):
        decode_row: lm.i32 = (
            math_wg * 128
            + half * 64
            + warp_in_wg * 16
            + row_in_warp
            + ((lm.lane_id & 1) << 3)
        )
        packed_row_addr: lm.i32 = (
            packed_stage_addr + decode_row * RS_WEIGHT_ROW_BYTES
        )
        scale_words: lm.u32[2]
        lm.smem_load_vec(
            dst=scale_words,
            src_addr=packed_row_addr + 64,
            count=2,
        )

        # Keep all four K32 register fragments live before issuing the one
        # source WGMMA group.
        a_frags: lm.u32[16]
        for k_slice in lm.range(0, 4, constexpr=True):
            packed_lo: lm.u32[1]
            packed_hi: lm.u32[1]
            lm.smem_load_vec(
                dst=packed_lo,
                src_addr=(
                    packed_row_addr + k_slice * 16 + word_sel * 4
                ),
                count=1,
            )
            lm.smem_load_vec(
                dst=packed_hi,
                src_addr=(
                    packed_row_addr
                    + k_slice * 16
                    + 8
                    + word_sel * 4
                ),
                count=1,
            )
            scale_word: lm.u32 = scale_words[0]
            if k_slice >= 2:
                scale_word = scale_words[1]
            scale_shift: lm.i32 = (k_slice & 1) * 16
            scale_lo: lm.u32 = (scale_word >> scale_shift) & 0x7F
            scale_hi: lm.u32 = (
                scale_word >> (scale_shift + 8)
            ) & 0x7F
            lut_lo: lm.u32[2]
            lut_hi: lm.u32[2]
            lm.smem_load_vec(
                dst=lut_lo,
                src_addr=lut_smem.addr + scale_lo.i32() * 8,
                count=2,
            )
            lm.smem_load_vec(
                dst=lut_hi,
                src_addr=lut_smem.addr + scale_hi.i32() * 8,
                count=2,
            )

            decoded_lo_hi: lm.u32 = 0
            decoded_lo_lo: lm.u32 = 0
            decoded_hi_hi: lm.u32 = 0
            decoded_hi_lo: lm.u32 = 0
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=decoded_lo_hi,
                dst_lo=decoded_lo_lo,
                packed=packed_lo[0],
                lut_lo=lut_lo[0],
                lut_hi=lut_lo[1],
            )
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=decoded_hi_hi,
                dst_lo=decoded_hi_lo,
                packed=packed_hi[0],
                lut_lo=lut_hi[0],
                lut_hi=lut_hi[1],
            )

            keep_lo: lm.u32 = decoded_lo_hi
            ship_lo: lm.u32 = decoded_lo_lo
            keep_hi: lm.u32 = decoded_hi_hi
            ship_hi: lm.u32 = decoded_hi_lo
            if (lm.lane_id & 1) != 0:
                keep_lo = decoded_lo_lo
                ship_lo = decoded_lo_hi
                keep_hi = decoded_hi_lo
                ship_hi = decoded_hi_hi
            recv_lo: lm.u32 = lm.shfl_xor_sync(
                ship_lo,
                1,
                dtype=lm.u32,
            )
            recv_hi: lm.u32 = lm.shfl_xor_sync(
                ship_hi,
                1,
                dtype=lm.u32,
            )
            frag_base: lm.i32 = k_slice * 4
            if (lm.lane_id & 1) == 0:
                a_frags[frag_base + 0] = keep_lo
                a_frags[frag_base + 1] = recv_lo
                a_frags[frag_base + 2] = keep_hi
                a_frags[frag_base + 3] = recv_hi
            else:
                a_frags[frag_base + 0] = recv_lo
                a_frags[frag_base + 1] = keep_lo
                a_frags[frag_base + 2] = recv_hi
                a_frags[frag_base + 3] = keep_hi
            for frag_word in lm.range(0, 4, constexpr=True):
                lm.wgmma_fence_operand(a_frags[frag_base + frag_word])

        swap_accum: lm.f32[accum_count]
        swap_accum.fill_(0.0)
        for accum_idx in lm.range(0, accum_count, constexpr=True):
            lm.wgmma_fence_operand(swap_accum[accum_idx])
        lm.wgmma_fence()
        lm.mma(
            swap_accum,
            a_frags[0:4],
            act_stage.tile((0, 0), (n_swap, 32)),
            init=True,
        )
        lm.mma(
            swap_accum,
            a_frags[4:8],
            act_stage.tile((0, 32), (n_swap, 32)),
            init=False,
        )
        lm.mma(
            swap_accum,
            a_frags[8:12],
            act_stage.tile((0, 64), (n_swap, 32)),
            init=False,
        )
        lm.mma(
            swap_accum,
            a_frags[12:16],
            act_stage.tile((0, 96), (n_swap, 32)),
            init=False,
        )
        lm.wgmma_commit_group()
        for accum_idx in lm.range(0, accum_count, constexpr=True):
            lm.wgmma_fence_operand(swap_accum[accum_idx])
        lm.wgmma_wait_group(n=0)
        for frag_idx in lm.range(0, 16, constexpr=True):
            lm.wgmma_fence_operand(a_frags[frag_idx])
        col_idx: lm.i32 = lm.lane_id % 4
        for token_chunk in lm.range(
            0,
            token_chunks,
            constexpr=True,
        ):
            token_0: lm.i32 = token_chunk * 8 + col_idx * 2
            token_1: lm.i32 = token_0 + 1
            accum_offset: lm.i32 = half * 32 + token_chunk * 4
            if token_0 < valid_m:
                scale_0: lm.f32[1]
                lm.smem_load_vec(
                    dst=scale_0,
                    src_addr=sfa_stage_addr + token_0 * 4,
                    count=1,
                )
                final_accum[accum_offset + 0] = (
                    final_accum[accum_offset + 0]
                    + scale_0[0] * swap_accum[token_chunk * 4 + 0]
                )
                final_accum[accum_offset + 2] = (
                    final_accum[accum_offset + 2]
                    + scale_0[0] * swap_accum[token_chunk * 4 + 2]
                )
            if token_1 < valid_m:
                scale_1: lm.f32[1]
                lm.smem_load_vec(
                    dst=scale_1,
                    src_addr=sfa_stage_addr + token_1 * 4,
                    count=1,
                )
                final_accum[accum_offset + 1] = (
                    final_accum[accum_offset + 1]
                    + scale_1[0] * swap_accum[token_chunk * 4 + 1]
                )
                final_accum[accum_offset + 3] = (
                    final_accum[accum_offset + 3]
                    + scale_1[0] * swap_accum[token_chunk * 4 + 3]
                )


@loom.weave_fragment
def _emit_nonswap_stage(
    lm,
    *,
    act_stage,
    packed_stage_addr,
    decoded_stage,
    decoded_smem,
    decoded_stage_addr,
    lut_smem,
    sfa_stage_addr,
    empty_barrier,
    stage,
    final_accum,
):
    """Decode and consume one source non-swap BK128 L1/L2 stage.

    The eight math warps cooperatively expand one packed N256-by-K128 Mode2
    tile.  Each warpgroup then consumes its own N128 half with four
    M64N128K32 shared/shared WGMMA instructions, matching
    ``fused_body.inl:877-900,1099-1133,1250-1293``.  The phase-specific
    activation, scale, and weight descriptors are selected by the loaders;
    both phases use this same shared-source WGMMA body.
    """

    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_thread_idx: lm.i32 = epilogue_warp_idx * 32 + lm.lane_id
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    r_0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    r_1: lm.i32 = r_0 + 8
    packed_row_addr: lm.i32 = (
        packed_stage_addr + epilogue_thread_idx * RS_WEIGHT_ROW_BYTES
    )
    scale_words: lm.u32[2]
    lm.smem_load_vec(
        dst=scale_words,
        src_addr=packed_row_addr + 64,
        count=2,
    )

    # dequant_smem_b_from_packed_mode2_nibble: each of the 256 math
    # threads owns one packed row and writes the eight 16-value groups with
    # the exact 128-byte XOR swizzle consumed by the WGMMA descriptors.
    # The source reads activation scales before warpgroup_arrive.
    scale_0: lm.f32[1]
    scale_1: lm.f32[1]
    lm.smem_load_vec(
        dst=scale_0,
        src_addr=sfa_stage_addr + r_0 * 4,
        count=1,
    )
    lm.smem_load_vec(
        dst=scale_1,
        src_addr=sfa_stage_addr + r_1 * 4,
        count=1,
    )
    accum: lm.f32[64]
    accum.fill_(0.0)
    for accum_idx in lm.range(0, 64, constexpr=True):
        lm.wgmma_fence_operand(accum[accum_idx])
    for k_half in range(SS_K_HALVES):  # Python-time unroll: K origins must be trace-time ints
        for quad_in_half in lm.range(0, SS_QUADS_PER_HALF, constexpr=True):
            quad_i = k_half * SS_QUADS_PER_HALF + quad_in_half
            packed_quad: lm.u32[4]
            lm.smem_load_vec(
                dst=packed_quad,
                src_addr=packed_row_addr + quad_i * 16,
                count=4,
            )
            scale_word: lm.u32 = scale_words[0]
            if quad_i >= 2:
                scale_word = scale_words[1]
            scale_i0 = quad_i * 2
            scale0: lm.u32 = (
                scale_word >> ((scale_i0 & 3) * 8)
            ) & 0x7F
            scale1: lm.u32 = (
                scale_word >> (((scale_i0 + 1) & 3) * 8)
            ) & 0x7F
            lut0: lm.u32[2]
            lut1: lm.u32[2]
            lm.smem_load_vec(
                dst=lut0,
                src_addr=lut_smem.addr + scale0.i32() * 8,
                count=2,
            )
            lm.smem_load_vec(
                dst=lut1,
                src_addr=lut_smem.addr + scale1.i32() * 8,
                count=2,
            )

            q0_hi: lm.u32 = 0
            q0_lo: lm.u32 = 0
            q1_hi: lm.u32 = 0
            q1_lo: lm.u32 = 0
            q2_hi: lm.u32 = 0
            q2_lo: lm.u32 = 0
            q3_hi: lm.u32 = 0
            q3_lo: lm.u32 = 0
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=q0_hi,
                dst_lo=q0_lo,
                packed=packed_quad[0],
                lut_lo=lut0[0],
                lut_hi=lut0[1],
            )
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=q1_hi,
                dst_lo=q1_lo,
                packed=packed_quad[1],
                lut_lo=lut0[0],
                lut_hi=lut0[1],
            )
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=q2_hi,
                dst_lo=q2_lo,
                packed=packed_quad[2],
                lut_lo=lut1[0],
                lut_hi=lut1[1],
            )
            lm.nvfp4_mode2_lut_decode_word(
                dst_hi=q3_hi,
                dst_lo=q3_lo,
                packed=packed_quad[3],
                lut_lo=lut1[0],
                lut_hi=lut1[1],
            )
            decoded_smem.swizzled_vec_store(
                row=epilogue_thread_idx,
                col_bytes=scale_i0 * 16,
                values=(q0_hi, q0_lo, q1_hi, q1_lo),
                base_addr=decoded_stage_addr,
            )
            decoded_smem.swizzled_vec_store(
                row=epilogue_thread_idx,
                col_bytes=(scale_i0 + 1) * 16,
                values=(q2_hi, q2_lo, q3_hi, q3_lo),
                base_addr=decoded_stage_addr,
            )


        lm.fence_proxy_shared_cta()
        if lm.warp_id < 8:
            lm.sync(bar_id=3, threads=128, aligned=True)
        else:
            lm.sync(bar_id=4, threads=128, aligned=True)
        lm.wgmma_fence()
        for k_slice in range(SS_QUADS_PER_HALF):
            k_idx = k_half * SS_QUADS_PER_HALF + k_slice
            lm.mma(
                accum,
                act_stage.tile((0, k_idx * 32), (64, 32)),
                decoded_stage.tile(
                    (((lm.warp_id - 4) // 4) * 128, k_idx * 32),
                    (128, 32),
                ),
                init=(k_idx == 0),
            )
        lm.wgmma_commit_group()
    for accum_idx in lm.range(0, 64, constexpr=True):
        lm.wgmma_fence_operand(accum[accum_idx])
    lm.wgmma_wait_group(n=0)

    # Release the loader stage immediately after WGMMA consumption, before
    # the register-only scale accumulation, as in the source pipeline.
    with lm.elected_thread():
        lm.arrive(empty_barrier, stage=stage)
    for accum_chunk in lm.range(0, 16, constexpr=True):
        accum_offset = accum_chunk * 4
        final_accum[accum_offset + 0] = (
            final_accum[accum_offset + 0]
            + scale_0[0] * accum[accum_offset + 0]
        )
        final_accum[accum_offset + 1] = (
            final_accum[accum_offset + 1]
            + scale_0[0] * accum[accum_offset + 1]
        )
        final_accum[accum_offset + 2] = (
            final_accum[accum_offset + 2]
            + scale_1[0] * accum[accum_offset + 2]
        )
        final_accum[accum_offset + 3] = (
            final_accum[accum_offset + 3]
            + scale_1[0] * accum[accum_offset + 3]
        )


@loom.weave_fragment
def _emit_l1_nonswap_epilogue(
    lm,
    *,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l1_global_scales,
    l1_topk_weights,
    l2_sf,
    l2_arrival_mask,
    l1_output,
    l1_output_smem,
    l1_amax_smem,
    padded_sf_tokens,
):
    """Emit the source non-swap M64N128 L1 publication epilogue."""

    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    r_0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    r_1: lm.i32 = r_0 + 8
    wg_l1_out_n_idx: lm.i32 = epilogue_wg_idx * 64
    m_idx: lm.i32 = pool_block_idx * 64

    global_scale_vec = lm.vec_load(
        l1_global_scales,
        index=local_expert_idx,
        count=1,
        dtype=lm.f32,
        dst_dtype=lm.f32,
        cache_hint="read_only",
    )
    l1_global_scale: lm.f32 = global_scale_vec[0]
    weight_0: lm.f32 = 0.0
    weight_1: lm.f32 = 0.0
    if r_0 < valid_m:
        weight_vec_0 = lm.vec_load(
            l1_topk_weights,
            index=m_idx + r_0,
            count=1,
            dtype=lm.f32,
            dst_dtype=lm.f32,
        )
        weight_0 = weight_vec_0[0]
    if r_1 < valid_m:
        weight_vec_1 = lm.vec_load(
            l1_topk_weights,
            index=m_idx + r_1,
            count=1,
            dtype=lm.f32,
            dst_dtype=lm.f32,
        )
        weight_1 = weight_vec_1[0]

    swiglu_0: lm.f32[16]
    swiglu_1: lm.f32[16]
    swiglu_0.fill_(0.0)
    swiglu_1.fill_(0.0)
    amax_0: lm.f32 = 0.0
    amax_1: lm.f32 = 0.0
    for pair_idx in lm.range(0, 8, constexpr=True):
        gate_offset = pair_idx * 8
        up_offset = gate_offset + 4
        if r_0 < valid_m:
            gate_00: lm.f32 = lm.min(
                final_accum[gate_offset + 0] * l1_global_scale,
                10.0,
            )
            gate_01: lm.f32 = lm.min(
                final_accum[gate_offset + 1] * l1_global_scale,
                10.0,
            )
            up_00: lm.f32 = lm.max(
                lm.min(
                    final_accum[up_offset + 0] * l1_global_scale,
                    10.0,
                ),
                -10.0,
            )
            up_01: lm.f32 = lm.max(
                lm.min(
                    final_accum[up_offset + 1] * l1_global_scale,
                    10.0,
                ),
                -10.0,
            )
            value_00: lm.f32 = (
                gate_00 * lm.rcp(1.0 + lm.expf(-gate_00))
                * up_00
            )
            value_01: lm.f32 = (
                gate_01 * lm.rcp(1.0 + lm.expf(-gate_01))
                * up_01
            )
            swiglu_0[pair_idx * 2 + 0] = value_00
            swiglu_0[pair_idx * 2 + 1] = value_01
            amax_0 = lm.max(
                amax_0,
                lm.max(lm.fabs(value_00), lm.fabs(value_01)),
            )
        if r_1 < valid_m:
            gate_10: lm.f32 = lm.min(
                final_accum[gate_offset + 2] * l1_global_scale,
                10.0,
            )
            gate_11: lm.f32 = lm.min(
                final_accum[gate_offset + 3] * l1_global_scale,
                10.0,
            )
            up_10: lm.f32 = lm.max(
                lm.min(
                    final_accum[up_offset + 2] * l1_global_scale,
                    10.0,
                ),
                -10.0,
            )
            up_11: lm.f32 = lm.max(
                lm.min(
                    final_accum[up_offset + 3] * l1_global_scale,
                    10.0,
                ),
                -10.0,
            )
            value_10: lm.f32 = (
                gate_10 * lm.rcp(1.0 + lm.expf(-gate_10))
                * up_10
            )
            value_11: lm.f32 = (
                gate_11 * lm.rcp(1.0 + lm.expf(-gate_11))
                * up_11
            )
            swiglu_1[pair_idx * 2 + 0] = value_10
            swiglu_1[pair_idx * 2 + 1] = value_11
            amax_1 = lm.max(
                amax_1,
                lm.max(lm.fabs(value_10), lm.fabs(value_11)),
            )

    # Source order matters here: route weights are applied after the
    # unweighted SwiGLU amax, then the amax is scaled by abs(weight).
    for value_idx in lm.range(0, 16, constexpr=True):
        swiglu_0[value_idx] = swiglu_0[value_idx] * weight_0
        swiglu_1[value_idx] = swiglu_1[value_idx] * weight_1
    amax_0 = amax_0 * lm.fabs(weight_0)
    amax_1 = amax_1 * lm.fabs(weight_1)

    amax_0 = lm.max(amax_0, lm.shfl_xor_sync(amax_0, 1))
    amax_1 = lm.max(amax_1, lm.shfl_xor_sync(amax_1, 1))
    amax_0 = lm.max(amax_0, lm.shfl_xor_sync(amax_0, 2))
    amax_1 = lm.max(amax_1, lm.shfl_xor_sync(amax_1, 2))
    if col_idx == 0:
        l1_amax_smem[epilogue_wg_idx * 64 + r_0] = amax_0
        l1_amax_smem[epilogue_wg_idx * 64 + r_1] = amax_1

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )
    amax_0 = lm.max(l1_amax_smem[r_0], l1_amax_smem[64 + r_0])
    amax_1 = lm.max(l1_amax_smem[r_1], l1_amax_smem[64 + r_1])

    scaled_amax_0: lm.f32 = amax_0 * (1.0 / 448.0)
    scaled_amax_1: lm.f32 = amax_1 * (1.0 / 448.0)
    scaled_bits_0: lm.u32 = 0
    scaled_bits_1: lm.u32 = 0
    lm.bit_cast_store(target=scaled_bits_0, src=scaled_amax_0, cast_type=lm.u32)
    lm.bit_cast_store(target=scaled_bits_1, src=scaled_amax_1, cast_type=lm.u32)
    exponent_0: lm.i32 = (scaled_bits_0 >> 23).i32()
    exponent_1: lm.i32 = (scaled_bits_1 >> 23).i32()
    scale_exponent_0: lm.i32 = exponent_0 - 127
    scale_exponent_1: lm.i32 = exponent_1 - 127
    if (scaled_bits_0 & 0x7FFFFF) != 0:
        scale_exponent_0 = scale_exponent_0 + 1
    if (scaled_bits_1 & 0x7FFFFF) != 0:
        scale_exponent_1 = scale_exponent_1 + 1
    scale_bits_0: lm.u32 = (scale_exponent_0 + 127).u32() << 23
    scale_bits_1: lm.u32 = (scale_exponent_1 + 127).u32() << 23
    inv_bits_0: lm.u32 = (-scale_exponent_0 + 127).u32() << 23
    inv_bits_1: lm.u32 = (-scale_exponent_1 + 127).u32() << 23
    scale_0: lm.f32 = 0.0
    scale_1: lm.f32 = 0.0
    scale_inv_0: lm.f32 = 0.0
    scale_inv_1: lm.f32 = 0.0
    lm.bit_cast_store(target=scale_0, src=scale_bits_0, cast_type=lm.f32)
    lm.bit_cast_store(target=scale_1, src=scale_bits_1, cast_type=lm.f32)
    lm.bit_cast_store(target=scale_inv_0, src=inv_bits_0, cast_type=lm.f32)
    lm.bit_cast_store(target=scale_inv_1, src=inv_bits_1, cast_type=lm.f32)

    if lm.logical_and(epilogue_wg_idx == 0, col_idx == 0):
        sf_base: lm.i32 = n_block_idx * padded_sf_tokens + m_idx
        if r_0 < valid_m:
            lm.gmem_store(
                l2_sf,
                scale_0,
                index=sf_base + r_0,
                dtype=lm.f32,
                src_dtype=lm.f32,
            )
        if r_1 < valid_m:
            lm.gmem_store(
                l2_sf,
                scale_1,
                index=sf_base + r_1,
                dtype=lm.f32,
                src_dtype=lm.f32,
            )

    for pair_idx in lm.range(0, 8, constexpr=True):
        quant_values: lm.f32[4]
        quant_values[0] = swiglu_0[pair_idx * 2 + 0] * scale_inv_0
        quant_values[1] = swiglu_0[pair_idx * 2 + 1] * scale_inv_0
        quant_values[2] = swiglu_1[pair_idx * 2 + 0] * scale_inv_1
        quant_values[3] = swiglu_1[pair_idx * 2 + 1] * scale_inv_1
        packed_values = quant_values.fp8_e4m3()
        out_col: lm.i32 = wg_l1_out_n_idx + pair_idx * 8 + col_idx * 2
        if r_0 < valid_m:
            l1_output_smem[r_0 * 128 + out_col + 0] = packed_values[0] & 0xFF
            l1_output_smem[r_0 * 128 + out_col + 1] = (
                packed_values[0] >> 8
            ) & 0xFF
        if r_1 < valid_m:
            l1_output_smem[r_1 * 128 + out_col + 0] = (
                packed_values[0] >> 16
            ) & 0xFF
            l1_output_smem[r_1 * 128 + out_col + 1] = (
                packed_values[0] >> 24
            ) & 0xFF

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )
    with lm.elected_thread(warp=4):
        lm.fence_proxy_shared_cta()
        lm.tma_store(
            l1_output,
            coords=(n_block_idx * 128, m_idx),
            src=l1_output_smem.addr,
        )
        lm.bulk_commit()
    lm.sync_warp()
    lm.bulk_wait(n=0)
    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )
    with lm.elected_thread(warp=4):
        ready_bit: lm.u64 = 1
        ready_bit = ready_bit << n_block_idx
        lm.gmem_reduction_release(
            l2_arrival_mask,
            ready_bit,
            index=pool_block_idx,
            op=lm.REDUCE_OR,
            dtype=lm.u64,
            scope=lm.FENCE_DEVICE,
        )
    lm.sync_warp()


@loom.weave_fragment
def _emit_l1_swap_epilogue(
    lm,
    *,
    block_m,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l1_global_scales,
    l1_topk_weights,
    l2_sf,
    l2_arrival_mask,
    l1_output,
    l1_output_smem,
    l1_amax_smem,
    padded_sf_tokens,
):
    """Emit the source swap-AB L1 SwiGLU/FP8 publication epilogue.

    ``block_m`` is a trace-time schedule constant.  The register ownership,
    quad amax butterfly, eight-warp shared reduction, UE8M0 power-of-two scale,
    FP8 row-major staging, TMA store, and release-OR match
    ``fused_body.inl:1304-1450``.
    """

    token_chunks = block_m // 8
    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    epilogue_thread_idx: lm.i32 = epilogue_warp_idx * 32 + lm.lane_id
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    wg_l1_out_n_idx: lm.i32 = epilogue_wg_idx * 64
    m_idx: lm.i32 = pool_block_idx * block_m

    global_scale_vec = lm.vec_load(
        l1_global_scales,
        index=local_expert_idx,
        count=1,
        dtype=lm.f32,
        dst_dtype=lm.f32,
        cache_hint="read_only",
    )
    l1_global_scale: lm.f32 = global_scale_vec[0]
    swap_v0: lm.f32[2 * token_chunks]
    swap_v1: lm.f32[2 * token_chunks]
    swap_v0.fill_(0.0)
    swap_v1.fill_(0.0)

    for token_chunk in lm.range(0, token_chunks, constexpr=True):
        token_0: lm.i32 = token_chunk * 8 + col_idx * 2
        token_1: lm.i32 = token_0 + 1
        v0_amax: lm.f32 = 0.0
        v1_amax: lm.f32 = 0.0
        for half in lm.range(0, 2, constexpr=True):
            accum_offset = half * 32 + token_chunk * 4
            value_slot = half * token_chunks + token_chunk
            if token_0 < valid_m:
                gate_0: lm.f32 = lm.min(
                    final_accum[accum_offset + 0] * l1_global_scale,
                    10.0,
                )
                up_0: lm.f32 = lm.max(
                    lm.min(
                        final_accum[accum_offset + 2] * l1_global_scale,
                        10.0,
                    ),
                    -10.0,
                )
                weight_0_vec = lm.vec_load(
                    l1_topk_weights,
                    index=m_idx + token_0,
                    count=1,
                    dtype=lm.f32,
                    dst_dtype=lm.f32,
                )
                sigmoid_0: lm.f32 = lm.rcp(1.0 + lm.expf(-gate_0))
                value_0: lm.f32 = (
                    gate_0 * sigmoid_0 * up_0 * weight_0_vec[0]
                )
                swap_v0[value_slot] = value_0
                v0_amax = lm.max(v0_amax, lm.fabs(value_0))
            if token_1 < valid_m:
                gate_1: lm.f32 = lm.min(
                    final_accum[accum_offset + 1] * l1_global_scale,
                    10.0,
                )
                up_1: lm.f32 = lm.max(
                    lm.min(
                        final_accum[accum_offset + 3] * l1_global_scale,
                        10.0,
                    ),
                    -10.0,
                )
                weight_1_vec = lm.vec_load(
                    l1_topk_weights,
                    index=m_idx + token_1,
                    count=1,
                    dtype=lm.f32,
                    dst_dtype=lm.f32,
                )
                sigmoid_1: lm.f32 = lm.rcp(1.0 + lm.expf(-gate_1))
                value_1: lm.f32 = (
                    gate_1 * sigmoid_1 * up_1 * weight_1_vec[0]
                )
                swap_v1[value_slot] = value_1
                v1_amax = lm.max(v1_amax, lm.fabs(value_1))

        # math::warp_reduce<4, true> (kIntergroupReduce): reduce across the
        # eight row_idx lane groups with xor 4, 8, 16 while keeping col_idx
        # fixed, so each token pair's amax covers all eight weight rows this
        # warp owns.  (xor 1/2 would mix different tokens' values instead.)
        peer_0: lm.f32 = lm.shfl_xor_sync(v0_amax, 4)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1: lm.f32 = lm.shfl_xor_sync(v1_amax, 4)
        v1_amax = lm.max(v1_amax, peer_1)
        peer_0 = lm.shfl_xor_sync(v0_amax, 8)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1 = lm.shfl_xor_sync(v1_amax, 8)
        v1_amax = lm.max(v1_amax, peer_1)
        peer_0 = lm.shfl_xor_sync(v0_amax, 16)
        v0_amax = lm.max(v0_amax, peer_0)
        peer_1 = lm.shfl_xor_sync(v1_amax, 16)
        v1_amax = lm.max(v1_amax, peer_1)
        if row_idx == 0:
            if token_0 < valid_m:
                l1_amax_smem[
                    token_0 * EPILOGUE_WARPS + epilogue_warp_idx
                ] = v0_amax
            if token_1 < valid_m:
                l1_amax_smem[
                    token_1 * EPILOGUE_WARPS + epilogue_warp_idx
                ] = v1_amax

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )

    # valid_m <= 24 for every swap-AB row, so this is the source loop's one
    # reachable iteration per epilogue thread.
    if epilogue_thread_idx < valid_m:
        amax: lm.f32 = 0.0
        for source_warp in lm.range(0, EPILOGUE_WARPS, constexpr=True):
            amax = lm.max(
                amax,
                l1_amax_smem[
                    epilogue_thread_idx * EPILOGUE_WARPS + source_warp
                ],
            )

        # common/math.cuh::get_e4m3_sf_and_sf_inv: fast_log2_ceil followed
        # by an IEEE exponent construction for sf and its reciprocal.
        scaled_amax: lm.f32 = amax * (1.0 / 448.0)
        scaled_bits: lm.u32 = 0
        lm.bit_cast_store(
            target=scaled_bits,
            src=scaled_amax,
            cast_type=lm.u32,
        )
        exponent: lm.i32 = (scaled_bits >> 23).i32()
        mantissa: lm.u32 = scaled_bits & 0x7FFFFF
        scale_exponent: lm.i32 = exponent - 127
        if mantissa != 0:
            scale_exponent = scale_exponent + 1
        scale_bits: lm.u32 = (scale_exponent + 127).u32() << 23
        scale_inv_bits: lm.u32 = (-scale_exponent + 127).u32() << 23
        scale: lm.f32 = 0.0
        scale_inv: lm.f32 = 0.0
        lm.bit_cast_store(target=scale, src=scale_bits, cast_type=lm.f32)
        lm.bit_cast_store(
            target=scale_inv,
            src=scale_inv_bits,
            cast_type=lm.f32,
        )
        sf_index: lm.i32 = (
            n_block_idx * padded_sf_tokens
            + m_idx
            + epilogue_thread_idx
        )
        lm.gmem_store(
            l2_sf,
            scale,
            index=sf_index,
            dtype=lm.f32,
            src_dtype=lm.f32,
        )
        l1_amax_smem[
            epilogue_thread_idx * EPILOGUE_WARPS
        ] = scale_inv

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )

    for token_chunk in lm.range(0, token_chunks, constexpr=True):
        token_0: lm.i32 = token_chunk * 8 + col_idx * 2
        token_1: lm.i32 = token_0 + 1
        for half in lm.range(0, 2, constexpr=True):
            value_slot = half * token_chunks + token_chunk
            out_col: lm.i32 = (
                wg_l1_out_n_idx
                + half * 32
                + warp_idx_in_wg * 8
                + row_idx
            )
            quant_pair: lm.f32[4]
            quant_pair.fill_(0.0)
            if token_0 < valid_m:
                sf_inv_0: lm.f32 = l1_amax_smem[
                    token_0 * EPILOGUE_WARPS
                ]
                quant_pair[0] = swap_v0[value_slot] * sf_inv_0
            if token_1 < valid_m:
                sf_inv_1: lm.f32 = l1_amax_smem[
                    token_1 * EPILOGUE_WARPS
                ]
                quant_pair[1] = swap_v1[value_slot] * sf_inv_1
            packed_pair = quant_pair.fp8_e4m3()
            if token_0 < valid_m:
                l1_output_smem[token_0 * 128 + out_col] = (
                    packed_pair[0] & 0xFF
                )
            if token_1 < valid_m:
                l1_output_smem[token_1 * 128 + out_col] = (
                    (packed_pair[0] >> 8) & 0xFF
                )

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )
    with lm.elected_thread(warp=4):
        lm.fence_proxy_shared_cta()
        lm.tma_store(
            l1_output,
            coords=(n_block_idx * 128, m_idx),
            src=l1_output_smem.addr,
        )
        lm.bulk_commit()
    lm.sync_warp()
    lm.bulk_wait(n=0)
    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )
    with lm.elected_thread(warp=4):
        ready_bit: lm.u64 = 1
        ready_bit = ready_bit << n_block_idx
        lm.gmem_reduction_release(
            l2_arrival_mask,
            ready_bit,
            index=pool_block_idx,
            op=lm.REDUCE_OR,
            dtype=lm.u64,
            scope=lm.FENCE_DEVICE,
        )
    lm.sync_warp()


@loom.weave_fragment
def _emit_l2_swap_remote_scatter(
    lm,
    *,
    block_m,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l2_global_scales,
    token_src_metadata,
    sym_buffer,
    combine_token_offset,
    hidden,
    l2_output_smem,
):
    """Retain the source swap-AB L2 BF16 staging and remote scatter."""

    global_scale_vec = lm.vec_load(
        l2_global_scales,
        index=local_expert_idx,
        count=1,
        dtype=lm.f32,
        dst_dtype=lm.f32,
        cache_hint="read_only",
    )
    l2_global_scale: lm.f32 = global_scale_vec[0]
    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    r_0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    r_1: lm.i32 = r_0 + 8
    wg_n_idx: lm.i32 = epilogue_wg_idx * 128
    num_swap_token_chunks: lm.i32 = (valid_m + 7) // 8

    # BLOCK_N=256 and WG_BLOCK_N=128 for every retained small-M swap arm.
    # Each register holds one source M64N128 fragment; transpose its logical
    # rows into source token rows before the peer-mapped scatter.
    for token_chunk in lm.range(0, block_m // 8, constexpr=True):
        if token_chunk < num_swap_token_chunks:
            token_0: lm.i32 = token_chunk * 8 + col_idx * 2
            token_1: lm.i32 = token_0 + 1
            for half in lm.range(0, 2, constexpr=True):
                accum_offset: lm.i32 = half * 32 + token_chunk * 4
                col_offset: lm.i32 = half * 64
                if token_0 < valid_m:
                    l2_output_smem.swizzled_scalar_store(
                        row=token_0,
                        col_bytes=(wg_n_idx + col_offset + r_0) * 2,
                        value=(
                            final_accum[accum_offset + 0]
                            * l2_global_scale
                        ),
                        dtype=lm.bf16,
                        row_stride_bytes=512,
                    )
                    l2_output_smem.swizzled_scalar_store(
                        row=token_0,
                        col_bytes=(wg_n_idx + col_offset + r_1) * 2,
                        value=(
                            final_accum[accum_offset + 2]
                            * l2_global_scale
                        ),
                        dtype=lm.bf16,
                        row_stride_bytes=512,
                    )
                if token_1 < valid_m:
                    l2_output_smem.swizzled_scalar_store(
                        row=token_1,
                        col_bytes=(wg_n_idx + col_offset + r_0) * 2,
                        value=(
                            final_accum[accum_offset + 1]
                            * l2_global_scale
                        ),
                        dtype=lm.bf16,
                        row_stride_bytes=512,
                    )
                    l2_output_smem.swizzled_scalar_store(
                        row=token_1,
                        col_bytes=(wg_n_idx + col_offset + r_1) * 2,
                        value=(
                            final_accum[accum_offset + 3]
                            * l2_global_scale
                        ),
                        dtype=lm.bf16,
                        row_stride_bytes=512,
                    )

    if lm.warp_id < 8:
        lm.sync(bar_id=3, threads=128, aligned=True)
    else:
        lm.sync(bar_id=4, threads=128, aligned=True)

    row_in_warp_block: lm.i32 = lm.lane_id // 16
    lane_in_row: lm.i32 = lm.lane_id % 16
    rows_per_warp = 4 if block_m == 8 else 8
    n_idx: lm.i32 = n_block_idx * 256 + wg_n_idx
    for row_iter in lm.range(0, rows_per_warp, constexpr=True):
        token: lm.i32 = (
            warp_idx_in_wg * 16 + row_iter * 2 + row_in_warp_block
        )
        if token >= valid_m:
            break

        metadata_base: lm.i32 = (
            pool_block_idx * block_m + token
        ) * 3
        dst_rank_word = lm.vec_load(
            token_src_metadata,
            index=metadata_base,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_token_word = lm.vec_load(
            token_src_metadata,
            index=metadata_base + 1,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_topk_word = lm.vec_load(
            token_src_metadata,
            index=metadata_base + 2,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_rank: lm.i32 = dst_rank_word[0]
        dst_token: lm.i32 = dst_token_word[0]
        dst_topk: lm.i32 = dst_topk_word[0]

        packed: lm.u32[4]
        lm.smem_load_vec(
            dst=packed,
            src_addr=(
                l2_output_smem.addr
                + (
                    token * 256
                    + wg_n_idx
                    + lane_in_row * 8
                )
                * 2
            ),
            count=4,
        )
        peer_combine_u32 = lm.ptr_to(
            lm.byte_ptr(sym_buffer.peer_ptr(dst_rank))
            + combine_token_offset,
            lm.u32,
        )
        dst_word_offset: lm.u64 = (
            (
                (dst_topk.u64() * MAX_TOKENS_PER_RANK + dst_token.u64())
                * hidden
                + n_idx.u64()
                + lane_in_row.u64() * 8
            )
            // 2
        )
        lm.gmem_store_vec(
            dst=peer_combine_u32 + dst_word_offset,
            src=packed,
        )

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )


@loom.weave_fragment
def _emit_l2_nonswap_remote_scatter(
    lm,
    *,
    final_accum,
    valid_m,
    local_expert_idx,
    n_block_idx,
    pool_block_idx,
    l2_global_scales,
    token_src_metadata,
    sym_buffer,
    combine_token_offset,
    hidden,
):
    """Emit the source non-swap direct BF16 symmetric remote scatter.

    Each M64N128 math warpgroup owns one contiguous N128 range.  Its lanes
    directly pack two adjacent BF16 values at a time from the register
    accumulator and publish those 32-bit words to the rank/token/top-k
    destination selected by the transported token metadata.  Unlike the
    swap-AB path, this source branch does not stage a BLOCK_M-by-256 CD tile.
    """

    global_scale_vec = lm.vec_load(
        l2_global_scales,
        index=local_expert_idx,
        count=1,
        dtype=lm.f32,
        dst_dtype=lm.f32,
        cache_hint="read_only",
    )
    l2_global_scale: lm.f32 = global_scale_vec[0]
    epilogue_warp_idx: lm.i32 = lm.warp_id - 4
    epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    row_offset_r0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    row_offset_r1: lm.i32 = row_offset_r0 + 8
    n_idx: lm.i32 = n_block_idx * 256 + epilogue_wg_idx * 128

    if row_offset_r0 < valid_m:
        metadata_base_r0: lm.i32 = (
            pool_block_idx * 64 + row_offset_r0
        ) * 3
        dst_rank_word_r0 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r0,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_token_word_r0 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r0 + 1,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_topk_word_r0 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r0 + 2,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        peer_combine_i32_r0 = lm.ptr_to(
            lm.byte_ptr(
                sym_buffer.peer_ptr(dst_rank_word_r0[0])
            )
            + combine_token_offset,
            lm.i32,
        )
        dst_word_base_r0: lm.u64 = (
            (
                (
                    dst_topk_word_r0[0].u64() * MAX_TOKENS_PER_RANK
                    + dst_token_word_r0[0].u64()
                )
                * hidden
                + n_idx.u64()
            )
            // 2
        )
        for pair_idx in lm.range(0, 8, constexpr=True):
            chunk_lo = pair_idx * 2
            chunk_hi = chunk_lo + 1
            col_lo = chunk_lo * 8 + col_idx * 2
            col_hi = chunk_hi * 8 + col_idx * 2
            pair_values_r0_lo: lm.f32[2]
            pair_values_r0_lo[0] = (
                final_accum[chunk_lo * 4 + 0] * l2_global_scale
            )
            pair_values_r0_lo[1] = (
                final_accum[chunk_lo * 4 + 1] * l2_global_scale
            )
            pair_values_r0_hi: lm.f32[2]
            pair_values_r0_hi[0] = (
                final_accum[chunk_hi * 4 + 0] * l2_global_scale
            )
            pair_values_r0_hi[1] = (
                final_accum[chunk_hi * 4 + 1] * l2_global_scale
            )
            packed_r0_lo: lm.u32[1]
            packed_r0_hi: lm.u32[1]
            lm.reg_array_pack(
                pair_values_r0_lo,
                packed_r0_lo,
                size=2,
                dtype=lm.bf16,
            )
            lm.reg_array_pack(
                pair_values_r0_hi,
                packed_r0_hi,
                size=2,
                dtype=lm.bf16,
            )
            lm.gmem_store(
                peer_combine_i32_r0,
                packed_r0_lo[0].i32(),
                index=dst_word_base_r0 + col_lo // 2,
                dtype=lm.i32,
                src_dtype=lm.i32,
            )
            lm.gmem_store(
                peer_combine_i32_r0,
                packed_r0_hi[0].i32(),
                index=dst_word_base_r0 + col_hi // 2,
                dtype=lm.i32,
                src_dtype=lm.i32,
            )

    if row_offset_r1 < valid_m:
        metadata_base_r1: lm.i32 = (
            pool_block_idx * 64 + row_offset_r1
        ) * 3
        dst_rank_word_r1 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r1,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_token_word_r1 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r1 + 1,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        dst_topk_word_r1 = lm.vec_load(
            token_src_metadata,
            index=metadata_base_r1 + 2,
            count=1,
            dtype=lm.i32,
            dst_dtype=lm.i32,
        )
        peer_combine_i32_r1 = lm.ptr_to(
            lm.byte_ptr(
                sym_buffer.peer_ptr(dst_rank_word_r1[0])
            )
            + combine_token_offset,
            lm.i32,
        )
        dst_word_base_r1: lm.u64 = (
            (
                (
                    dst_topk_word_r1[0].u64() * MAX_TOKENS_PER_RANK
                    + dst_token_word_r1[0].u64()
                )
                * hidden
                + n_idx.u64()
            )
            // 2
        )
        for pair_idx in lm.range(0, 8, constexpr=True):
            chunk_lo = pair_idx * 2
            chunk_hi = chunk_lo + 1
            col_lo = chunk_lo * 8 + col_idx * 2
            col_hi = chunk_hi * 8 + col_idx * 2
            pair_values_r1_lo: lm.f32[2]
            pair_values_r1_lo[0] = (
                final_accum[chunk_lo * 4 + 2] * l2_global_scale
            )
            pair_values_r1_lo[1] = (
                final_accum[chunk_lo * 4 + 3] * l2_global_scale
            )
            pair_values_r1_hi: lm.f32[2]
            pair_values_r1_hi[0] = (
                final_accum[chunk_hi * 4 + 2] * l2_global_scale
            )
            pair_values_r1_hi[1] = (
                final_accum[chunk_hi * 4 + 3] * l2_global_scale
            )
            packed_r1_lo: lm.u32[1]
            packed_r1_hi: lm.u32[1]
            lm.reg_array_pack(
                pair_values_r1_lo,
                packed_r1_lo,
                size=2,
                dtype=lm.bf16,
            )
            lm.reg_array_pack(
                pair_values_r1_hi,
                packed_r1_hi,
                size=2,
                dtype=lm.bf16,
            )
            lm.gmem_store(
                peer_combine_i32_r1,
                packed_r1_lo[0].i32(),
                index=dst_word_base_r1 + col_lo // 2,
                dtype=lm.i32,
                src_dtype=lm.i32,
            )
            lm.gmem_store(
                peer_combine_i32_r1,
                packed_r1_hi[0].i32(),
                index=dst_word_base_r1 + col_hi // 2,
                dtype=lm.i32,
                src_dtype=lm.i32,
            )

    lm.sync(
        bar_id=EPILOGUE_FULL_BARRIER_ID,
        threads=EPILOGUE_THREADS,
        aligned=True,
    )


def _partial_l1_tma_kernel(
    *,
    block_m: int,
    stages: int,
    loader_registers: int,
):
    """Build the source L1 TMA full/empty stage protocol.

    This fragment retains fused_body.inl:717-797 as a parent-shaped, inactive
    compile target.  It deliberately stops at the consumer boundary: the
    retained RS mathematical slice is still separate, so this is not an
    active or correctness-complete provider path.
    """

    if block_m not in (8, 16, 24, 64):
        raise ValueError(f"unsupported source BLOCK_M={block_m}")
    if stages not in (3, 4, 6, 8):
        raise ValueError(f"unsupported source stage count {stages}")
    if loader_registers not in (40, 64):
        raise ValueError(f"unsupported loader register budget {loader_registers}")

    act_stage_bytes = block_m * RS_ACT_ROW_BYTES
    sfa_tile_bytes = block_m * 4
    sfa_stage_bytes = _align(sfa_tile_bytes, 128)
    packed_stage_bytes = RS_TMA_PACKED_B_BYTES
    act_offset = 0
    sfa_offset = act_offset + stages * act_stage_bytes
    packed_offset = sfa_offset + stages * sfa_stage_bytes

    def small_rs_l1_tma_fragment(
        lm,
        activation,
        activation_scale,
        packed_weights,
        l1_arrival_count,
        pool_block_idx,
        local_expert_idx,
        valid_m,
        num_k_blocks,
        compile_sentinel,
    ):
        # Preserve the exact parent launch envelope. Loom's 1024-byte barrier
        # control page is included above this user-visible pool.
        pool = lm.smem(SMEM_BYTES - LOOM_CONTROL_BYTES)
        act_smem = pool.view(
            offset=act_offset,
            shape=(block_m, RS_ACT_ROW_BYTES),
            dtype=lm.u8,
            logical_shape=(block_m, RS_BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=stages,
            stride=act_stage_bytes,
        )
        sfa_smem = pool.view(
            offset=sfa_offset,
            shape=(block_m,),
            dtype=lm.f32,
            swizzle=None,
            stage=stages,
            stride=sfa_stage_bytes,
        )
        packed_smem = pool.view(
            offset=packed_offset,
            shape=(RS_TMA_PACKED_B_BYTES,),
            dtype=lm.u8,
            swizzle=None,
            stage=stages,
            stride=packed_stage_bytes,
        )

        dispatch = lm.role("dispatch", warps=[0, 1])
        loader = lm.role("loader", warps=[2, 3])
        math = lm.role("math", warps=[4, 5, 6, 7, 8, 9, 10, 11])
        l1_pipe = lm.pipeline(stages=stages)
        stage_full = lm.barrier(
            count=stages,
            prod=[loader],
            cons=[math],
            init_count=2,
            pipeline=l1_pipe,
        )
        stage_empty = lm.barrier(
            count=stages,
            prod=[math],
            cons=[loader],
            init_count=8,
            init_phase=1,
            pipeline=l1_pipe,
        )

        with dispatch:
            lm.setmaxnreg_dealloc(48)
            with lm.elected_thread(warp=0):
                lm.gmem_store(
                    compile_sentinel,
                    48,
                    dtype=lm.i32,
                    src_dtype=lm.i32,
                )
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )

        with loader:
            lm.setmaxnreg_dealloc(loader_registers)
            load_stage: lm.u32 = 0

            if lm.warp_id == 2:
                if valid_m > 0:
                    expected_arrivals: lm.u32 = valid_m.u32()
                    lm.gmem_wait_acquire(
                        l1_arrival_count,
                        expected_arrivals,
                        index=pool_block_idx,
                        dtype=lm.u32,
                        predicate=lm.WAIT_EQ,
                        scope=lm.FENCE_DEVICE,
                    )

            for k_block_idx in lm.range(
                0,
                num_k_blocks,
                unroll=1,
                dtype=lm.i32,
            ):
                lm.wait(stage_empty, stage=load_stage)
                if lm.warp_id == 2:
                    if valid_m > 0:
                        with lm.elected_thread():
                            lm.tma_load(
                                activation,
                                dst=act_smem.stage_addr(load_stage),
                                coords=(
                                    k_block_idx * RS_BLOCK_K,
                                    pool_block_idx * block_m,
                                ),
                                barrier=stage_full,
                                stage=load_stage,
                            )
                            lm.tma_load(
                                activation_scale,
                                dst=sfa_smem.stage_addr(load_stage),
                                coords=(
                                    pool_block_idx * block_m,
                                    k_block_idx,
                                ),
                                barrier=stage_full,
                                stage=load_stage,
                            )
                            lm.arrive_expect_tx(
                                stage_full,
                                tx_bytes=act_stage_bytes + sfa_tile_bytes,
                                stage=load_stage,
                            )
                    else:
                        with lm.elected_thread():
                            lm.arrive(stage_full, stage=load_stage)
                else:
                    with lm.elected_thread():
                        lm.tma_load(
                            packed_weights,
                            dst=packed_smem.stage_addr(load_stage),
                            coords=(
                                k_block_idx * RS_WEIGHT_ROW_BYTES,
                                local_expert_idx * RS_WEIGHT_ROWS,
                            ),
                            barrier=stage_full,
                            stage=load_stage,
                        )
                        lm.arrive_expect_tx(
                            stage_full,
                            tx_bytes=packed_stage_bytes,
                            stage=load_stage,
                        )
                lm.sync_warp()
                lm.advance(load_stage, l1_pipe)

        with math:
            lm.setmaxnreg_alloc(208)
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )
            math_stage: lm.u32 = 0
            stage_checksum: lm.u32 = 0
            for _k_block_idx in lm.range(
                0,
                num_k_blocks,
                unroll=1,
                dtype=lm.i32,
            ):
                lm.wait(stage_full, stage=math_stage)
                lm.fence_proxy_shared_cta()

                # Materialize one typed read from each transaction so the
                # compile target preserves all three producer sites through
                # ptxas without pretending to execute the RS mathematics.
                packed_word: lm.u32[1]
                lm.smem_load_vec(
                    dst=packed_word,
                    src_addr=packed_smem.stage_addr(math_stage),
                    count=1,
                )
                stage_checksum = stage_checksum ^ packed_word[0]
                if valid_m > 0:
                    act_word: lm.u32[1]
                    sfa_word: lm.u32[1]
                    lm.smem_load_vec(
                        dst=act_word,
                        src_addr=act_smem.stage_addr(math_stage),
                        count=1,
                    )
                    lm.smem_load_vec(
                        dst=sfa_word,
                        src_addr=sfa_smem.stage_addr(math_stage),
                        count=1,
                    )
                    stage_checksum = (
                        stage_checksum ^ act_word[0] ^ sfa_word[0]
                    )

                with lm.elected_thread():
                    lm.arrive(stage_empty, stage=math_stage)
                lm.advance(math_stage, l1_pipe)

            with lm.elected_thread(warp=4):
                lm.gmem_store(
                    compile_sentinel + 1,
                    stage_checksum.i32(),
                    dtype=lm.i32,
                    src_dtype=lm.i32,
                )

    # Postponed annotations cannot resolve factory locals. Attach the typed
    # descriptors before applying @weave so every static BLOCK_M/stage arm has
    # the exact source transaction box rather than a max-shape surrogate.
    small_rs_l1_tma_fragment.__annotations__ = {
        "activation": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_BLOCK_K, block_m),
            swizzle=LM.swizzle_128b,
        ),
        "activation_scale": LM.tma2d(
            dtype=LM.f32,
            box_shape=(block_m, 1),
            swizzle=LM.swizzle_none,
        ),
        "packed_weights": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_WEIGHT_ROW_BYTES, RS_WEIGHT_ROWS),
            swizzle=LM.swizzle_none,
        ),
        "l1_arrival_count": LM.ptr[LM.u32],
        "pool_block_idx": LM.i32,
        "local_expert_idx": LM.i32,
        "valid_m": LM.i32,
        "num_k_blocks": LM.i32,
        "compile_sentinel": LM.ptr[LM.i32],
    }
    return loom.weave(
        threads=THREADS,
        cluster_dims=CLUSTER_DIMS,
    )(small_rs_l1_tma_fragment)


def _partial_l1_tma_rs_kernel(
    *,
    block_m: int,
    stages: int,
    loader_registers: int,
    interleaved_scheduler: bool,
):
    """Build the integrated source L1 TMA-to-RS consumer pipeline.

    Only the source ``swap_ab`` BLOCK_M=8/16/24 arms enter this family.
    Runtime ``valid_m`` selects one of the static N8/N16/N24 register-source
    WGMMA shapes while both loader warps and all eight math warps reuse the
    exact full/empty stage protocol.  The interleaved template axis
    instantiates the two-stage TaskInfo mailbox; the delivered current wrapper
    selects that axis for every row even when the retained request bit is
    false.
    """

    if block_m not in (8, 16, 24):
        raise ValueError(f"unsupported source swap_ab BLOCK_M={block_m}")
    if stages not in (3, 4, 6, 8):
        raise ValueError(f"unsupported source stage count {stages}")
    if loader_registers not in (40, 64):
        raise ValueError(f"unsupported loader register budget {loader_registers}")

    act_stage_bytes = block_m * RS_ACT_ROW_BYTES
    sfa_tile_bytes = block_m * 4
    sfa_stage_bytes = _align(sfa_tile_bytes, 128)
    packed_stage_bytes = RS_TMA_PACKED_B_BYTES
    act_offset = 0
    sfa_offset = act_offset + stages * act_stage_bytes
    packed_offset = sfa_offset + stages * sfa_stage_bytes
    lut_offset = packed_offset + stages * packed_stage_bytes
    task_info_offset = lut_offset + RS_LUT_BYTES
    is_block_m8 = block_m == 8
    is_block_m16 = block_m == 16

    def small_rs_l1_tma_rs_fragment(
        lm,
        activation,
        activation_scale,
        packed_weights,
        l1_arrival_count,
        block_phase,
        pool_block_idx,
        local_expert_idx,
        m_block_idx,
        n_block_idx,
        valid_m,
        shape_n,
        shape_k,
        output,
        compile_sentinel,
    ):
        lut = lm.constant_array(
            "kE2M1AndUe4m3ToFp8Lut",
            dtype="u32",
            values=RS_LUT_WORD_VALUES,
            align=16,
        )
        pool = lm.smem(SMEM_BYTES - LOOM_CONTROL_BYTES)
        act_smem = pool.view(
            offset=act_offset,
            shape=(block_m, RS_ACT_ROW_BYTES),
            dtype=lm.u8,
            logical_shape=(block_m, RS_BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=stages,
            stride=act_stage_bytes,
        )
        sfa_smem = pool.view(
            offset=sfa_offset,
            shape=(block_m,),
            dtype=lm.f32,
            swizzle=None,
            stage=stages,
            stride=sfa_stage_bytes,
        )
        packed_smem = pool.view(
            offset=packed_offset,
            shape=(RS_TMA_PACKED_B_BYTES,),
            dtype=lm.u8,
            swizzle=None,
            stage=stages,
            stride=packed_stage_bytes,
        )
        lut_smem = pool.view(
            offset=lut_offset,
            shape=(RS_LUT_WORDS,),
            dtype=lm.u32,
            swizzle=None,
        )
        if interleaved_scheduler:
            task_info_smem = pool.view(
                offset=task_info_offset,
                shape=(TASK_INFO_FIELDS,),
                dtype=lm.u32,
                swizzle=None,
                stage=TASK_INFO_STAGES,
                stride=TASK_INFO_BYTES,
            )

        dispatch = lm.role("dispatch", warps=[0, 1])
        loader = lm.role("loader", warps=[2, 3])
        math = lm.role("math", warps=[4, 5, 6, 7, 8, 9, 10, 11])
        l1_pipe = lm.pipeline(stages=stages)
        stage_full = lm.barrier(
            count=stages,
            prod=[loader],
            cons=[math],
            init_count=2,
            pipeline=l1_pipe,
        )
        stage_empty = lm.barrier(
            count=stages,
            prod=[math],
            cons=[loader],
            init_count=8,
            init_phase=1,
            pipeline=l1_pipe,
        )
        if interleaved_scheduler:
            task_info_pipe = lm.pipeline(stages=TASK_INFO_STAGES)
            task_info_full = lm.barrier(
                count=TASK_INFO_STAGES,
                prod=[loader],
                cons=[loader, math],
                init_count=1,
                pipeline=task_info_pipe,
            )
            task_info_empty = lm.barrier(
                count=TASK_INFO_STAGES,
                prod=[math],
                cons=[loader],
                init_count=8,
                init_phase=1,
                pipeline=task_info_pipe,
            )

        with dispatch:
            lm.setmaxnreg_dealloc(48)
            # Source shape: threads 0..63 each copy one uint4 from the typed
            # device-constant table into the shared Mode2 LUT.
            lm.constant_array_copy16b(
                lut,
                dst_addr=lut_smem.addr + lm.tid * 16,
                chunk=lm.tid,
            )
            lm.fence_shared()
            with lm.elected_thread(warp=0):
                lm.gmem_store(
                    compile_sentinel,
                    48,
                    dtype=lm.i32,
                    src_dtype=lm.i32,
                )
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )

        with loader:
            lm.setmaxnreg_dealloc(loader_registers)
            load_stage: lm.u32 = 0
            task_block_phase: lm.i32 = block_phase
            task_pool_block: lm.i32 = pool_block_idx
            task_local_expert: lm.i32 = local_expert_idx
            task_m_block: lm.i32 = m_block_idx
            task_n_block: lm.i32 = n_block_idx
            task_valid_m: lm.i32 = valid_m
            task_shape_n: lm.i32 = shape_n
            task_shape_k: lm.i32 = shape_k

            if interleaved_scheduler:
                producer_task_stage: lm.u32 = 0
                loader_task_stage: lm.u32 = 0
                if lm.warp_id == 3:
                    # Source producer cursor: wait for the stage's previous
                    # generation, publish exactly one 32-byte TaskInfo, fence
                    # its shared stores at CTA scope, then release full.
                    lm.wait(task_info_empty, stage=producer_task_stage)
                    with lm.elected_thread():
                        task_info_words: lm.u32[8]
                        task_info_words[0] = block_phase.u32()
                        task_info_words[1] = local_expert_idx.u32()
                        task_info_words[2] = m_block_idx.u32()
                        task_info_words[3] = n_block_idx.u32()
                        task_info_words[4] = pool_block_idx.u32()
                        task_info_words[5] = valid_m.u32()
                        task_info_words[6] = shape_n.u32()
                        task_info_words[7] = shape_k.u32()
                        for task_word in lm.range(
                            0,
                            TASK_INFO_FIELDS,
                            constexpr=True,
                        ):
                            lm.smem_store_vec(
                                dst_addr=(
                                    task_info_smem.stage_addr(
                                        producer_task_stage
                                    )
                                    + task_word * 4
                                ),
                                src=task_info_words[task_word],
                            )
                        lm.threadfence(scope=lm.FENCE_BLOCK)
                        lm.arrive(
                            task_info_full,
                            stage=producer_task_stage,
                        )
                    lm.sync_warp()
                    lm.advance(producer_task_stage, task_info_pipe)
                else:
                    # Activation loader owns an independent mailbox cursor.
                    lm.wait(task_info_full, stage=loader_task_stage)
                    loader_task_words: lm.u32[8]
                    lm.smem_load_vec(
                        dst=loader_task_words,
                        src_addr=task_info_smem.stage_addr(loader_task_stage),
                        count=4,
                    )
                    lm.smem_load_vec(
                        dst=loader_task_words,
                        src_addr=(
                            task_info_smem.stage_addr(loader_task_stage) + 16
                        ),
                        count=4,
                        dst_offset=4,
                    )
                    task_block_phase = loader_task_words[0].i32()
                    task_local_expert = loader_task_words[1].i32()
                    task_m_block = loader_task_words[2].i32()
                    task_n_block = loader_task_words[3].i32()
                    task_pool_block = loader_task_words[4].i32()
                    task_valid_m = loader_task_words[5].i32()
                    task_shape_n = loader_task_words[6].i32()
                    task_shape_k = loader_task_words[7].i32()
                    lm.advance(loader_task_stage, task_info_pipe)

            if lm.warp_id == 2:
                if task_valid_m > 0:
                    expected_arrivals: lm.u32 = task_valid_m.u32()
                    lm.gmem_wait_acquire(
                        l1_arrival_count,
                        expected_arrivals,
                        index=task_pool_block,
                        dtype=lm.u32,
                        predicate=lm.WAIT_EQ,
                        scope=lm.FENCE_DEVICE,
                    )

            for k_block_idx in lm.range(
                0,
                task_shape_k // RS_BLOCK_K,
                unroll=1,
                dtype=lm.i32,
            ):
                lm.wait(stage_empty, stage=load_stage)
                if lm.warp_id == 2:
                    if task_valid_m > 0:
                        with lm.elected_thread():
                            lm.tma_load(
                                activation,
                                dst=act_smem.stage_addr(load_stage),
                                coords=(
                                    k_block_idx * RS_BLOCK_K,
                                    task_pool_block * block_m,
                                ),
                                barrier=stage_full,
                                stage=load_stage,
                            )
                            lm.tma_load(
                                activation_scale,
                                dst=sfa_smem.stage_addr(load_stage),
                                coords=(
                                    task_pool_block * block_m,
                                    k_block_idx,
                                ),
                                barrier=stage_full,
                                stage=load_stage,
                            )
                            lm.arrive_expect_tx(
                                stage_full,
                                tx_bytes=(
                                    act_stage_bytes + sfa_tile_bytes
                                ),
                                stage=load_stage,
                            )
                    else:
                        with lm.elected_thread():
                            lm.arrive(stage_full, stage=load_stage)
                else:
                    with lm.elected_thread():
                        lm.tma_load(
                            packed_weights,
                            dst=packed_smem.stage_addr(load_stage),
                            coords=(
                                k_block_idx * RS_WEIGHT_ROW_BYTES,
                                task_local_expert * task_shape_n
                                + task_n_block * RS_WEIGHT_ROWS,
                            ),
                            barrier=stage_full,
                            stage=load_stage,
                        )
                        lm.arrive_expect_tx(
                            stage_full,
                            tx_bytes=packed_stage_bytes,
                            stage=load_stage,
                        )
                lm.sync_warp()
                lm.advance(load_stage, l1_pipe)

        with math:
            lm.setmaxnreg_alloc(208)
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )
            math_stage: lm.u32 = 0
            math_valid_m: lm.i32 = valid_m
            math_shape_k: lm.i32 = shape_k
            if interleaved_scheduler:
                math_task_stage: lm.u32 = 0
                lm.wait(task_info_full, stage=math_task_stage)
                consumed_task_stage: lm.u32 = math_task_stage
                math_task_words: lm.u32[8]
                lm.smem_load_vec(
                    dst=math_task_words,
                    src_addr=task_info_smem.stage_addr(math_task_stage),
                    count=4,
                )
                lm.smem_load_vec(
                    dst=math_task_words,
                    src_addr=task_info_smem.stage_addr(math_task_stage) + 16,
                    count=4,
                    dst_offset=4,
                )
                math_valid_m = math_task_words[5].i32()
                math_shape_k = math_task_words[7].i32()
                lm.advance(math_task_stage, task_info_pipe)
            final_accum: lm.f32[64]
            final_accum.fill_(0.0)

            for math_k_block_idx in lm.range(
                0,
                math_shape_k // RS_BLOCK_K,
                unroll=1,
                dtype=lm.i32,
            ):
                lm.wait(stage_full, stage=math_stage)
                if interleaved_scheduler:
                    if math_k_block_idx == 0:
                        # Eight math warps release the consumed mailbox slot
                        # only after the first TMA stage is visible, matching
                        # the source boundary at fused_body.inl:984-990.
                        with lm.elected_thread():
                            lm.arrive(
                                task_info_empty,
                                stage=consumed_task_stage,
                            )

                if is_block_m8:
                    _emit_l1_rs_stage(
                        lm,
                        n_swap=8,
                        act_stage=act_smem[math_stage],
                        packed_stage_addr=(
                            packed_smem.stage_addr(math_stage)
                        ),
                        lut_smem=lut_smem,
                        sfa_stage_addr=sfa_smem.stage_addr(math_stage),
                        valid_m=math_valid_m,
                        final_accum=final_accum,
                    )
                elif is_block_m16:
                    n_swap: lm.i32 = ((math_valid_m + 7) // 8) * 8
                    if n_swap <= 8:
                        _emit_l1_rs_stage(
                            lm,
                            n_swap=8,
                            act_stage=act_smem[math_stage],
                            packed_stage_addr=(
                                packed_smem.stage_addr(math_stage)
                            ),
                            lut_smem=lut_smem,
                            sfa_stage_addr=(
                                sfa_smem.stage_addr(math_stage)
                            ),
                            valid_m=math_valid_m,
                            final_accum=final_accum,
                        )
                    else:
                        _emit_l1_rs_stage(
                            lm,
                            n_swap=16,
                            act_stage=act_smem[math_stage],
                            packed_stage_addr=(
                                packed_smem.stage_addr(math_stage)
                            ),
                            lut_smem=lut_smem,
                            sfa_stage_addr=(
                                sfa_smem.stage_addr(math_stage)
                            ),
                            valid_m=math_valid_m,
                            final_accum=final_accum,
                        )
                else:
                    n_swap: lm.i32 = ((math_valid_m + 7) // 8) * 8
                    if n_swap <= 8:
                        _emit_l1_rs_stage(
                            lm,
                            n_swap=8,
                            act_stage=act_smem[math_stage],
                            packed_stage_addr=(
                                packed_smem.stage_addr(math_stage)
                            ),
                            lut_smem=lut_smem,
                            sfa_stage_addr=(
                                sfa_smem.stage_addr(math_stage)
                            ),
                            valid_m=math_valid_m,
                            final_accum=final_accum,
                        )
                    elif n_swap <= 16:
                        _emit_l1_rs_stage(
                            lm,
                            n_swap=16,
                            act_stage=act_smem[math_stage],
                            packed_stage_addr=(
                                packed_smem.stage_addr(math_stage)
                            ),
                            lut_smem=lut_smem,
                            sfa_stage_addr=(
                                sfa_smem.stage_addr(math_stage)
                            ),
                            valid_m=math_valid_m,
                            final_accum=final_accum,
                        )
                    else:
                        _emit_l1_rs_stage(
                            lm,
                            n_swap=24,
                            act_stage=act_smem[math_stage],
                            packed_stage_addr=(
                                packed_smem.stage_addr(math_stage)
                            ),
                            lut_smem=lut_smem,
                            sfa_stage_addr=(
                                sfa_smem.stage_addr(math_stage)
                            ),
                            valid_m=math_valid_m,
                            final_accum=final_accum,
                        )

                # The source's per-warpgroup dynamic barrier IDs become the
                # identical static instructions in these two role branches.
                if lm.warp_id < 8:
                    lm.sync(bar_id=3, threads=128, aligned=True)
                else:
                    lm.sync(bar_id=4, threads=128, aligned=True)
                with lm.elected_thread():
                    lm.arrive(stage_empty, stage=math_stage)
                lm.advance(math_stage, l1_pipe)

            math_tid: lm.i32 = (lm.warp_id - 4) * 32 + lm.lane_id
            for out_idx in lm.range(0, 64, constexpr=True):
                output[math_tid * 64 + out_idx] = final_accum[out_idx]

    small_rs_l1_tma_rs_fragment.__annotations__ = {
        "activation": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_BLOCK_K, block_m),
            swizzle=LM.swizzle_128b,
        ),
        "activation_scale": LM.tma2d(
            dtype=LM.f32,
            box_shape=(block_m, 1),
            swizzle=LM.swizzle_none,
        ),
        "packed_weights": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_WEIGHT_ROW_BYTES, RS_WEIGHT_ROWS),
            swizzle=LM.swizzle_none,
        ),
        "l1_arrival_count": LM.ptr[LM.u32],
        "block_phase": LM.i32,
        "pool_block_idx": LM.i32,
        "local_expert_idx": LM.i32,
        "m_block_idx": LM.i32,
        "n_block_idx": LM.i32,
        "valid_m": LM.i32,
        "shape_n": LM.i32,
        "shape_k": LM.i32,
        "output": LM.ptr[LM.f32],
        "compile_sentinel": LM.ptr[LM.i32],
    }
    return loom.weave(
        threads=THREADS,
        cluster_dims=CLUSTER_DIMS,
    )(small_rs_l1_tma_rs_fragment)


def _partial_l1_rs_kernel(*, n_swap: int, loader_registers: int):
    """Build one parent-shaped L1 handoff/acquire/RS-WGMMA fragment.

    This retains fused_body.inl:709, 835-836 and 908-1098 without claiming that
    its generic compile-only staging below is the eventual TMA scheduler
    integration. ``n_swap`` is one of the three static WGMMA shapes selected
    by the source's runtime ``valid_m`` branches.
    """

    if n_swap not in (8, 16, 24):
        raise ValueError(f"unsupported source N_SWAP={n_swap}")
    if loader_registers not in (40, 64):
        raise ValueError(f"unsupported loader register budget {loader_registers}")
    accum_count = n_swap // 2
    token_chunks = n_swap // 8
    act_offset = RS_WEIGHT_BYTES
    lut_offset = act_offset + RS_ACT_BYTES
    sfa_offset = lut_offset + RS_LUT_BYTES

    @loom.weave(threads=THREADS, cluster_dims=CLUSTER_DIMS)
    def small_rs_l1_rs_fragment(
        lm,
        packed_weights: LM.ptr[LM.u32],
        activation: LM.ptr[LM.u32],
        activation_scale: LM.ptr[LM.f32],
        l1_arrival_count: LM.ptr[LM.u32],
        pool_block_idx: LM.i32,
        valid_m: LM.i32,
        output: LM.ptr[LM.f32],
        compile_sentinel: LM.ptr[LM.i32],
    ):
        lut = lm.constant_array(
            "kE2M1AndUe4m3ToFp8Lut",
            dtype="u32",
            values=RS_LUT_WORD_VALUES,
            align=16,
        )
        # Keep the compile fragment inside the exact parent launch envelope.
        # The first 1024 bytes above this user pool are Loom's typed mbarriers.
        pool = lm.smem(SMEM_BYTES - LOOM_CONTROL_BYTES)
        packed_smem = pool.view(
            offset=0,
            shape=(RS_WEIGHT_BYTES,),
            dtype=lm.u8,
            swizzle=None,
        )
        act_smem = pool.view(
            offset=act_offset,
            shape=(RS_ACT_ROWS, RS_ACT_ROW_BYTES),
            dtype=lm.u8,
            logical_shape=(RS_ACT_ROWS, RS_BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
        )
        lut_smem = pool.view(
            offset=lut_offset,
            shape=(RS_LUT_WORDS,),
            dtype=lm.u32,
            swizzle=None,
        )
        sfa_smem = pool.view(
            offset=sfa_offset,
            shape=(RS_ACT_ROWS,),
            dtype=lm.f32,
            swizzle=None,
        )

        dispatch = lm.role("dispatch", warps=[0, 1])
        loader = lm.role("loader", warps=[2, 3])
        math = lm.role("math", warps=[4, 5, 6, 7, 8, 9, 10, 11])
        rs_ready = lm.barrier(
            count=2,
            prod=[loader],
            cons=[math],
            init_count=1,
        )

        with dispatch:
            lm.setmaxnreg_dealloc(48)
            lm.constant_array_copy16b(
                lut,
                dst_addr=lut_smem.addr + lm.tid * 16,
                chunk=lm.tid,
            )
            with lm.elected_thread(warp=0):
                lm.gmem_store(
                    compile_sentinel,
                    48,
                    dtype=lm.i32,
                    src_dtype=lm.i32,
                )
            # Source dispatch cleanup and math scheduling meet at this
            # unaligned 64 + 256 thread handoff (fused_body.inl:674/836).
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )

        with loader:
            lm.setmaxnreg_dealloc(loader_registers)

            # Warp 2 cannot expose an L1 block to its TMA stage until every
            # routed token has published arrival (fused_body.inl:705-709).
            if lm.warp_id == 2:
                if valid_m > 0:
                    expected_arrivals: lm.u32 = valid_m.u32()
                    lm.gmem_wait_acquire(
                        l1_arrival_count,
                        expected_arrivals,
                        index=pool_block_idx,
                        dtype=lm.u32,
                        predicate=lm.WAIT_EQ,
                        scope=lm.FENCE_DEVICE,
                    )

            # Materialize one BK128 source stage. The complete consumer will
            # replace these generic compile-only copies with the source TMA
            # activation/SFA and packed-weight transactions.
            if lm.warp_id == 2:
                for vec_pass in lm.range(0, 6, unroll=1):
                    vec_idx: lm.i32 = vec_pass * 32 + lm.lane_id
                    word_base: lm.i32 = vec_idx * 4
                    act_smem.swizzled_vec_store(
                        row=vec_idx // 8,
                        col_bytes=(vec_idx % 8) * 16,
                        values=(
                            activation[word_base + 0],
                            activation[word_base + 1],
                            activation[word_base + 2],
                            activation[word_base + 3],
                        ),
                    )
                if lm.lane_id < RS_ACT_ROWS:
                    sfa_smem.ptr[lm.lane_id] = activation_scale[lm.lane_id]
            else:
                for vec_pass in lm.range(0, 40, unroll=1):
                    vec_idx: lm.i32 = vec_pass * 32 + lm.lane_id
                    word_base: lm.i32 = vec_idx * 4
                    lm.smem_store_vec(
                        dst_addr=packed_smem.addr + (word_base + 0) * 4,
                        src=packed_weights[word_base + 0],
                    )
                    lm.smem_store_vec(
                        dst_addr=packed_smem.addr + (word_base + 1) * 4,
                        src=packed_weights[word_base + 1],
                    )
                    lm.smem_store_vec(
                        dst_addr=packed_smem.addr + (word_base + 2) * 4,
                        src=packed_weights[word_base + 2],
                    )
                    lm.smem_store_vec(
                        dst_addr=packed_smem.addr + (word_base + 3) * 4,
                        src=packed_weights[word_base + 3],
                    )
            lm.fence_shared()
            lm.sync_warp()
            with lm.elected_thread():
                lm.arrive(rs_ready, stage=0)

        with math:
            lm.setmaxnreg_alloc(208)
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=DISPATCH_WITH_MATH_THREADS,
            )
            lm.wait(rs_ready, stage=0)
            lm.fence_shared()

            math_wg: lm.i32 = (lm.warp_id - 4) // 4
            warp_in_wg: lm.i32 = (lm.warp_id - 4) % 4
            row_in_warp: lm.i32 = lm.lane_id // 4
            word_sel: lm.i32 = (lm.lane_id >> 1) & 1
            final_accum: lm.f32[64]
            final_accum.fill_(0.0)

            for half in lm.range(0, 2, constexpr=True):
                decode_row: lm.i32 = (
                    math_wg * 128
                    + half * 64
                    + warp_in_wg * 16
                    + row_in_warp
                    + ((lm.lane_id & 1) << 3)
                )
                packed_row_addr: lm.i32 = (
                    packed_smem.addr + decode_row * RS_WEIGHT_ROW_BYTES
                )
                scale_words: lm.u32[2]
                lm.smem_load_vec(
                    dst=scale_words,
                    src_addr=packed_row_addr + 64,
                    count=2,
                )

                # Keep all four K32 register fragments live before issuing the
                # single source WGMMA group.
                a_frags: lm.u32[16]
                for k_slice in lm.range(0, 4, constexpr=True):
                    packed_lo: lm.u32[1]
                    packed_hi: lm.u32[1]
                    lm.smem_load_vec(
                        dst=packed_lo,
                        src_addr=(
                            packed_row_addr
                            + k_slice * 16
                            + word_sel * 4
                        ),
                        count=1,
                    )
                    lm.smem_load_vec(
                        dst=packed_hi,
                        src_addr=(
                            packed_row_addr
                            + k_slice * 16
                            + 8
                            + word_sel * 4
                        ),
                        count=1,
                    )
                    scale_word: lm.u32 = scale_words[0]
                    if k_slice >= 2:
                        scale_word = scale_words[1]
                    scale_shift: lm.i32 = (k_slice & 1) * 16
                    scale_lo: lm.u32 = (scale_word >> scale_shift) & 0x7F
                    scale_hi: lm.u32 = (
                        scale_word >> (scale_shift + 8)
                    ) & 0x7F
                    lut_lo: lm.u32[2]
                    lut_hi: lm.u32[2]
                    lm.smem_load_vec(
                        dst=lut_lo,
                        src_addr=lut_smem.addr + scale_lo.i32() * 8,
                        count=2,
                    )
                    lm.smem_load_vec(
                        dst=lut_hi,
                        src_addr=lut_smem.addr + scale_hi.i32() * 8,
                        count=2,
                    )

                    decoded_lo_hi: lm.u32 = 0
                    decoded_lo_lo: lm.u32 = 0
                    decoded_hi_hi: lm.u32 = 0
                    decoded_hi_lo: lm.u32 = 0
                    lm.nvfp4_mode2_lut_decode_word(
                        dst_hi=decoded_lo_hi,
                        dst_lo=decoded_lo_lo,
                        packed=packed_lo[0],
                        lut_lo=lut_lo[0],
                        lut_hi=lut_lo[1],
                    )
                    lm.nvfp4_mode2_lut_decode_word(
                        dst_hi=decoded_hi_hi,
                        dst_lo=decoded_hi_lo,
                        packed=packed_hi[0],
                        lut_lo=lut_hi[0],
                        lut_hi=lut_hi[1],
                    )

                    keep_lo: lm.u32 = decoded_lo_hi
                    ship_lo: lm.u32 = decoded_lo_lo
                    keep_hi: lm.u32 = decoded_hi_hi
                    ship_hi: lm.u32 = decoded_hi_lo
                    if (lm.lane_id & 1) != 0:
                        keep_lo = decoded_lo_lo
                        ship_lo = decoded_lo_hi
                        keep_hi = decoded_hi_lo
                        ship_hi = decoded_hi_hi
                    recv_lo: lm.u32 = lm.shfl_xor_sync(
                        ship_lo,
                        1,
                        dtype=lm.u32,
                    )
                    recv_hi: lm.u32 = lm.shfl_xor_sync(
                        ship_hi,
                        1,
                        dtype=lm.u32,
                    )
                    frag_base: lm.i32 = k_slice * 4
                    if (lm.lane_id & 1) == 0:
                        a_frags[frag_base + 0] = keep_lo
                        a_frags[frag_base + 1] = recv_lo
                        a_frags[frag_base + 2] = keep_hi
                        a_frags[frag_base + 3] = recv_hi
                    else:
                        a_frags[frag_base + 0] = recv_lo
                        a_frags[frag_base + 1] = keep_lo
                        a_frags[frag_base + 2] = recv_hi
                        a_frags[frag_base + 3] = keep_hi
                    for frag_word in lm.range(0, 4, constexpr=True):
                        lm.wgmma_fence_operand(a_frags[frag_base + frag_word])

                swap_accum: lm.f32[accum_count]
                swap_accum.fill_(0.0)
                for accum_idx in lm.range(0, accum_count, constexpr=True):
                    lm.wgmma_fence_operand(swap_accum[accum_idx])
                lm.wgmma_fence()
                lm.mma(
                    swap_accum,
                    a_frags[0:4],
                    act_smem[0].tile((0, 0), (n_swap, 32)),
                    init=True,
                )
                lm.mma(
                    swap_accum,
                    a_frags[4:8],
                    act_smem[0].tile((0, 32), (n_swap, 32)),
                    init=False,
                )
                lm.mma(
                    swap_accum,
                    a_frags[8:12],
                    act_smem[0].tile((0, 64), (n_swap, 32)),
                    init=False,
                )
                lm.mma(
                    swap_accum,
                    a_frags[12:16],
                    act_smem[0].tile((0, 96), (n_swap, 32)),
                    init=False,
                )
                lm.wgmma_commit_group()
                for accum_idx in lm.range(0, accum_count, constexpr=True):
                    lm.wgmma_fence_operand(swap_accum[accum_idx])
                lm.wgmma_wait_group(n=0)
                for frag_idx in lm.range(0, 16, constexpr=True):
                    lm.wgmma_fence_operand(a_frags[frag_idx])

                col_idx: lm.i32 = lm.lane_id % 4
                for token_chunk in lm.range(
                    0,
                    token_chunks,
                    constexpr=True,
                ):
                    token_0: lm.i32 = token_chunk * 8 + col_idx * 2
                    token_1: lm.i32 = token_0 + 1
                    accum_offset: lm.i32 = half * 32 + token_chunk * 4
                    if token_0 < valid_m:
                        scale_0: lm.f32[1]
                        lm.smem_load_vec(
                            dst=scale_0,
                            src_addr=sfa_smem.addr + token_0 * 4,
                            count=1,
                        )
                        final_accum[accum_offset + 0] = (
                            final_accum[accum_offset + 0]
                            + scale_0[0] * swap_accum[token_chunk * 4 + 0]
                        )
                        final_accum[accum_offset + 2] = (
                            final_accum[accum_offset + 2]
                            + scale_0[0] * swap_accum[token_chunk * 4 + 2]
                        )
                    if token_1 < valid_m:
                        scale_1: lm.f32[1]
                        lm.smem_load_vec(
                            dst=scale_1,
                            src_addr=sfa_smem.addr + token_1 * 4,
                            count=1,
                        )
                        final_accum[accum_offset + 1] = (
                            final_accum[accum_offset + 1]
                            + scale_1[0] * swap_accum[token_chunk * 4 + 1]
                        )
                        final_accum[accum_offset + 3] = (
                            final_accum[accum_offset + 3]
                            + scale_1[0] * swap_accum[token_chunk * 4 + 3]
                        )

            # The source's dynamic WG barrier IDs become static IDs in the
            # two role branches and therefore emit the identical instructions.
            if lm.warp_id < 8:
                lm.sync(bar_id=3, threads=128, aligned=True)
            else:
                lm.sync(bar_id=4, threads=128, aligned=True)

            math_tid: lm.i32 = (lm.warp_id - 4) * 32 + lm.lane_id
            for out_idx in lm.range(0, 64, constexpr=True):
                output[math_tid * 64 + out_idx] = final_accum[out_idx]

    return small_rs_l1_rs_fragment


_PARTIAL_DISPATCH_KERNEL_CACHE: dict = {}


def _partial_dispatch_kernel(schedule: SmallRSSchedule):
    """Return the (immutable, memoized) source dispatch fragment for one row.

    The IR is built once per material schedule; every host call reuses it for
    descriptor binding, argument packing and the artifact identity check.
    """

    kernel = _PARTIAL_DISPATCH_KERNEL_CACHE.get(schedule)
    if kernel is None:
        kernel = _build_partial_dispatch_kernel(schedule)
        _PARTIAL_DISPATCH_KERNEL_CACHE[schedule] = kernel
    return kernel


def _build_partial_dispatch_kernel(schedule: SmallRSSchedule):
    """Build one retained post-initialization source dispatch fragment."""

    layout = _dispatch_layout(schedule)
    uses_interleaved_scheduler = _uses_current_interleaved_scheduler()
    loader_registers = 64 if uses_interleaved_scheduler else 40
    active_dispatch_warps = 1 if schedule.single_dispatch else 2
    active_dispatch_threads = active_dispatch_warps * 32
    active_lanes = TOKENS_PER_WARP * TOP_K
    experts_per_rank = schedule.experts // NUM_RANKS
    experts_per_lane = (experts_per_rank + 31) // 32
    l1_shape_n = schedule.intermediate * 2
    l1_shape_k = schedule.hidden
    l2_shape_n = schedule.hidden
    l2_shape_k = schedule.intermediate
    l1_n_blocks = l1_shape_n // 256
    l2_n_blocks = l2_shape_n // 256
    l2_expected_arrival_mask = (1 << l1_n_blocks) - 1
    first_l2_wave_m_blocks = (NUM_CTAS + l2_n_blocks - 1) // l2_n_blocks
    l1_warmup_for_first_l2 = (
        first_l2_wave_m_blocks * l1_n_blocks + NUM_CTAS - 1
    ) // NUM_CTAS
    interleave_task_diff = max(l1_n_blocks - l2_n_blocks, 0)
    # Both production model families have l1_n_blocks <= l2_n_blocks, so the
    # source's runtime-total-dependent term folds to this exact constant.
    l1_warmup_for_interleave = (l1_n_blocks + NUM_CTAS - 1) // NUM_CTAS + 1
    min_l1_warmup_waves = max(
        l1_warmup_for_first_l2,
        l1_warmup_for_interleave,
    )
    if interleave_task_diff != 0:
        raise ValueError("unexpected source interleave task-count ordering")
    max_recv_tokens_per_expert = NUM_RANKS * MAX_TOKENS_PER_RANK
    smem_expert_count_bytes = _align(schedule.experts * 4, 1024)
    smem_send_buffer_bytes = _align(
        schedule.hidden * active_dispatch_warps,
        1024,
    )
    task_info_offset = smem_expert_count_bytes + smem_send_buffer_bytes
    l1_act_stage_bytes = schedule.block_m * RS_ACT_ROW_BYTES
    # Source SMEM_SFA_SIZE_PER_STAGE is constexpr_align(BLOCK_M * 4, 128): every
    # per-stage SFA slot is a 128-byte-aligned TMA destination even though the
    # transaction itself moves only BLOCK_M floats.
    l1_sfa_tile_bytes = schedule.block_m * 4
    l1_sfa_stage_bytes = _align(l1_sfa_tile_bytes, 128)
    l1_packed_stage_bytes = RS_TMA_PACKED_B_BYTES
    l1_act_offset = _align(
        task_info_offset + TASK_INFO_STAGES * TASK_INFO_BYTES,
        1024,
    )
    l1_sfa_offset = l1_act_offset + schedule.stages * l1_act_stage_bytes
    l1_packed_offset = l1_sfa_offset + schedule.stages * l1_sfa_stage_bytes
    l1_lut_offset = (
        l1_packed_offset + schedule.stages * l1_packed_stage_bytes
    )
    l1_lutc_offset = l1_lut_offset + RS_LUT_BYTES
    l1_lutc_bytes = RS_LUT_COMPACT_BYTES if (RS_LUT_COMPACT and schedule.swap_ab) else 0
    l1_decoded_offset = _align(l1_lutc_offset + l1_lutc_bytes, 1024)
    l1_decoded_bytes = schedule.stages * SS_DECODED_B_STAGE_BYTES
    # Swap-AB rows with the dispatch decode team own a 256-row x 32-byte FP8
    # slice tile per stage in the same phase-disjoint range.
    use_team_slice3 = bool(RS_TEAM_SLICE3 and schedule.swap_ab)
    use_wide_math = _wide_math_for(schedule)
    use_lut_compact = bool(RS_LUT_COMPACT and schedule.swap_ab)

    def _compact_lut_constant(lm):
        if not use_lut_compact:
            return None
        return lm.constant_array(
            "kE2M1AndUe4m3ToFp8LutCompact",
            dtype="u32",
            values=RS_LUT_COMPACT_WORD_VALUES,
            align=16,
        )

    def _compact_lut_view(lm, pool, offset):
        if not use_lut_compact:
            return None
        return pool.view(offset=offset, shape=(RS_LUT_COMPACT_WORDS,), dtype=lm.u32, swizzle=None)
    math_warps = _math_warps_for(schedule)
    math_threads = 32 * math_warps
    dispatch_with_math_threads = DISPATCH_THREADS + math_threads
    kernel_threads = _kernel_threads_for(schedule)
    math_registers = RS_WIDE_MATH_REGISTERS if use_wide_math else 208
    task_info_empty_count = math_warps + (2 if use_team_slice3 else 0)
    dispatch_registers = RS_TEAM_REGISTERS if use_team_slice3 else 48

    def _with_team(*roles, team):
        return list(roles) + ([team] if use_team_slice3 else [])

    def _team_decoded_view(lm, pool):
        if not use_team_slice3:
            return None
        return pool.view(
            offset=l1_decoded_offset,
            shape=(RS_WEIGHT_ROWS, RS_TEAM_DECODED_ROW_BYTES),
            dtype=lm.u8,
            logical_shape=(RS_WEIGHT_ROWS, RS_TEAM_DECODED_ROW_BYTES),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_32b,
            stage=schedule.stages,
            stride=RS_TEAM_DECODED_STAGE_BYTES,
        )
    l1_team_decoded_bytes = (
        schedule.stages * RS_TEAM_DECODED_STAGE_BYTES if use_team_slice3 else 0
    )
    l1_smem_end = (
        (
            l1_decoded_offset + l1_team_decoded_bytes
            if use_team_slice3
            else l1_lutc_offset + l1_lutc_bytes
        )
        if schedule.swap_ab
        else l1_decoded_offset + l1_decoded_bytes
    )
    l1_output_offset = _align(l1_smem_end, 1024)
    l1_output_bytes = schedule.block_m * 128
    l1_amax_offset = _align(l1_output_offset + l1_output_bytes, 128)
    l1_amax_bytes = schedule.block_m * math_warps * 4
    # The source allocates the BF16 CD alias only for swap-AB.  Non-swap L1
    # needs the E4M3 output plus its two-warpgroup amax scratch, not the later
    # BLOCK_M-by-256 BF16 remote-scatter tile.
    l2_output_bytes = (
        schedule.block_m * 256 * 2 if schedule.swap_ab else 0
    )
    l1_epilogue_smem_end = max(
        l1_amax_offset + l1_amax_bytes,
        l1_output_offset + l2_output_bytes,
    )
    is_swap_ab = schedule.swap_ab
    is_block_m8 = schedule.block_m == 8
    is_block_m16 = schedule.block_m == 16
    if l1_epilogue_smem_end > SMEM_BYTES - LOOM_CONTROL_BYTES:
        raise ValueError(
            f"small-RS L1 integration needs {l1_epilogue_smem_end} bytes, "
            f"only {SMEM_BYTES - LOOM_CONTROL_BYTES} are available"
        )
    combine_num_chunks = (
        1
        if (
            3 * EPILOGUE_WARPS * schedule.hidden * 2
            <= SMEM_BYTES - LOOM_CONTROL_BYTES
            and schedule.hidden <= 32 * 128
        )
        else 2
    )
    if use_wide_math:
        # 112-register math budget: halve the per-lane combine accumulator
        # (64 or 56 floats instead of 128 or 112); per-element arithmetic is
        # unchanged, so the output stays bit-identical.
        combine_num_chunks *= 2
    combine_chunk_bytes = schedule.hidden * 2 // combine_num_chunks
    combine_elems_per_lane = (
        combine_chunk_bytes // 2 // 32
    )
    combine_smem_bytes = 3 * EPILOGUE_WARPS * combine_chunk_bytes
    if combine_smem_bytes > SMEM_BYTES - LOOM_CONTROL_BYTES:
        raise ValueError(
            f"small-RS combine needs {combine_smem_bytes} bytes, only "
            f"{SMEM_BYTES - LOOM_CONTROL_BYTES} are available"
        )

    def small_rs_dispatch_fragment(
        lm,
        num_tokens,
        l1_activation,
        l1_activation_scale,
        l1_packed_weights,
        l1_output,
        l2_activation,
        l2_activation_scale,
        l2_packed_weights,
        l1_global_scales,
        l2_global_scales,
        output_bf16,
    ):
        lut = lm.constant_array(
            "kE2M1AndUe4m3ToFp8Lut",
            dtype="u32",
            values=RS_LUT_WORD_VALUES,
            align=16,
        )
        lutc = _compact_lut_constant(lm)
        pg = lm.process_group("pg", world_size=NUM_RANKS)
        sym_buffer = lm.symmetric_memory(
            "sym_buffer",
            shape=(layout.symmetric_bytes,),
            dtype="u8",
            group=pg,
        )

        # Retain the source's single contiguous symmetric allocation. Typed
        # views below are exact byte slices of layout::Workspace and the fused
        # input/L1 buffers, including peer mappings for remote publication.
        sym_bytes = lm.byte_ptr(sym_buffer)
        workspace_u32 = lm.ptr_to(sym_bytes, lm.u32)
        workspace_i32 = lm.ptr_to(sym_bytes, lm.i32)
        l1_task_count_u32 = workspace_u32 + layout.l1_task_count_offset // 4
        l2_task_count_u32 = workspace_u32 + layout.l2_task_count_offset // 4
        l1_task_count_i32 = workspace_i32 + layout.l1_task_count_offset // 4
        l2_task_count_i32 = workspace_i32 + layout.l2_task_count_offset // 4
        send_count_u64 = lm.ptr_to(
            sym_bytes + layout.expert_send_count_offset,
            lm.u64,
        )
        recv_count_u64 = lm.ptr_to(
            sym_bytes + layout.expert_recv_count_offset,
            lm.u64,
        )
        recv_sum_u64 = lm.ptr_to(
            sym_bytes + layout.expert_recv_sum_offset,
            lm.u64,
        )
        src_token_topk_i32 = lm.ptr_to(
            sym_bytes + layout.src_token_topk_offset,
            lm.i32,
        )
        token_src_metadata_i32 = lm.ptr_to(
            sym_bytes + layout.token_src_metadata_offset,
            lm.i32,
        )
        input_topk_idx_i64 = lm.ptr_to(
            sym_bytes + layout.input_topk_idx_offset,
            lm.i64,
        )
        l1_token_u8 = lm.ptr_to(
            sym_bytes + layout.l1_token_offset,
            lm.u8,
        )
        l1_sf_f32 = lm.ptr_to(
            sym_bytes + layout.l1_sf_offset,
            lm.f32,
        )
        l1_topk_weights_f32 = lm.ptr_to(
            sym_bytes + layout.l1_topk_weights_offset,
            lm.f32,
        )
        l1_arrival_u32 = lm.ptr_to(
            sym_bytes + layout.l1_arrival_count_offset,
            lm.u32,
        )
        l1_arrival_i32 = lm.ptr_to(
            sym_bytes + layout.l1_arrival_count_offset,
            lm.i32,
        )
        l2_arrival_u64 = lm.ptr_to(
            sym_bytes + layout.l2_arrival_mask_offset,
            lm.u64,
        )
        l2_sf_f32 = lm.ptr_to(
            sym_bytes + layout.l2_sf_offset,
            lm.f32,
        )

        # Loom owns the first aligned control page for the typed pull
        # mbarriers. Keep that page inside the source's fixed 232448-byte
        # launch envelope rather than increasing dynamic shared memory.
        pool = lm.smem(SMEM_BYTES - LOOM_CONTROL_BYTES)
        smem_expert_count = pool.view(
            offset=0,
            shape=(schedule.experts,),
            dtype=lm.u32,
        )
        pull_stage = pool.view(
            offset=smem_expert_count_bytes,
            shape=(smem_send_buffer_bytes,),
            dtype=lm.u8,
            swizzle=None,
        )
        task_info_smem = pool.view(
            offset=task_info_offset,
            shape=(TASK_INFO_FIELDS,),
            dtype=lm.u32,
            swizzle=None,
            stage=TASK_INFO_STAGES,
            stride=TASK_INFO_BYTES,
        )
        l1_act_smem = pool.view(
            offset=l1_act_offset,
            shape=(schedule.block_m, RS_ACT_ROW_BYTES),
            dtype=lm.u8,
            logical_shape=(schedule.block_m, RS_BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=schedule.stages,
            stride=l1_act_stage_bytes,
        )
        l1_sfa_smem = pool.view(
            offset=l1_sfa_offset,
            shape=(schedule.block_m,),
            dtype=lm.f32,
            swizzle=None,
            stage=schedule.stages,
            stride=l1_sfa_stage_bytes,
        )
        l1_packed_smem = pool.view(
            offset=l1_packed_offset,
            shape=(RS_TMA_PACKED_B_BYTES,),
            dtype=lm.u8,
            swizzle=None,
            stage=schedule.stages,
            stride=l1_packed_stage_bytes,
        )
        l1_lut_smem = pool.view(
            offset=l1_lut_offset,
            shape=(RS_LUT_WORDS,),
            dtype=lm.u32,
            swizzle=None,
        )
        l1_lutc_smem = _compact_lut_view(lm, pool, l1_lutc_offset)
        # The non-swap M64N128 path expands each packed N256-by-K128 tile into
        # the source's 128-byte-swizzled FP8 operand.  Swap-AB does not touch
        # this phase-disjoint alias and retains its existing compact layout.
        l1_decoded_smem = pool.view(
            offset=l1_decoded_offset,
            shape=(RS_WEIGHT_ROWS, RS_BLOCK_K),
            dtype=lm.u8,
            logical_shape=(RS_WEIGHT_ROWS, RS_BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=schedule.stages,
            stride=SS_DECODED_B_STAGE_BYTES,
        )
        # Dispatch decode team output (team rows only): K-slice 3 of every
        # packed row as one 32-byte-swizzled FP8 row for the SS WGMMA slice.
        l1_team_decoded_smem = _team_decoded_view(lm, pool)
        l1_output_smem = pool.view(
            offset=l1_output_offset,
            shape=(schedule.block_m, 128),
            dtype=lm.u8,
            swizzle=None,
        )
        l1_amax_smem = pool.view(
            offset=l1_amax_offset,
            shape=(schedule.block_m * math_warps,),
            dtype=lm.f32,
            swizzle=None,
        )
        # Source CD is a phase-disjoint union: L1 reuses this range as an
        # E4M3 tile plus amax scratch; L2 reinterprets it as BM x BN256 BF16.
        l2_output_smem = pool.view(
            offset=l1_output_offset,
            shape=(schedule.block_m, 256),
            dtype=lm.bf16,
            swizzle=lm.swizzle_none,
        )
        # After GEMM and remote scatter, the source reuses the entire pre-barrier
        # SMEM range for two per-warp combine load slots and one store slot.
        combine_load_smem = pool.view(
            offset=0,
            shape=(combine_chunk_bytes,),
            dtype=lm.u8,
            swizzle=lm.swizzle_none,
            stage=EPILOGUE_WARPS * 2,
            stride=combine_chunk_bytes,
        )
        combine_store_smem = pool.view(
            offset=EPILOGUE_WARPS * 2 * combine_chunk_bytes,
            shape=(combine_chunk_bytes,),
            dtype=lm.u8,
            swizzle=lm.swizzle_none,
            stage=EPILOGUE_WARPS,
            stride=combine_chunk_bytes,
        )

        dispatch = lm.role("dispatch", warps=[0, 1])
        loader = lm.role("loader", warps=[2, 3])
        math = lm.role("math", warps=list(range(4, 4 + math_warps)))
        pull_full = lm.barrier(
            count=active_dispatch_warps,
            prod=[dispatch],
            cons=[dispatch],
            init_count=1,
        )
        task_info_pipe = lm.pipeline(stages=TASK_INFO_STAGES)
        task_info_full = lm.barrier(
            count=TASK_INFO_STAGES,
            prod=[loader],
            cons=_with_team(loader, math, team=dispatch),
            init_count=1,
            pipeline=task_info_pipe,
        )
        # The eight math warps return every valid slot; with the decode team
        # both dispatch warps return it as well.
        task_info_empty = lm.barrier(
            count=TASK_INFO_STAGES,
            prod=_with_team(math, team=dispatch),
            cons=[loader],
            init_count=task_info_empty_count,
            init_phase=1,
            pipeline=task_info_pipe,
        )
        l1_pipe = lm.pipeline(stages=schedule.stages)
        l1_stage_full = lm.barrier(
            count=schedule.stages,
            prod=[loader],
            cons=_with_team(math, team=dispatch),
            init_count=2,
            pipeline=l1_pipe,
        )
        l1_stage_empty = lm.barrier(
            count=schedule.stages,
            prod=[math],
            cons=[loader],
            init_count=math_warps,
            init_phase=1,
            pipeline=l1_pipe,
        )
        l1_stage_decoded = None
        if use_team_slice3:
            l1_stage_decoded = lm.barrier(
                count=schedule.stages,
                prod=[dispatch],
                cons=[math],
                init_count=1,
                pipeline=l1_pipe,
            )
        combine_pipe = lm.pipeline(stages=EPILOGUE_WARPS * 2)
        combine_full = lm.barrier(
            count=EPILOGUE_WARPS * 2,
            prod=[math],
            cons=[math],
            init_count=1,
            pipeline=combine_pipe,
        )

        with dispatch:
            lm.setmaxnreg_dealloc(dispatch_registers)

            # The source's first 64 threads initialize the exact Mode2
            # E2M1/UE4M3 lookup table for both compact RS and expanded SS
            # consumers before the first dispatch/math handoff.
            lm.constant_array_copy16b(
                lut,
                dst_addr=l1_lut_smem.addr + lm.tid * 16,
                chunk=lm.tid,
            )
            if use_lut_compact:
                if lm.tid < RS_LUT_COMPACT_BYTES // 16:
                    lm.constant_array_copy16b(
                        lutc,
                        dst_addr=l1_lutc_smem.addr + lm.tid * 16,
                        chunk=lm.tid,
                    )

            # The full source kernel initializes this region before role
            # divergence. This retained fragment begins at the same clean
            # post-initialization boundary and materializes that state with
            # the two dispatch warps only.
            if lm.warp_id == 0:
                for expert in lm.range(
                    lm.lane_id,
                    schedule.experts,
                    32,
                    unroll=1,
                    dtype=lm.i32,
                ):
                    lm.smem_store_vec(
                        dst_addr=smem_expert_count.addr + expert * 4,
                        src=0,
                    )
            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # Count token-topk routes into the CTA-local expert histogram.
            if lm.warp_id < active_dispatch_warps:
                route_base: lm.i32 = (
                    (lm.bid * active_dispatch_warps + lm.warp_id)
                    * TOKENS_PER_WARP
                )
                route_stride: lm.i32 = (
                    NUM_CTAS * active_dispatch_warps * TOKENS_PER_WARP
                )
                for token_base in lm.range(
                    route_base,
                    num_tokens,
                    route_stride,
                    unroll=1,
                    dtype=lm.i32,
                ):
                    token_idx: lm.i32 = token_base + lm.lane_id // TOP_K
                    token_topk_idx: lm.i32 = token_base * TOP_K + lm.lane_id
                    if lm.logical_and(
                        lm.lane_id < active_lanes,
                        token_idx < num_tokens,
                    ):
                        expert_raw = lm.vec_load(
                            input_topk_idx_i64,
                            index=token_topk_idx,
                            count=1,
                            dtype=lm.i64,
                            dst_dtype=lm.i64,
                            cache_hint="read_only",
                        )
                        expert_idx: lm.i32 = expert_raw[0].i32()
                        if expert_idx >= 0:
                            _count_slot: lm.u32 = lm.smem_atomic_fetch_add(
                                smem_expert_count.addr,
                                1,
                                index=expert_idx,
                            )
                    lm.sync_warp()

            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # Stake this CTA's per-expert slots using the source's one relaxed
            # global u64 fetch-add: high32 counts contributing CTAs and low32
            # accumulates routed token-topk entries.
            for expert in lm.range(
                lm.tid,
                schedule.experts,
                DISPATCH_THREADS,
                unroll=1,
                dtype=lm.i32,
            ):
                count_word: lm.u32[1]
                lm.smem_load_vec(
                    dst=count_word,
                    src_addr=smem_expert_count.addr + expert * 4,
                    count=1,
                )
                send_value: lm.u64 = 0x100000000 + count_word[0].u64()
                old_status: lm.u64 = lm.atomic_fetch_add(
                    send_count_u64,
                    send_value,
                    index=expert,
                    dtype=lm.u64,
                    scope=lm.ATOMIC_SCOPE_GPU,
                )
                lm.smem_store_vec(
                    dst_addr=smem_expert_count.addr + expert * 4,
                    src=old_status.u32(),
                )

            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # Re-read the routes and publish source token-topk indices into
            # the destination rank's exact workspace slot.
            if lm.warp_id < active_dispatch_warps:
                route_base: lm.i32 = (
                    (lm.bid * active_dispatch_warps + lm.warp_id)
                    * TOKENS_PER_WARP
                )
                route_stride: lm.i32 = (
                    NUM_CTAS * active_dispatch_warps * TOKENS_PER_WARP
                )
                for token_base in lm.range(
                    route_base,
                    num_tokens,
                    route_stride,
                    unroll=1,
                    dtype=lm.i32,
                ):
                    token_idx: lm.i32 = token_base + lm.lane_id // TOP_K
                    token_topk_idx: lm.i32 = token_base * TOP_K + lm.lane_id
                    if lm.logical_and(
                        lm.lane_id < active_lanes,
                        token_idx < num_tokens,
                    ):
                        expert_raw = lm.vec_load(
                            input_topk_idx_i64,
                            index=token_topk_idx,
                            count=1,
                            dtype=lm.i64,
                            dst_dtype=lm.i64,
                            cache_hint="read_only",
                        )
                        expert_idx: lm.i32 = expert_raw[0].i32()
                        if expert_idx >= 0:
                            dst_rank: lm.i32 = expert_idx // experts_per_rank
                            dst_slot: lm.u32 = lm.smem_atomic_fetch_add(
                                smem_expert_count.addr,
                                1,
                                index=expert_idx,
                            )
                            peer_src_token_topk_i32 = lm.ptr_to(
                                lm.byte_ptr(sym_buffer.peer_ptr(dst_rank))
                                + layout.src_token_topk_offset,
                                lm.i32,
                            )
                            dst_index: lm.u64 = (
                                (expert_idx % experts_per_rank).u64()
                                * (NUM_RANKS * max_recv_tokens_per_expert)
                                + pg.rank.u64() * max_recv_tokens_per_expert
                                + dst_slot.u64()
                            )
                            lm.gmem_store(
                                peer_src_token_topk_i32,
                                token_topk_idx,
                                index=dst_index,
                                dtype=lm.i32,
                                src_dtype=lm.i32,
                            )
                    lm.sync_warp()

            # Exact dispatch grid boundary around the remote source-index
            # scatter. One elected thread per CTA drives the workspace tag.
            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )
            with lm.elected_thread(warp=0):
                lm.grid_sync_workspace(
                    counter=(
                        workspace_u32
                        + layout.grid_sync_offset // 4
                        + DISPATCH_GRID_SYNC_INDEX
                    ),
                    num_ctas=NUM_CTAS,
                    leader=lm.bid == 0,
                )
            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # CTA 0 publishes each expert's final status to its owning rank:
            # the rank-specific low32 count plus the ordered u64 aggregate.
            if lm.logical_and(
                lm.bid == 0,
                lm.tid < active_dispatch_threads,
            ):
                for expert in lm.range(
                    lm.tid,
                    schedule.experts,
                    active_dispatch_threads,
                    unroll=1,
                    dtype=lm.i32,
                ):
                    dst_rank: lm.i32 = expert // experts_per_rank
                    dst_local_expert: lm.i32 = expert % experts_per_rank
                    status_vec = lm.vec_load(
                        send_count_u64,
                        index=expert,
                        count=1,
                        dtype=lm.u64,
                        dst_dtype=lm.u64,
                    )
                    expert_status: lm.u64 = status_vec[0]
                    peer_bytes = lm.byte_ptr(sym_buffer.peer_ptr(dst_rank))
                    peer_recv_count_u64 = lm.ptr_to(
                        peer_bytes + layout.expert_recv_count_offset,
                        lm.u64,
                    )
                    peer_recv_sum_u64 = lm.ptr_to(
                        peer_bytes + layout.expert_recv_sum_offset,
                        lm.u64,
                    )
                    lm.gmem_store(
                        peer_recv_count_u64,
                        expert_status & 0xFFFFFFFF,
                        index=pg.rank * experts_per_rank + dst_local_expert,
                        dtype=lm.u64,
                        src_dtype=lm.u64,
                    )
                    _published: lm.u64 = lm.atomic_fetch_add(
                        peer_recv_sum_u64,
                        expert_status,
                        index=dst_local_expert,
                        dtype=lm.u64,
                        scope=lm.ATOMIC_SCOPE_SYSTEM,
                    )

            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # The source call disables the rendezvous prologue because the
            # scatter grid_sync already completed, then performs the exact
            # stateful one-warp NVLink protocol and its epilogue grid_sync.
            if lm.logical_and(lm.bid == 0, lm.warp_id == 0):
                lm.nvlink_rendezvous(
                    counter=(
                        workspace_u32
                        + layout.rendezvous_counter_offset // 4
                    ),
                    signal_index=(
                        layout.rendezvous_signal_offset
                        - layout.rendezvous_counter_offset
                    )
                    // 4,
                    sym_buf=sym_buffer,
                    signal_offset=layout.rendezvous_signal_offset,
                    group=pg,
                )

            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )
            with lm.elected_thread(warp=0):
                lm.grid_sync_workspace(
                    counter=(
                        workspace_u32
                        + layout.grid_sync_offset // 4
                        + DISPATCH_GRID_SYNC_INDEX
                    ),
                    num_ctas=NUM_CTAS,
                    leader=lm.bid == 0,
                )
            lm.sync(
                bar_id=DISPATCH_BARRIER_ID,
                threads=DISPATCH_THREADS,
                aligned=True,
            )

            # First source dispatch/epilogue handoff.  Dispatch proceeds into
            # token/SF pulls while math consumes completed pool blocks.
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=dispatch_with_math_threads,
            )

            # Token/SF pull. Cache each expert's completed receive count with
            # the exact volatile u64 high32 readiness poll (78 CTAs * 8 ranks).
            if lm.warp_id < active_dispatch_warps:
                ready0: lm.u64 = 0
                ready1: lm.u64 = 0
                count0: lm.u32 = 0
                count1: lm.u32 = 0
                if lm.lane_id < experts_per_rank:
                    while (ready0 >> 32).u32() != NUM_CTAS * NUM_RANKS:
                        ready0 = lm.sys_volatile_load(
                            recv_sum_u64,
                            index=lm.lane_id,
                            dtype=lm.u64,
                        )
                    count0 = ready0.u32()
                if experts_per_rank > 32:
                    if lm.lane_id + 32 < experts_per_rank:
                        while (ready1 >> 32).u32() != NUM_CTAS * NUM_RANKS:
                            ready1 = lm.sys_volatile_load(
                                recv_sum_u64,
                                index=lm.lane_id + 32,
                                dtype=lm.u64,
                            )
                        count1 = ready1.u32()
                lm.sync_warp()

                current_expert: lm.i32 = -1
                expert_start: lm.u32 = 0
                expert_end: lm.u32 = 0
                expert_pool_block_offset: lm.u32 = 0
                stored_rank_count: lm.u32 = 0
                pull_token_idx: lm.u32 = (
                    lm.bid * active_dispatch_warps + lm.warp_id
                ).u32()
                while True:
                    old_expert: lm.i32 = current_expert
                    while pull_token_idx >= expert_end:
                        current_expert = current_expert + 1
                        if current_expert >= experts_per_rank:
                            break
                        expert_pool_block_offset = (
                            expert_pool_block_offset
                            + (expert_end - expert_start + schedule.block_m - 1)
                            // schedule.block_m
                        )
                        expert_start = expert_end
                        lane_count: lm.u32 = count0
                        if current_expert >= 32:
                            lane_count = count1
                        current_count: lm.u32 = lm.shfl_sync(
                            lane_count,
                            current_expert % 32,
                            dtype=lm.u32,
                        )
                        expert_end = expert_end + current_count
                    if current_expert >= experts_per_rank:
                        break

                    if lm.logical_and(
                        old_expert != current_expert,
                        lm.lane_id < NUM_RANKS,
                    ):
                        rank_count_vec = lm.vec_load(
                            recv_count_u64,
                            index=(
                                lm.lane_id * experts_per_rank
                                + current_expert
                            ),
                            count=1,
                            dtype=lm.u64,
                            dst_dtype=lm.u64,
                        )
                        stored_rank_count = rank_count_vec[0].u32()

                    remaining: lm.u32 = stored_rank_count
                    round_offset: lm.u32 = 0
                    token_idx_in_expert: lm.u32 = (
                        pull_token_idx - expert_start
                    )
                    slot_idx: lm.u32 = token_idx_in_expert
                    selected_rank: lm.u32 = 0
                    token_idx_in_rank: lm.u32 = 0
                    while True:
                        lane_active: lm.u32 = 0
                        lane_min: lm.u32 = 0xFFFFFFFF
                        if remaining > 0:
                            lane_active = 1
                            lane_min = remaining
                        num_active_ranks: lm.u32 = lm.warp_redux_u32(
                            lane_active,
                            op=lm.REDUCE_ADD,
                        )
                        round_length: lm.u32 = lm.warp_redux_u32(
                            lane_min,
                            op=lm.REDUCE_MIN,
                        )
                        num_round_tokens: lm.u32 = (
                            round_length * num_active_ranks
                        )
                        if slot_idx < num_round_tokens:
                            slot_in_round: lm.u32 = slot_idx % num_active_ranks
                            active_mask: lm.u32 = lm.warp_vote(
                                vote=lm.VOTE_BALLOT,
                                predicate=remaining > 0,
                                mask=0xFFFFFFFF,
                            )
                            num_active_lanes: lm.i32 = lm.popc(active_mask)
                            if slot_in_round < num_active_lanes:
                                selected_rank = lm.fns(
                                    active_mask,
                                    0,
                                    slot_in_round + 1,
                                )
                            token_idx_in_rank = (
                                round_offset + slot_idx // num_active_ranks
                            )
                            break
                        slot_idx = slot_idx - num_round_tokens
                        round_offset = round_offset + round_length
                        if remaining > round_length:
                            remaining = remaining - round_length
                        else:
                            remaining = 0

                    src_index: lm.u64 = (
                        current_expert
                        * (NUM_RANKS * max_recv_tokens_per_expert)
                        + selected_rank * max_recv_tokens_per_expert
                        + token_idx_in_rank
                    ).u64()
                    src_token_topk_vec = lm.vec_load(
                        src_token_topk_i32,
                        index=src_index,
                        count=1,
                        dtype=lm.i32,
                        dst_dtype=lm.i32,
                    )
                    src_token_topk_idx: lm.u32 = src_token_topk_vec[0].u32()
                    src_token_idx: lm.u32 = src_token_topk_idx // TOP_K
                    src_topk_idx: lm.u32 = src_token_topk_idx % TOP_K
                    pool_token_idx: lm.u32 = (
                        expert_pool_block_offset * schedule.block_m
                        + token_idx_in_expert
                    )
                    pull_smem_addr: lm.i32 = (
                        pull_stage.addr + lm.warp_id * schedule.hidden
                    )

                    with lm.elected_thread():
                        lm.arrive_expect_tx(
                            pull_full,
                            stage=lm.warp_id,
                            tx_bytes=schedule.hidden,
                        )
                        lm.nvlink_pull(
                            src_buf=sym_buffer,
                            src_rank=selected_rank,
                            src_offset=(
                                layout.input_token_offset
                                + src_token_idx.u64() * schedule.hidden
                            ),
                            dst=pull_smem_addr,
                            count=schedule.hidden,
                            barrier=pull_full,
                            stage=lm.warp_id,
                            cache_hint="evict_first",
                        )
                    lm.wait(pull_full, stage=lm.warp_id)
                    lm.fence_proxy_shared_cta()

                    remote_bytes = lm.byte_ptr(
                        sym_buffer.peer_ptr(selected_rank)
                    )
                    remote_sf_f32 = lm.ptr_to(
                        remote_bytes + layout.input_sf_offset,
                        lm.f32,
                    )
                    for sf_group in range(
                        0,
                        (schedule.hidden // 128 + 31) // 32,
                    ):
                        sf_idx: lm.i32 = sf_group * 32 + lm.lane_id
                        if sf_idx < schedule.hidden // 128:
                            sf_value = lm.vec_load(
                                remote_sf_f32,
                                index=(
                                    src_token_idx * (schedule.hidden // 128)
                                    + sf_idx
                                ),
                                count=1,
                                dtype=lm.f32,
                                dst_dtype=lm.f32,
                            )
                            lm.gmem_store(
                                l1_sf_f32,
                                sf_value[0],
                                index=(
                                    sf_idx * schedule.padded_sf_tokens
                                    + pool_token_idx
                                ),
                                dtype=lm.f32,
                                src_dtype=lm.f32,
                            )
                    lm.sync_warp()

                    with lm.elected_thread():
                        remote_weight_f32 = lm.ptr_to(
                            remote_bytes + layout.input_topk_weights_offset,
                            lm.f32,
                        )
                        weight = lm.vec_load(
                            remote_weight_f32,
                            index=src_token_topk_idx,
                            count=1,
                            dtype=lm.f32,
                            dst_dtype=lm.f32,
                        )
                        lm.gmem_store(
                            l1_topk_weights_f32,
                            weight[0],
                            index=pool_token_idx,
                            dtype=lm.f32,
                            src_dtype=lm.f32,
                        )
                        lm.bulk_store(
                            dst=l1_token_u8 + pool_token_idx * schedule.hidden,
                            src=pull_smem_addr,
                            bytes=schedule.hidden,
                            cache_hint="evict_normal",
                        )
                        metadata_idx: lm.u64 = pool_token_idx.u64() * 3
                        lm.gmem_store(
                            token_src_metadata_i32,
                            selected_rank.i32(),
                            index=metadata_idx,
                            dtype=lm.i32,
                            src_dtype=lm.i32,
                        )
                        lm.gmem_store(
                            token_src_metadata_i32,
                            src_token_idx.i32(),
                            index=metadata_idx + 1,
                            dtype=lm.i32,
                            src_dtype=lm.i32,
                        )
                        lm.gmem_store(
                            token_src_metadata_i32,
                            src_topk_idx.i32(),
                            index=metadata_idx + 2,
                            dtype=lm.i32,
                            src_dtype=lm.i32,
                        )
                        lm.bulk_commit()
                        lm.bulk_wait(n=0)
                        lm.gmem_reduction_release(
                            l1_arrival_u32,
                            1,
                            index=(
                                expert_pool_block_offset
                                + token_idx_in_expert // schedule.block_m
                            ),
                            op=lm.REDUCE_ADD,
                            dtype=lm.u32,
                            scope=lm.FENCE_DEVICE,
                        )
                    lm.sync_warp()
                    pull_token_idx = (
                        pull_token_idx
                        + NUM_CTAS * active_dispatch_warps
                    )

            if use_team_slice3:
                # After the token/SF pulls both dispatch warps become the
                # weight decode team for every published task (both phases).
                _emit_team_slice3_loop(
                    lm,
                    task_info_full=task_info_full,
                    task_info_empty=task_info_empty,
                    task_info_smem=task_info_smem,
                    task_info_pipe=task_info_pipe,
                    full_barrier=l1_stage_full,
                    decoded_barrier=l1_stage_decoded,
                    pipe=l1_pipe,
                    packed_smem=l1_packed_smem,
                    decoded_smem=l1_team_decoded_smem,
                    lut_smem=l1_lut_smem,
                )

            if uses_interleaved_scheduler:
                # The second source dispatch/math handoff follows completed
                # peer scatter and the math-side pre-combine rendezvous.
                lm.sync(
                    bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                    threads=dispatch_with_math_threads,
                )

                # Production-wrapper cleanup specialization.  The wrapper
                # passes nullptr for cumulative_local_expert_recv_stats, so
                # only the rank-visible workspace resets remain live.
                if lm.bid == 0:
                    for expert in lm.range(
                        lm.tid,
                        schedule.experts,
                        DISPATCH_THREADS,
                        unroll=1,
                        dtype=lm.i32,
                    ):
                        lm.gmem_store(
                            send_count_u64,
                            0,
                            index=expert,
                            dtype=lm.u64,
                            src_dtype=lm.u64,
                        )
                    if lm.tid == 0:
                        lm.gmem_store(
                            l1_task_count_i32,
                            0,
                            dtype=lm.i32,
                            src_dtype=lm.i32,
                        )
                        lm.gmem_store(
                            l2_task_count_i32,
                            0,
                            dtype=lm.i32,
                            src_dtype=lm.i32,
                        )
                else:
                    cleanup_local_expert: lm.i32 = lm.bid - 1
                    if cleanup_local_expert < experts_per_rank:
                        cleanup_recv = lm.vec_load(
                            recv_sum_u64,
                            index=cleanup_local_expert,
                            count=1,
                            dtype=lm.u64,
                            dst_dtype=lm.u64,
                        )
                        cleanup_num_tokens: lm.u32 = cleanup_recv[0].u32()
                        cleanup_num_m_blocks: lm.u32 = (
                            cleanup_num_tokens + schedule.block_m - 1
                        ) // schedule.block_m

                        # scheduler.get_pool_block_offset(expert): each lane
                        # sums its cached experts and the warp reduces them.
                        cleanup_prefix_lane: lm.u32 = 0
                        for expert_group in lm.range(
                            0,
                            experts_per_lane,
                            constexpr=True,
                        ):
                            prefix_expert: lm.i32 = (
                                expert_group * 32 + lm.lane_id
                            )
                            if prefix_expert < cleanup_local_expert:
                                prefix_recv = lm.vec_load(
                                    recv_sum_u64,
                                    index=prefix_expert,
                                    count=1,
                                    dtype=lm.u64,
                                    dst_dtype=lm.u64,
                                )
                                prefix_num_tokens: lm.u32 = (
                                    prefix_recv[0].u32()
                                )
                                cleanup_prefix_lane = (
                                    cleanup_prefix_lane
                                    + (
                                        prefix_num_tokens
                                        + schedule.block_m
                                        - 1
                                    )
                                    // schedule.block_m
                                )
                        cleanup_pool_block_offset: lm.u32 = lm.warp_redux_u32(
                            cleanup_prefix_lane,
                            op=lm.REDUCE_ADD,
                        )

                        lm.sync(
                            bar_id=DISPATCH_BARRIER_ID,
                            threads=DISPATCH_THREADS,
                            aligned=True,
                        )
                        if lm.warp_id == 0:
                            lm.gmem_store(
                                recv_sum_u64,
                                0,
                                index=cleanup_local_expert,
                                dtype=lm.u64,
                                src_dtype=lm.u64,
                            )
                        elif lm.warp_id == 1:
                            # The source keeps this warp sync after the
                            # optional cumulative-stat update.  The update is
                            # dead for the pinned nullptr wrapper argument.
                            lm.sync_warp()

                        if lm.tid < NUM_RANKS:
                            lm.gmem_store(
                                recv_count_u64,
                                0,
                                index=(
                                    lm.tid * experts_per_rank
                                    + cleanup_local_expert
                                ),
                                dtype=lm.u64,
                                src_dtype=lm.u64,
                            )
                        lm.sync_warp()

                        for cleanup_block in lm.range(
                            lm.tid,
                            cleanup_num_m_blocks,
                            DISPATCH_THREADS,
                            unroll=1,
                            dtype=lm.i32,
                        ):
                            cleanup_pool_block: lm.u32 = (
                                cleanup_pool_block_offset + cleanup_block
                            )
                            lm.gmem_store(
                                l1_arrival_i32,
                                0,
                                index=cleanup_pool_block,
                                dtype=lm.i32,
                                src_dtype=lm.i32,
                            )
                            lm.gmem_store(
                                l2_arrival_u64,
                                0,
                                index=cleanup_pool_block,
                                dtype=lm.u64,
                                src_dtype=lm.u64,
                            )
                        lm.sync_warp()

                # kAfterWorkspaceCleanBarrierTag: source enables only the
                # dispatch-grid prologue around its third rank rendezvous.
                lm.sync(
                    bar_id=DISPATCH_BARRIER_ID,
                    threads=DISPATCH_THREADS,
                    aligned=True,
                )
                with lm.elected_thread(warp=0):
                    lm.grid_sync_workspace(
                        counter=(
                            workspace_u32
                            + layout.grid_sync_offset // 4
                            + DISPATCH_GRID_SYNC_INDEX
                        ),
                        num_ctas=NUM_CTAS,
                        leader=lm.bid == 0,
                    )
                lm.sync(
                    bar_id=DISPATCH_BARRIER_ID,
                    threads=DISPATCH_THREADS,
                    aligned=True,
                )
                if lm.logical_and(lm.bid == 0, lm.warp_id == 0):
                    lm.nvlink_rendezvous(
                        counter=(
                            workspace_u32
                            + layout.rendezvous_counter_offset // 4
                        ),
                        signal_index=(
                            layout.rendezvous_signal_offset
                            - layout.rendezvous_counter_offset
                        )
                        // 4,
                        sym_buf=sym_buffer,
                        signal_offset=layout.rendezvous_signal_offset,
                        group=pg,
                    )

        with loader:
            lm.setmaxnreg_dealloc(loader_registers)
            l1_load_stage: lm.u32 = 0
            if uses_interleaved_scheduler:
                # The source's weight-loader warp is the sole global task
                # producer. Retain its receive-count cache, L1 warm-up/L2
                # alternating claim state machine, and exact task payload
                # construction in the same kernel as dispatch publication.
                if lm.warp_id == 3:
                    stored_num_tokens: lm.u32[experts_per_lane]
                    for expert_group in lm.range(
                        0,
                        experts_per_lane,
                        constexpr=True,
                    ):
                        ready_status: lm.u64 = 0
                        local_expert: lm.i32 = (
                            expert_group * 32 + lm.lane_id
                        )
                        if local_expert < experts_per_rank:
                            while (
                                ready_status >> 32
                            ).u32() != NUM_CTAS * NUM_RANKS:
                                ready_status = lm.sys_volatile_load(
                                    recv_sum_u64,
                                    index=local_expert,
                                    dtype=lm.u64,
                                )
                        stored_num_tokens[expert_group] = ready_status.u32()
                    lm.sync_warp()

                    lane_num_m_blocks: lm.u32 = 0
                    for expert_group in lm.range(
                        0,
                        experts_per_lane,
                        constexpr=True,
                    ):
                        local_expert: lm.i32 = (
                            expert_group * 32 + lm.lane_id
                        )
                        if local_expert < experts_per_rank:
                            lane_num_m_blocks = (
                                lane_num_m_blocks
                                + (
                                    stored_num_tokens[expert_group]
                                    + schedule.block_m
                                    - 1
                                )
                                // schedule.block_m
                            )
                    num_total_m_blocks: lm.u32 = lm.warp_redux_u32(
                        lane_num_m_blocks,
                        op=lm.REDUCE_ADD,
                    )
                    num_total_l1_tasks: lm.u32 = (
                        num_total_m_blocks * l1_n_blocks
                    )
                    num_total_l1_waves: lm.u32 = (
                        num_total_l1_tasks + NUM_CTAS - 1
                    ) // NUM_CTAS
                    num_l1_warmup_waves: lm.u32 = 0
                    if num_total_m_blocks > 0:
                        num_l1_warmup_waves = lm.min(
                            min_l1_warmup_waves,
                            num_total_l1_waves,
                        )

                    l1_waves_done: lm.u32 = 0xFFFFFFFF
                    producer_task_stage: lm.u32 = 0
                    task_phase: lm.u32 = 0
                    task_idx: lm.u32 = 0
                    task_num_n_blocks: lm.u32 = 0
                    task_shape_n: lm.u32 = 0
                    task_shape_k: lm.u32 = 0
                    task_pool_block: lm.u32 = 0
                    task_local_expert: lm.u32 = 0
                    task_m_block: lm.u32 = 0
                    task_n_block: lm.u32 = 0
                    task_valid_m: lm.u32 = 0

                    # Source produce_interleaved_blocks: reserve one mailbox
                    # stage, finish claim_next_task (including its internal
                    # continue), publish the resulting TaskInfo, and publish
                    # one final all-zero/None payload before exiting.
                    while True:
                        lm.wait(
                            task_info_empty,
                            stage=producer_task_stage,
                        )
                        while True:
                            task_phase = 0
                            task_idx = 0
                            task_num_n_blocks = 0
                            task_shape_n = 0
                            task_shape_k = 0
                            task_pool_block = 0
                            task_local_expert = 0
                            task_m_block = 0
                            task_n_block = 0
                            task_valid_m = 0

                            if lm.logical_and(
                                num_l1_warmup_waves != l1_waves_done,
                                num_l1_warmup_waves > 0,
                            ):
                                num_l1_warmup_waves = (
                                    num_l1_warmup_waves - 1
                                )
                                if lm.lane_id == 0:
                                    task_idx = lm.atomic_fetch_add(
                                        l1_task_count_u32,
                                        1,
                                        dtype=lm.u32,
                                        scope=lm.ATOMIC_SCOPE_GPU,
                                    )
                                task_idx = lm.shfl_sync(
                                    task_idx,
                                    0,
                                    dtype=lm.u32,
                                )
                                if task_idx >= num_total_l1_tasks:
                                    num_l1_warmup_waves = l1_waves_done
                                    continue
                                task_phase = 1
                                task_num_n_blocks = l1_n_blocks
                                task_shape_n = l1_shape_n
                                task_shape_k = l1_shape_k
                            else:
                                if lm.lane_id == 0:
                                    task_idx = lm.atomic_fetch_add(
                                        l2_task_count_u32,
                                        1,
                                        dtype=lm.u32,
                                        scope=lm.ATOMIC_SCOPE_GPU,
                                    )
                                task_idx = lm.shfl_sync(
                                    task_idx,
                                    0,
                                    dtype=lm.u32,
                                )
                                if (
                                    task_idx
                                    >= num_total_m_blocks * l2_n_blocks
                                ):
                                    break
                                if num_l1_warmup_waves != l1_waves_done:
                                    num_l1_warmup_waves = 1
                                task_phase = 2
                                task_num_n_blocks = l2_n_blocks
                                task_shape_n = l2_shape_n
                                task_shape_k = l2_shape_k
                                required_l1_tasks: lm.u32 = (
                                    task_idx // l2_n_blocks + 1
                                ) * l1_n_blocks
                                seen_l1_tasks: lm.u32 = 0
                                while seen_l1_tasks < required_l1_tasks:
                                    seen_l1_tasks = lm.sys_volatile_load(
                                        l1_task_count_u32,
                                        dtype=lm.u32,
                                    )

                            task_pool_block = task_idx // task_num_n_blocks
                            task_n_block = task_idx % task_num_n_blocks
                            block_offset: lm.u32 = 0

                            for expert_group in lm.range(
                                0,
                                experts_per_lane,
                                constexpr=True,
                            ):
                                expert_idx: lm.u32 = (
                                    expert_group * 32 + lm.lane_id
                                ).u32()
                                expert_tokens: lm.u32 = (
                                    stored_num_tokens[expert_group]
                                )
                                expert_m_blocks: lm.u32 = (
                                    expert_tokens + schedule.block_m - 1
                                ) // schedule.block_m
                                inclusive_m_blocks: lm.u32 = expert_m_blocks
                                for scan_delta in (1, 2, 4, 8, 16):
                                    scan_peer: lm.u32 = lm.shfl_up_sync(
                                        inclusive_m_blocks,
                                        scan_delta,
                                        dtype=lm.u32,
                                    )
                                    if lm.lane_id >= scan_delta:
                                        inclusive_m_blocks = (
                                            inclusive_m_blocks + scan_peer
                                        )
                                lane_pool_offset: lm.u32 = (
                                    block_offset
                                    + inclusive_m_blocks
                                    - expert_m_blocks
                                )
                                is_owner = lm.logical_and(
                                    expert_idx < experts_per_rank,
                                    lm.logical_and(
                                        task_pool_block >= lane_pool_offset,
                                        task_pool_block
                                        < lane_pool_offset + expert_m_blocks,
                                    ),
                                )
                                owner_mask: lm.u32 = lm.warp_vote(
                                    vote=lm.VOTE_BALLOT,
                                    predicate=is_owner,
                                    mask=0xFFFFFFFF,
                                )
                                if owner_mask != 0:
                                    owner_lane: lm.i32 = (
                                        lm.ffs(owner_mask) - 1
                                    )
                                    owner_m_block: lm.u32 = (
                                        task_pool_block - lane_pool_offset
                                    )
                                    owner_valid_m: lm.u32 = lm.min(
                                        expert_tokens
                                        - owner_m_block * schedule.block_m,
                                        schedule.block_m,
                                    )
                                    task_local_expert = lm.shfl_sync(
                                        expert_idx,
                                        owner_lane,
                                        dtype=lm.u32,
                                    )
                                    task_m_block = lm.shfl_sync(
                                        owner_m_block,
                                        owner_lane,
                                        dtype=lm.u32,
                                    )
                                    task_valid_m = lm.shfl_sync(
                                        owner_valid_m,
                                        owner_lane,
                                        dtype=lm.u32,
                                    )
                                block_offset = (
                                    block_offset
                                    + lm.shfl_sync(
                                        inclusive_m_blocks,
                                        31,
                                        dtype=lm.u32,
                                    )
                                )
                            break

                        with lm.elected_thread():
                            task_info_words: lm.u32[8]
                            task_info_words[0] = task_phase
                            task_info_words[1] = task_local_expert
                            task_info_words[2] = task_m_block
                            task_info_words[3] = task_n_block
                            task_info_words[4] = task_pool_block
                            task_info_words[5] = task_valid_m
                            task_info_words[6] = task_shape_n
                            task_info_words[7] = task_shape_k
                            for task_word in lm.range(
                                0,
                                TASK_INFO_FIELDS,
                                constexpr=True,
                            ):
                                lm.smem_store_vec(
                                    dst_addr=(
                                        task_info_smem.stage_addr(
                                            producer_task_stage
                                        )
                                        + task_word * 4
                                    ),
                                    src=task_info_words[task_word],
                                )
                            lm.threadfence(scope=lm.FENCE_BLOCK)
                            lm.arrive(
                                task_info_full,
                                stage=producer_task_stage,
                            )
                        lm.sync_warp()
                        lm.advance(producer_task_stage, task_info_pipe)
                        if task_phase != 0:
                            if uses_interleaved_scheduler:
                                # Source produce_interleaved_blocks invokes
                                # load_b_task immediately after publication.
                                # One persistent cursor spans L1 and L2 tasks;
                                # the phase selects the corresponding packed
                                # weight descriptor without changing the
                                # pipeline topology.
                                for k_block_idx in lm.range(
                                    0,
                                    task_shape_k // RS_BLOCK_K,
                                    unroll=1,
                                    dtype=lm.i32,
                                ):
                                    lm.wait(
                                        l1_stage_empty,
                                        stage=l1_load_stage,
                                    )
                                    with lm.elected_thread():
                                        if task_phase == 1:
                                            lm.tma_load(
                                                l1_packed_weights,
                                                dst=(
                                                    l1_packed_smem.stage_addr(
                                                        l1_load_stage
                                                    )
                                                ),
                                                coords=(
                                                    k_block_idx
                                                    * RS_WEIGHT_ROW_BYTES,
                                                    task_local_expert
                                                    * task_shape_n
                                                    + task_n_block
                                                    * RS_WEIGHT_ROWS,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                        else:
                                            lm.tma_load(
                                                l2_packed_weights,
                                                dst=(
                                                    l1_packed_smem.stage_addr(
                                                        l1_load_stage
                                                    )
                                                ),
                                                coords=(
                                                    k_block_idx
                                                    * RS_WEIGHT_ROW_BYTES,
                                                    task_local_expert
                                                    * task_shape_n
                                                    + task_n_block
                                                    * RS_WEIGHT_ROWS,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                        lm.arrive_expect_tx(
                                            l1_stage_full,
                                            tx_bytes=l1_packed_stage_bytes,
                                            stage=l1_load_stage,
                                        )
                                    lm.sync_warp()
                                    lm.advance(l1_load_stage, l1_pipe)
                        if task_phase == 0:
                            break
                else:
                    # The activation-loader consumer owns an independent
                    # mailbox cursor and exits only after reading the final
                    # invalid TaskInfo. It deliberately does not recycle the
                    # slot; the eight math warps own that source count.
                    loader_task_stage: lm.u32 = 0
                    loader_task_phase: lm.u32 = 0
                    while True:
                        lm.wait(task_info_full, stage=loader_task_stage)
                        loader_task_words: lm.u32[8]
                        lm.smem_load_vec(
                            dst=loader_task_words,
                            src_addr=task_info_smem.stage_addr(
                                loader_task_stage
                            ),
                            count=4,
                        )
                        lm.smem_load_vec(
                            dst=loader_task_words,
                            src_addr=(
                                task_info_smem.stage_addr(loader_task_stage)
                                + 16
                            ),
                            count=4,
                            dst_offset=4,
                        )
                        loader_task_phase = loader_task_words[0]
                        lm.advance(loader_task_stage, task_info_pipe)
                        if loader_task_phase == 0:
                            break
                        if uses_interleaved_scheduler:
                            loader_task_pool_block: lm.u32 = (
                                loader_task_words[4]
                            )
                            loader_task_valid_m: lm.u32 = (
                                loader_task_words[5]
                            )
                            loader_task_shape_k: lm.u32 = (
                                loader_task_words[7]
                            )
                            if loader_task_valid_m > 0:
                                if loader_task_phase == 1:
                                    lm.gmem_wait_acquire(
                                        l1_arrival_u32,
                                        loader_task_valid_m,
                                        index=loader_task_pool_block,
                                        dtype=lm.u32,
                                        predicate=lm.WAIT_EQ,
                                        scope=lm.FENCE_DEVICE,
                                    )
                                else:
                                    lm.gmem_wait_acquire(
                                        l2_arrival_u64,
                                        l2_expected_arrival_mask,
                                        index=loader_task_pool_block,
                                        dtype=lm.u64,
                                        predicate=lm.WAIT_EQ,
                                        scope=lm.FENCE_DEVICE,
                                    )
                            for k_block_idx in lm.range(
                                0,
                                loader_task_shape_k // RS_BLOCK_K,
                                unroll=1,
                                dtype=lm.i32,
                            ):
                                lm.wait(
                                    l1_stage_empty,
                                    stage=l1_load_stage,
                                )
                                if loader_task_valid_m > 0:
                                    with lm.elected_thread():
                                        if loader_task_phase == 1:
                                            lm.tma_load(
                                                l1_activation,
                                                dst=l1_act_smem.stage_addr(
                                                    l1_load_stage
                                                ),
                                                coords=(
                                                    k_block_idx * RS_BLOCK_K,
                                                    loader_task_pool_block
                                                    * schedule.block_m,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                            lm.tma_load(
                                                l1_activation_scale,
                                                dst=l1_sfa_smem.stage_addr(
                                                    l1_load_stage
                                                ),
                                                coords=(
                                                    loader_task_pool_block
                                                    * schedule.block_m,
                                                    k_block_idx,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                        else:
                                            lm.tma_load(
                                                l2_activation,
                                                dst=l1_act_smem.stage_addr(
                                                    l1_load_stage
                                                ),
                                                coords=(
                                                    k_block_idx * RS_BLOCK_K,
                                                    loader_task_pool_block
                                                    * schedule.block_m,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                            lm.tma_load(
                                                l2_activation_scale,
                                                dst=l1_sfa_smem.stage_addr(
                                                    l1_load_stage
                                                ),
                                                coords=(
                                                    loader_task_pool_block
                                                    * schedule.block_m,
                                                    k_block_idx,
                                                ),
                                                barrier=l1_stage_full,
                                                stage=l1_load_stage,
                                            )
                                        lm.arrive_expect_tx(
                                            l1_stage_full,
                                            tx_bytes=(
                                                l1_act_stage_bytes
                                                + l1_sfa_tile_bytes
                                            ),
                                            stage=l1_load_stage,
                                        )
                                else:
                                    with lm.elected_thread():
                                        lm.arrive(
                                            l1_stage_full,
                                            stage=l1_load_stage,
                                        )
                                lm.sync_warp()
                                lm.advance(l1_load_stage, l1_pipe)

        with math:
            lm.setmaxnreg_alloc(math_registers)
            lm.sync(
                bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                threads=dispatch_with_math_threads,
            )

            # Each math warp owns an independent published-task cursor. Both
            # phases enter one persistent loader pipeline, so a mailbox stage
            # is released only after its first full GEMM stage is visible.
            # Phase 1 publishes FP8; phase 2 converts scaled accumulators to
            # BF16 and scatters them to the owning peer.
            math_task_stage: lm.u32 = 0
            l1_math_stage: lm.u32 = 0
            math_task_phase: lm.u32 = 0
            while True:
                lm.wait(task_info_full, stage=math_task_stage)
                consumed_task_stage: lm.u32 = math_task_stage
                math_task_words: lm.u32[8]
                lm.smem_load_vec(
                    dst=math_task_words,
                    src_addr=task_info_smem.stage_addr(math_task_stage),
                    count=4,
                )
                lm.smem_load_vec(
                    dst=math_task_words,
                    src_addr=(
                        task_info_smem.stage_addr(math_task_stage) + 16
                    ),
                    count=4,
                    dst_offset=4,
                )
                math_task_phase = math_task_words[0]
                lm.advance(math_task_stage, task_info_pipe)
                if math_task_phase == 0:
                    break
                if is_swap_ab:
                    if math_task_phase != 0:
                        math_task_valid_m: lm.i32 = (
                            math_task_words[5].i32()
                        )
                        math_task_shape_k: lm.i32 = (
                            math_task_words[7].i32()
                        )
                        final_accum: lm.f32[64]
                        final_accum.fill_(0.0)
                        if RS_KBLOCK_ROTATE:
                            math_num_k_blocks: lm.i32 = (
                                math_task_shape_k // RS_BLOCK_K
                            )
                            rotated_kwargs = dict(
                                num_k_blocks=math_num_k_blocks,
                                stage_var=l1_math_stage,
                                pipe=l1_pipe,
                                full_barrier=l1_stage_full,
                                empty_barrier=l1_stage_empty,
                                task_info_empty=task_info_empty,
                                consumed_task_stage=consumed_task_stage,
                                act_smem=l1_act_smem,
                                packed_smem=l1_packed_smem,
                                sfa_smem=l1_sfa_smem,
                                lut_smem=l1_lut_smem,
                                valid_m=math_task_valid_m,
                                final_accum=final_accum,
                                team_slice3=use_team_slice3,
                                decoded_smem=l1_team_decoded_smem,
                                decoded_barrier=l1_stage_decoded,
                                wide=use_wide_math,
                                lutc_smem=l1_lutc_smem,
                            )
                            if is_block_m8:
                                _emit_rs_task(lm, n_swap=8, **rotated_kwargs)
                            elif is_block_m16:
                                rot_n_swap: lm.i32 = (
                                    (math_task_valid_m + 7) // 8
                                ) * 8
                                if rot_n_swap <= 8:
                                    _emit_rs_task(lm, n_swap=8, **rotated_kwargs)
                                else:
                                    _emit_rs_task(lm, n_swap=16, **rotated_kwargs)
                            else:
                                rot_n_swap: lm.i32 = (
                                    (math_task_valid_m + 7) // 8
                                ) * 8
                                if rot_n_swap <= 8:
                                    _emit_rs_task(lm, n_swap=8, **rotated_kwargs)
                                elif rot_n_swap <= 16:
                                    _emit_rs_task(lm, n_swap=16, **rotated_kwargs)
                                else:
                                    _emit_rs_task(lm, n_swap=24, **rotated_kwargs)
                        if not RS_KBLOCK_ROTATE:
                            for math_k_block_idx in lm.range(
                                0,
                                math_task_shape_k // RS_BLOCK_K,
                                unroll=1,
                                dtype=lm.i32,
                            ):
                                lm.wait(
                                    l1_stage_full,
                                    stage=l1_math_stage,
                                )
                                if math_k_block_idx == 0:
                                    with lm.elected_thread():
                                        lm.arrive(
                                            task_info_empty,
                                            stage=consumed_task_stage,
                                        )
                                lm.fence_proxy_shared_cta()

                                if is_block_m8:
                                    _emit_l1_rs_stage(
                                        lm,
                                        n_swap=8,
                                        act_stage=l1_act_smem[l1_math_stage],
                                        packed_stage_addr=(
                                            l1_packed_smem.stage_addr(
                                                l1_math_stage
                                            )
                                        ),
                                        lut_smem=l1_lut_smem,
                                        sfa_stage_addr=(
                                            l1_sfa_smem.stage_addr(
                                                l1_math_stage
                                            )
                                        ),
                                        valid_m=math_task_valid_m,
                                        final_accum=final_accum,
                                    )
                                elif is_block_m16:
                                    n_swap: lm.i32 = (
                                        (math_task_valid_m + 7) // 8
                                    ) * 8
                                    if n_swap <= 8:
                                        _emit_l1_rs_stage(
                                            lm,
                                            n_swap=8,
                                            act_stage=(
                                                l1_act_smem[l1_math_stage]
                                            ),
                                            packed_stage_addr=(
                                                l1_packed_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            lut_smem=l1_lut_smem,
                                            sfa_stage_addr=(
                                                l1_sfa_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            valid_m=math_task_valid_m,
                                            final_accum=final_accum,
                                        )
                                    else:
                                        _emit_l1_rs_stage(
                                            lm,
                                            n_swap=16,
                                            act_stage=(
                                                l1_act_smem[l1_math_stage]
                                            ),
                                            packed_stage_addr=(
                                                l1_packed_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            lut_smem=l1_lut_smem,
                                            sfa_stage_addr=(
                                                l1_sfa_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            valid_m=math_task_valid_m,
                                            final_accum=final_accum,
                                        )
                                else:
                                    n_swap: lm.i32 = (
                                        (math_task_valid_m + 7) // 8
                                    ) * 8
                                    if n_swap <= 8:
                                        _emit_l1_rs_stage(
                                            lm,
                                            n_swap=8,
                                            act_stage=(
                                                l1_act_smem[l1_math_stage]
                                            ),
                                            packed_stage_addr=(
                                                l1_packed_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            lut_smem=l1_lut_smem,
                                            sfa_stage_addr=(
                                                l1_sfa_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            valid_m=math_task_valid_m,
                                            final_accum=final_accum,
                                        )
                                    elif n_swap <= 16:
                                        _emit_l1_rs_stage(
                                            lm,
                                            n_swap=16,
                                            act_stage=(
                                                l1_act_smem[l1_math_stage]
                                            ),
                                            packed_stage_addr=(
                                                l1_packed_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            lut_smem=l1_lut_smem,
                                            sfa_stage_addr=(
                                                l1_sfa_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            valid_m=math_task_valid_m,
                                            final_accum=final_accum,
                                        )
                                    else:
                                        _emit_l1_rs_stage(
                                            lm,
                                            n_swap=24,
                                            act_stage=(
                                                l1_act_smem[l1_math_stage]
                                            ),
                                            packed_stage_addr=(
                                                l1_packed_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            lut_smem=l1_lut_smem,
                                            sfa_stage_addr=(
                                                l1_sfa_smem.stage_addr(
                                                    l1_math_stage
                                                )
                                            ),
                                            valid_m=math_task_valid_m,
                                            final_accum=final_accum,
                                        )

                                if lm.warp_id < 8:
                                    lm.sync(bar_id=3, threads=128, aligned=True)
                                else:
                                    lm.sync(bar_id=4, threads=128, aligned=True)
                                with lm.elected_thread():
                                    lm.arrive(
                                        l1_stage_empty,
                                        stage=l1_math_stage,
                                    )
                                lm.advance(l1_math_stage, l1_pipe)

                        if math_task_phase == 1:
                            _emit_swap_l1_epilogue(
                                lm,
                                wide=use_wide_math,
                                math_warps=math_warps,
                                block_m=schedule.block_m,
                                final_accum=final_accum,
                                valid_m=math_task_valid_m,
                                local_expert_idx=math_task_words[1].i32(),
                                n_block_idx=math_task_words[3].i32(),
                                pool_block_idx=math_task_words[4].i32(),
                                l1_global_scales=l1_global_scales,
                                l1_topk_weights=l1_topk_weights_f32,
                                l2_sf=l2_sf_f32,
                                l2_arrival_mask=l2_arrival_u64,
                                l1_output=l1_output,
                                l1_output_smem=l1_output_smem,
                                l1_amax_smem=l1_amax_smem,
                                padded_sf_tokens=(
                                    schedule.padded_sf_tokens
                                ),
                            )
                        else:
                            _emit_swap_l2_scatter(
                                lm,
                                wide=use_wide_math,
                                math_warps=math_warps,
                                block_m=schedule.block_m,
                                final_accum=final_accum,
                                valid_m=math_task_valid_m,
                                local_expert_idx=(
                                    math_task_words[1].i32()
                                ),
                                n_block_idx=math_task_words[3].i32(),
                                pool_block_idx=(
                                    math_task_words[4].i32()
                                ),
                                l2_global_scales=l2_global_scales,
                                token_src_metadata=(
                                    token_src_metadata_i32
                                ),
                                sym_buffer=sym_buffer,
                                combine_token_offset=(
                                    layout.combine_token_offset
                                ),
                                hidden=schedule.hidden,
                                l2_output_smem=l2_output_smem,
                            )
                else:
                    if math_task_phase == 1:
                        math_task_valid_m: lm.i32 = (
                            math_task_words[5].i32()
                        )
                        math_task_shape_k: lm.i32 = (
                            math_task_words[7].i32()
                        )
                        final_accum: lm.f32[64]
                        final_accum.fill_(0.0)
                        for math_k_block_idx in lm.range(
                            0,
                            math_task_shape_k // RS_BLOCK_K,
                            unroll=1,
                            dtype=lm.i32,
                        ):
                            lm.wait(
                                l1_stage_full,
                                stage=l1_math_stage,
                            )
                            if math_k_block_idx == 0:
                                with lm.elected_thread():
                                    lm.arrive(
                                        task_info_empty,
                                        stage=consumed_task_stage,
                                    )
                            lm.fence_proxy_shared_cta()
                            _emit_nonswap_stage(
                                lm,
                                act_stage=l1_act_smem[l1_math_stage],
                                packed_stage_addr=(
                                    l1_packed_smem.stage_addr(
                                        l1_math_stage
                                    )
                                ),
                                decoded_stage=(
                                    l1_decoded_smem[l1_math_stage]
                                ),
                                decoded_smem=l1_decoded_smem,
                                decoded_stage_addr=(
                                    l1_decoded_smem.stage_addr(
                                        l1_math_stage
                                    )
                                ),
                                lut_smem=l1_lut_smem,
                                sfa_stage_addr=(
                                    l1_sfa_smem.stage_addr(l1_math_stage)
                                ),
                                empty_barrier=l1_stage_empty,
                                stage=l1_math_stage,
                                final_accum=final_accum,
                            )
                            lm.advance(l1_math_stage, l1_pipe)

                        _emit_l1_nonswap_epilogue(
                            lm,
                            final_accum=final_accum,
                            valid_m=math_task_valid_m,
                            local_expert_idx=math_task_words[1].i32(),
                            n_block_idx=math_task_words[3].i32(),
                            pool_block_idx=math_task_words[4].i32(),
                            l1_global_scales=l1_global_scales,
                            l1_topk_weights=l1_topk_weights_f32,
                            l2_sf=l2_sf_f32,
                            l2_arrival_mask=l2_arrival_u64,
                            l1_output=l1_output,
                            l1_output_smem=l1_output_smem,
                            l1_amax_smem=l1_amax_smem,
                            padded_sf_tokens=schedule.padded_sf_tokens,
                        )
                    else:
                        math_task_valid_m: lm.i32 = (
                            math_task_words[5].i32()
                        )
                        math_task_shape_k: lm.i32 = (
                            math_task_words[7].i32()
                        )
                        final_accum: lm.f32[64]
                        final_accum.fill_(0.0)
                        for math_k_block_idx in lm.range(
                            0,
                            math_task_shape_k // RS_BLOCK_K,
                            unroll=1,
                            dtype=lm.i32,
                        ):
                            lm.wait(
                                l1_stage_full,
                                stage=l1_math_stage,
                            )
                            if math_k_block_idx == 0:
                                with lm.elected_thread():
                                    lm.arrive(
                                        task_info_empty,
                                        stage=consumed_task_stage,
                                    )
                            lm.fence_proxy_shared_cta()
                            _emit_nonswap_stage(
                                lm,
                                act_stage=l1_act_smem[l1_math_stage],
                                packed_stage_addr=(
                                    l1_packed_smem.stage_addr(
                                        l1_math_stage
                                    )
                                ),
                                decoded_stage=(
                                    l1_decoded_smem[l1_math_stage]
                                ),
                                decoded_smem=l1_decoded_smem,
                                decoded_stage_addr=(
                                    l1_decoded_smem.stage_addr(
                                        l1_math_stage
                                    )
                                ),
                                lut_smem=l1_lut_smem,
                                sfa_stage_addr=(
                                    l1_sfa_smem.stage_addr(l1_math_stage)
                                ),
                                empty_barrier=l1_stage_empty,
                                stage=l1_math_stage,
                                final_accum=final_accum,
                            )
                            lm.advance(l1_math_stage, l1_pipe)

                        _emit_l2_nonswap_remote_scatter(
                            lm,
                            final_accum=final_accum,
                            valid_m=math_task_valid_m,
                            local_expert_idx=math_task_words[1].i32(),
                            n_block_idx=math_task_words[3].i32(),
                            pool_block_idx=math_task_words[4].i32(),
                            l2_global_scales=l2_global_scales,
                            token_src_metadata=token_src_metadata_i32,
                            sym_buffer=sym_buffer,
                            combine_token_offset=(
                                layout.combine_token_offset
                            ),
                            hidden=schedule.hidden,
                        )

            if uses_interleaved_scheduler:
                # kBeforeCombineReduceBarrierTag.  The source brackets its
                # second stateful rank rendezvous with epilogue grid_sync on
                # grid counter 1, after every remote BF16 scatter completed.
                lm.sync(
                    bar_id=EPILOGUE_FULL_BARRIER_ID,
                    threads=math_threads,
                    aligned=True,
                )
                with lm.elected_thread(warp=4):
                    lm.grid_sync_workspace(
                        counter=(
                            workspace_u32
                            + layout.grid_sync_offset // 4
                            + EPILOGUE_GRID_SYNC_INDEX
                        ),
                        num_ctas=NUM_CTAS,
                        leader=lm.bid == 0,
                    )
                lm.sync(
                    bar_id=EPILOGUE_FULL_BARRIER_ID,
                    threads=math_threads,
                    aligned=True,
                )
                if lm.logical_and(lm.bid == 0, lm.warp_id == 4):
                    lm.nvlink_rendezvous(
                        counter=(
                            workspace_u32
                            + layout.rendezvous_counter_offset // 4
                        ),
                        signal_index=(
                            layout.rendezvous_signal_offset
                            - layout.rendezvous_counter_offset
                        )
                        // 4,
                        sym_buf=sym_buffer,
                        signal_offset=layout.rendezvous_signal_offset,
                        group=pg,
                    )
                lm.sync(
                    bar_id=EPILOGUE_FULL_BARRIER_ID,
                    threads=math_threads,
                    aligned=True,
                )
                with lm.elected_thread(warp=4):
                    lm.grid_sync_workspace(
                        counter=(
                            workspace_u32
                            + layout.grid_sync_offset // 4
                            + EPILOGUE_GRID_SYNC_INDEX
                        ),
                        num_ctas=NUM_CTAS,
                        leader=lm.bid == 0,
                    )
                lm.sync(
                    bar_id=EPILOGUE_FULL_BARRIER_ID,
                    threads=math_threads,
                    aligned=True,
                )

                # Paired with dispatch immediately before cleanup.  Top-k
                # combine begins immediately after this source boundary.
                lm.sync(
                    bar_id=DISPATCH_WITH_MATH_BARRIER_ID,
                    threads=dispatch_with_math_threads,
                )
                # Source top-k combine.  input_topk_idx is organized as one
                # marker lane per possible source rank; only its sign controls
                # whether the corresponding symmetric top-k slot participates.
                epilogue_warp_idx: lm.i32 = lm.warp_id - 4
                combine_phase: lm.u32 = 0
                combine_load_stage: lm.u32 = 0
                for combine_token_idx in lm.range(
                    _combine_token_start(lm, epilogue_warp_idx, num_tokens, use_wide_math),
                    num_tokens,
                    NUM_CTAS * EPILOGUE_WARPS,
                    unroll=1,
                    dtype=lm.u32,
                ):
                    stored_topk_slot_idx: lm.i32 = -1
                    if lm.lane_id < TOP_K:
                        stored_topk_slot = lm.vec_load(
                            input_topk_idx_i64,
                            index=(
                                combine_token_idx * TOP_K + lm.lane_id
                            ),
                            count=1,
                            dtype=lm.i64,
                            dst_dtype=lm.i64,
                            cache_hint="read_only",
                        )
                        stored_topk_slot_idx = stored_topk_slot[0].i32()
                    total_mask: lm.u32 = lm.warp_vote(
                        vote=lm.VOTE_BALLOT,
                        predicate=stored_topk_slot_idx >= 0,
                        mask=0xFFFFFFFF,
                    )

                    for combine_chunk in lm.range(
                        0,
                        combine_num_chunks,
                        constexpr=True,
                    ):
                        chunk_byte_offset: lm.u32 = (
                            combine_chunk * combine_chunk_bytes
                        )
                        combine_mask: lm.u32 = total_mask
                        do_reduce: lm.i32 = 0
                        if combine_mask != 0:
                            slot_idx: lm.i32 = lm.ffs(combine_mask) - 1
                            combine_mask = (
                                combine_mask ^ (1 << slot_idx)
                            )
                            load_barrier_stage: lm.u32 = (
                                epilogue_warp_idx * 2
                                + combine_load_stage
                            )
                            with lm.elected_thread():
                                lm.arrive_expect_tx(
                                    combine_full,
                                    stage=load_barrier_stage,
                                    tx_bytes=combine_chunk_bytes,
                                )
                                lm.copy(
                                    src=(
                                        sym_bytes
                                        + layout.combine_token_offset
                                        + (
                                            slot_idx.u64()
                                            * MAX_TOKENS_PER_RANK
                                            + combine_token_idx.u64()
                                        )
                                        * schedule.hidden
                                        * 2
                                        + chunk_byte_offset
                                    ),
                                    dst=combine_load_smem.stage_addr(
                                        load_barrier_stage
                                    ),
                                    bytes=combine_chunk_bytes,
                                    barrier=combine_full,
                                    stage=load_barrier_stage,
                                )
                            lm.sync_warp()
                            do_reduce = 1

                        reduced: lm.f32[combine_elems_per_lane]
                        reduced.fill_(0.0)
                        while do_reduce != 0:
                            do_reduce = 0
                            if combine_mask != 0:
                                next_slot_idx: lm.i32 = (
                                    lm.ffs(combine_mask) - 1
                                )
                                combine_mask = (
                                    combine_mask ^ (1 << next_slot_idx)
                                )
                                next_load_stage: lm.u32 = (
                                    combine_load_stage ^ 1
                                )
                                next_barrier_stage: lm.u32 = (
                                    epilogue_warp_idx * 2
                                    + next_load_stage
                                )
                                with lm.elected_thread():
                                    lm.arrive_expect_tx(
                                        combine_full,
                                        stage=next_barrier_stage,
                                        tx_bytes=combine_chunk_bytes,
                                    )
                                    lm.copy(
                                        src=(
                                            sym_bytes
                                            + layout.combine_token_offset
                                            + (
                                                next_slot_idx.u64()
                                                * MAX_TOKENS_PER_RANK
                                                + combine_token_idx.u64()
                                            )
                                            * schedule.hidden
                                            * 2
                                            + chunk_byte_offset
                                        ),
                                        dst=combine_load_smem.stage_addr(
                                            next_barrier_stage
                                        ),
                                        bytes=combine_chunk_bytes,
                                        barrier=combine_full,
                                        stage=next_barrier_stage,
                                    )
                                lm.sync_warp()
                                do_reduce = 1

                            current_barrier_stage: lm.u32 = (
                                epilogue_warp_idx * 2
                                + combine_load_stage
                            )
                            lm.wait(
                                combine_full,
                                stage=current_barrier_stage,
                                phase=combine_phase,
                            )
                            lm.fence_proxy_shared_cta()
                            for combine_vec in lm.range(
                                0,
                                combine_elems_per_lane // 8,
                                constexpr=True,
                            ):
                                combine_packed: lm.u32[4]
                                lm.smem_load_vec(
                                    dst=combine_packed,
                                    src_addr=(
                                        combine_load_smem.stage_addr(
                                            current_barrier_stage
                                        )
                                        + (
                                            combine_vec * 32
                                            + lm.lane_id
                                        )
                                        * 16
                                    ),
                                    count=4,
                                )
                                combine_values = combine_packed.f32()
                                for combine_elem in lm.range(
                                    0,
                                    8,
                                    constexpr=True,
                                ):
                                    reduced[
                                        combine_vec * 8 + combine_elem
                                    ] = (
                                        reduced[
                                            combine_vec * 8 + combine_elem
                                        ]
                                        + combine_values[combine_elem]
                                    )
                            combine_phase = (
                                combine_phase ^ combine_load_stage
                            )
                            combine_load_stage = (
                                combine_load_stage ^ 1
                            )

                        # Wait before reusing the third slot, exactly as the
                        # source's first j iteration waits for the prior store.
                        lm.bulk_wait(n=0)
                        lm.sync_warp()
                        for combine_vec in lm.range(
                            0,
                            combine_elems_per_lane // 8,
                            constexpr=True,
                        ):
                            cast_values: lm.f32[8]
                            for cast_elem in lm.range(
                                0,
                                8,
                                constexpr=True,
                            ):
                                cast_values[cast_elem] = reduced[
                                    combine_vec * 8 + cast_elem
                                ]
                            casted: lm.u32[4]
                            lm.reg_array_pack(
                                cast_values,
                                casted,
                                size=8,
                                dtype=lm.bf16,
                            )
                            for cast_word in lm.range(
                                0,
                                4,
                                constexpr=True,
                            ):
                                lm.smem_store_vec(
                                    dst_addr=(
                                        combine_store_smem.stage_addr(
                                            epilogue_warp_idx
                                        )
                                        + (
                                            combine_vec * 32
                                            + lm.lane_id
                                        )
                                        * 16
                                        + cast_word * 4
                                    ),
                                    src=casted[cast_word],
                                )
                        lm.sync_warp()
                        with lm.elected_thread():
                            lm.fence_proxy_shared_cta()
                            lm.bulk_store(
                                dst=(
                                    output_bf16
                                    + combine_token_idx.u64()
                                    * schedule.hidden
                                    + chunk_byte_offset // 2
                                ),
                                src=combine_store_smem.stage_addr(
                                    epilogue_warp_idx
                                ),
                                bytes=combine_chunk_bytes,
                            )
                            lm.bulk_commit()
                        lm.sync_warp()

    small_rs_dispatch_fragment.__annotations__ = {
        "num_tokens": LM.i32,
        "l1_activation": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_BLOCK_K, schedule.block_m),
            swizzle=LM.swizzle_128b,
            l2_promotion="l2_256b",
        ),
        "l1_activation_scale": LM.tma2d(
            dtype=LM.f32,
            box_shape=(schedule.block_m, 1),
            swizzle=LM.swizzle_none,
            l2_promotion="l2_256b",
        ),
        "l1_packed_weights": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_WEIGHT_ROW_BYTES, RS_WEIGHT_ROWS),
            swizzle=LM.swizzle_none,
            l2_promotion="l2_256b",
        ),
        "l1_output": LM.tma2d(
            dtype=LM.u8,
            box_shape=(128, schedule.block_m),
            swizzle=LM.swizzle_none,
            l2_promotion="l2_256b",
        ),
        "l2_activation": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_BLOCK_K, schedule.block_m),
            swizzle=LM.swizzle_128b,
            l2_promotion="l2_256b",
        ),
        "l2_activation_scale": LM.tma2d(
            dtype=LM.f32,
            box_shape=(schedule.block_m, 1),
            swizzle=LM.swizzle_none,
            l2_promotion="l2_256b",
        ),
        "l2_packed_weights": LM.tma2d(
            dtype=LM.u8,
            box_shape=(RS_WEIGHT_ROW_BYTES, RS_WEIGHT_ROWS),
            swizzle=LM.swizzle_none,
            l2_promotion="l2_256b",
        ),
        "l1_global_scales": LM.ptr[LM.f32],
        "l2_global_scales": LM.ptr[LM.f32],
        "output_bf16": LM.ptr[LM.bf16],
    }
    return loom.weave(
        threads=kernel_threads,
        cluster_dims=CLUSTER_DIMS,
        mbarrier_wait_suspend_ticks=MBARRIER_WAIT_SUSPEND_TICKS,
        # ``%laneid`` is opaque to ptxas; the source-equal ``threadIdx.x % 32``
        # keeps lane-indexed decode/scatter addressing range-analyzable. On H20
        # the opaque form grew the cubin 25% and cost 5-10% on the Pro rows.
        lane_index_source=LaneIndexSource.THREAD_INDEX,
    )(small_rs_dispatch_fragment)


def compile_partial_weave() -> dict:
    """Compile every accumulated inactive small-RS Weave fragment."""

    from loom.codegen.kernel import generate_kernel
    from loom.examples.gemm_common import _cuda_include_dirs
    from loom.runtime.compiler import compile_cuda, detect_gpu_arch

    arch = detect_gpu_arch()
    if arch != "sm_90a":
        raise RuntimeError(f"small-RS partial compile requires sm_90a, got {arch}")

    lut_payload = b"".join(
        word.to_bytes(4, "little") for word in RS_LUT_WORD_VALUES
    )
    actual_lut_sha256 = hashlib.sha256(lut_payload).hexdigest()
    if len(RS_LUT_WORD_VALUES) != RS_LUT_WORDS:
        raise RuntimeError(
            f"small-RS LUT has {len(RS_LUT_WORD_VALUES)} words, "
            f"expected {RS_LUT_WORDS}"
        )
    if actual_lut_sha256 != RS_LUT_PAYLOAD_SHA256:
        raise RuntimeError(
            "small-RS LUT payload mismatch: expected "
            f"{RS_LUT_PAYLOAD_SHA256}, got {actual_lut_sha256}"
        )

    constant_lut_declaration = (
        "static __device__ __constant__ __align__(16) const unsigned int "
        "kE2M1AndUe4m3ToFp8Lut[256] = {"
    )
    constant_lut_read = (
        "reinterpret_cast<const uint4*>(kE2M1AndUe4m3ToFp8Lut)["
    )

    def validate_constant_lut(label, kernel, source):
        argument_names = tuple(argument.name for argument in kernel.args)
        if "lut" in argument_names:
            raise RuntimeError(f"{label} still exposes the LUT as a kernel argument")
        if source.count(constant_lut_declaration) != 1:
            raise RuntimeError(
                f"{label} must emit exactly one typed constant LUT declaration"
            )
        if source.count(constant_lut_read) != 1:
            raise RuntimeError(
                f"{label} must emit exactly one source-shaped uint4 LUT read"
            )
        if source.count("st.shared.v4.b32") < 1:
            raise RuntimeError(f"{label} is missing the 16-byte LUT shared store")

    required_tokens = {
        "atomicAdd(": 1,
        "atom.cta.shared.add.u32": 2,
        "atom.sys.global.add.u64": 1,
        "redux.sync.add.u32": 1,
        "redux.sync.min.u32": 1,
        "fns.b32": 1,
        "ld.global.nc.b64": 2,
        (
            "cp.async.bulk.shared::cluster.global.mbarrier::"
            "complete_tx::bytes.L2::cache_hint"
        ): 1,
        (
            "cp.async.bulk.global.shared::cta.bulk_group."
            "L2::cache_hint"
        ): 1,
        "red.release.gpu.global.add.u32": 1,
        "red.release.sys.global.add.s32": 1,
        "ld.acquire.sys.global.s32": 1,
        "atom.release.gpu.global.add.u32": 2,
        "ld.acquire.gpu.global.b32": 2,
    }
    receipts = {}
    for (hidden, m), schedule in sorted(SMALL_RS_SCHEDULES.items()):
        label = f"{schedule.model}_m{m}"
        uses_interleaved_scheduler = _uses_current_interleaved_scheduler()
        l2_expected_arrival_mask = (
            1 << (schedule.intermediate * 2 // 256)
        ) - 1
        combine_num_chunks = (
            1
            if (
                3 * EPILOGUE_WARPS * schedule.hidden * 2
                <= SMEM_BYTES - LOOM_CONTROL_BYTES
                and schedule.hidden <= 32 * 128
            )
            else 2
        )
        if _wide_math_for(schedule):
            combine_num_chunks *= 2
        combine_chunk_bytes = schedule.hidden * 2 // combine_num_chunks
        audit_wide = _wide_math_for(schedule)
        audit_math_threads = 32 * _math_warps_for(schedule)
        expected_regs = (
            RS_TEAM_REGISTERS if (RS_TEAM_SLICE3 and schedule.swap_ab) else 48,
            64 if uses_interleaved_scheduler else 40,
            RS_WIDE_MATH_REGISTERS if audit_wide else 208,
        )
        kernel = _partial_dispatch_kernel(schedule)
        tma_buffers = tuple(
            buffer for buffer in kernel.buffers if buffer.tma is not None
        )
        tma_resources = tuple(str(buffer.resource) for buffer in tma_buffers)
        tma_l2_promotions = tuple(
            str(buffer.tma.l2_promotion) for buffer in tma_buffers
        )
        if tma_resources != PRODUCTION_TMA_RESOURCES:
            raise RuntimeError(
                f"{label} tensor-map resource order mismatch: expected "
                f"{PRODUCTION_TMA_RESOURCES}, got {tma_resources}"
            )
        if tma_l2_promotions != ("l2_256b",) * len(PRODUCTION_TMA_RESOURCES):
            raise RuntimeError(
                f"{label} tensor-map promotion mismatch: expected seven "
                f"l2_256b descriptors, got {tma_l2_promotions}"
            )
        descriptor_by_resource = {
            str(buffer.resource): buffer.tma for buffer in tma_buffers
        }
        external_binding_plan = _external_tma_binding_plan(schedule)
        weight_binding_plan = _weight_tma_binding_plan(schedule)
        planned_resources = {
            plan.resource for plan in (*external_binding_plan, *weight_binding_plan)
        }
        if planned_resources != set(PRODUCTION_TMA_RESOURCES):
            raise RuntimeError(
                f"{label} host tensor-map plan mismatch: got "
                f"{sorted(planned_resources)}"
            )
        layout = _dispatch_layout(schedule)
        dtype_bytes = {"u8": 1, "f32": 4}
        dtype_names = {"u8": "uint8", "f32": "float32"}
        for plan in external_binding_plan:
            descriptor = descriptor_by_resource[plan.resource]
            if str(descriptor.dtype) != dtype_names[plan.dtype]:
                raise RuntimeError(
                    f"{label} {plan.resource} dtype mismatch: plan "
                    f"{plan.dtype}, descriptor {descriptor.dtype}"
                )
            box = tuple(
                int(getattr(dimension, "value", dimension))
                for dimension in descriptor.box_shape
            )
            if box[0] > plan.cols or box[1] > plan.rows:
                raise RuntimeError(
                    f"{label} {plan.resource} box {box} exceeds host window "
                    f"{(plan.cols, plan.rows)}"
                )
            window_bytes = (
                (plan.rows - 1) * plan.row_pitch_bytes
                + plan.cols * dtype_bytes[plan.dtype]
            )
            if plan.byte_offset + window_bytes > layout.symmetric_bytes:
                raise RuntimeError(
                    f"{label} {plan.resource} external slice exceeds the "
                    "symmetric allocation"
                )
        for plan in weight_binding_plan:
            descriptor = descriptor_by_resource[plan.resource]
            box = tuple(
                int(getattr(dimension, "value", dimension))
                for dimension in descriptor.box_shape
            )
            if str(descriptor.dtype) != "uint8":
                raise RuntimeError(
                    f"{label} {plan.resource} must bind packed uint8 weights"
                )
            if box[0] > plan.cols or box[1] > plan.rows:
                raise RuntimeError(
                    f"{label} {plan.resource} box {box} exceeds weight view "
                    f"{(plan.cols, plan.rows)}"
                )
        buffer_names = tuple(str(buffer.name) for buffer in kernel.buffers)
        if "compile_sentinel" in buffer_names:
            raise RuntimeError(
                f"{label} production fragment still exposes compile_sentinel"
            )
        source = generate_kernel(kernel, validate=True)
        validate_constant_lut(label, kernel, source)
        if source.count("pg_flags") != 1:
            raise RuntimeError(
                f"{label} process-group flags must remain an unused ABI-only "
                "parameter"
            )
        if "compile_sentinel" in source:
            raise RuntimeError(
                f"{label} generated CUDA still exposes compile_sentinel"
            )
        emitted_regs = tuple(
            int(value)
            for value in re.findall(
                r"setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 (\d+)",
                source,
            )
        )
        if emitted_regs != expected_regs:
            raise RuntimeError(
                f"{label} register plan mismatch: expected {expected_regs}, "
                f"got {emitted_regs}"
            )
        barrier_instruction = (
            f"bar.sync {DISPATCH_BARRIER_ID}, {DISPATCH_THREADS}"
        )
        expected_dispatch_barriers = 11
        if source.count(barrier_instruction) != expected_dispatch_barriers:
            raise RuntimeError(
                f"{label} generated CUDA has {source.count(barrier_instruction)} "
                "dispatch scope barriers, expected "
                f"{expected_dispatch_barriers}"
            )
        missing = {
            token: expected
            for token, expected in required_tokens.items()
            if source.count(token) < expected
        }
        if missing:
            raise RuntimeError(
                f"{label} generated CUDA is missing typed dispatch ops {missing}"
            )
        expected_count_polls = (
            schedule.experts // NUM_RANKS + 31
        ) // 32
        # The scheduler keeps its per-lane expert groups in one pragma-unrolled
        # runtime loop, so it has one volatile instruction site independent of
        # the number of groups. Dispatch spells its two possible groups as
        # separate source sites above.
        scheduler_count_polls = 1 if uses_interleaved_scheduler else 0
        expected_u64_polls = expected_count_polls + scheduler_count_polls
        expected_u32_polls = 1 if uses_interleaved_scheduler else 0
        expected_runtime_loops = (
            6
            + expected_count_polls
            + scheduler_count_polls
            + (5 if uses_interleaved_scheduler else 0)
            + 8
            # decode team: one task-stream `while (true)`.
            + (1 if (RS_TEAM_SLICE3 and schedule.swap_ab) else 0)
        )
        if source.count("ld.volatile.global.b64") != expected_u64_polls:
            raise RuntimeError(
                f"{label} generated CUDA has "
                f"{source.count('ld.volatile.global.b64')} volatile count polls, "
                f"expected {expected_u64_polls}"
            )
        if source.count("ld.volatile.global.b32") != expected_u32_polls:
            raise RuntimeError(
                f"{label} generated CUDA has "
                f"{source.count('ld.volatile.global.b32')} volatile task polls, "
                f"expected {expected_u32_polls}"
            )
        if source.count("while (") != expected_runtime_loops:
            raise RuntimeError(
                f"{label} generated CUDA has {source.count('while (')} "
                f"runtime loops, expected {expected_runtime_loops}"
            )
        scheduler_tokens = {
            "atomicAdd(reinterpret_cast<unsigned int*>": (0, 2),
            "redux.sync.add.u32": (1, 2),
            "continue;": (0, 1),
            "__shfl_up_sync": (0, 5),
            "__ffs(": (0, 1),
        }
        for token, (static_expected, interleaved_expected) in (
            scheduler_tokens.items()
        ):
            actual = source.count(token)
            wanted = (
                interleaved_expected
                if uses_interleaved_scheduler
                else static_expected
            )
            if token == "redux.sync.add.u32":
                wanted += 1
            if token == "__ffs(":
                wanted += 2
            if actual != wanted:
                raise RuntimeError(
                    f"{label} scheduler token {token!r} count mismatch: "
                    f"expected {wanted}, got {actual}"
                )
        mailbox_tokens = {
            (
                f"// task_info_full: {TASK_INFO_STAGES} barriers, "
                "init_count=1"
            ): 1,
            (
                f"// task_info_empty: {TASK_INFO_STAGES} barriers, "
                f"init_count={_math_warps_for(schedule) + (2 if (RS_TEAM_SLICE3 and schedule.swap_ab) else 0)}"
            ): 1,
            f"if (producer_task_stage == {TASK_INFO_STAGES})": 1,
            f"if (loader_task_stage == {TASK_INFO_STAGES})": 1,
            f"if (math_task_stage == {TASK_INFO_STAGES})": 1,
            "if (task_phase == 0)": 1,
            "if (loader_task_phase == 0)": 1,
            "if (math_task_phase == 0)": 1,
            "unsigned int task_info_words[8]": 1,
            "unsigned int loader_task_words[8]": 1,
            "unsigned int math_task_words[8]": 1,
            "__threadfence_block();": 1,
            "mbarrier_arrive(task_info_full": 1,
            "mbarrier_arrive(task_info_empty": 1,
        }
        missing_mailbox = {
            token: expected
            for token, expected in mailbox_tokens.items()
            if source.count(token) < expected
        }
        if uses_interleaved_scheduler and missing_mailbox:
            raise RuntimeError(
                f"{label} scheduler mailbox transport is missing "
                f"{missing_mailbox}"
            )
        selected_shapes = tuple(
            n_swap
            for n_swap in (8, 16, 24)
            if schedule.swap_ab and n_swap <= schedule.block_m
        )
        task_wgmma_sites = {
            n_swap: source.count(
                "wgmma.mma_async.sync.aligned."
                f"m64n{n_swap}k32.f32.e4m3.e4m3"
            )
            for n_swap in (8, 16, 24)
        }
        # Half-level rotation: two halves x four K32 slices per selected shape.
        # Stage-level rotation: four block bodies (even, odd with/without a
        # successor, odd tail) x 8.
        def _task_sites(n_swap: int) -> int:
            if n_swap not in selected_shapes:
                return 0
            if audit_wide:
                return 16
            if RS_STAGE_ROTATE and n_swap <= RS_STAGE_ROTATE_MAX_N and not (RS_TEAM_SLICE3 and schedule.swap_ab):
                return 32
            return 8

        expected_task_wgmma_sites = {n_swap: _task_sites(n_swap) for n_swap in (8, 16, 24)}
        if task_wgmma_sites != expected_task_wgmma_sites:
            raise RuntimeError(
                f"{label} scheduler-bound task WGMMA sites mismatch: "
                f"expected {expected_task_wgmma_sites}, "
                f"got {task_wgmma_sites}"
            )
        nonswap_wgmma_sites = source.count(
            "wgmma.mma_async.sync.aligned."
            "m64n128k32.f32.e4m3.e4m3"
        )
        expected_nonswap_wgmma_sites = 0 if schedule.swap_ab else 8
        if nonswap_wgmma_sites != expected_nonswap_wgmma_sites:
            raise RuntimeError(
                f"{label} scheduler-bound non-swap WGMMA sites mismatch: "
                f"expected {expected_nonswap_wgmma_sites}, "
                f"got {nonswap_wgmma_sites}"
            )
        if schedule.swap_ab:
            l1_tokens = {
                (
                    f"// l1_stage_full: {schedule.stages} barriers, "
                    "init_count=2"
                ): 1,
                (
                    f"// l1_stage_empty: {schedule.stages} barriers, "
                    f"init_count={_math_warps_for(schedule)}"
                ): 1,
                f"if (l1_load_stage == {schedule.stages})": 2,
                f"if (l1_math_stage == {schedule.stages})": 1,
                "ld.acquire.gpu.global.u32": 1,
                "ld.acquire.gpu.global.u64": 1,
                "ld.global.nc.b32": 2,
                f"barrier.sync 1, {64 + audit_math_threads}": 4,
                "ld.shared.v2.b32": len(selected_shapes),
                "wgmma.commit_group.sync.aligned": len(selected_shapes),
                "wgmma.wait_group.sync.aligned 0": len(selected_shapes),
            }
            missing_l1 = {
                token: expected
                for token, expected in l1_tokens.items()
                if source.count(token) < expected
            }
            if missing_l1:
                raise RuntimeError(
                    f"{label} scheduler-bound L1 body is missing {missing_l1}"
                )
            task_tma_call_sites = source.count("tma_2d_gmem2smem(") - 1
            if task_tma_call_sites != 6:
                raise RuntimeError(
                    f"{label} scheduler-bound L1/L2 TMA sites mismatch: "
                    f"expected 6, got {task_tma_call_sites}"
                )
            l1_tma_call_sites = 3
            l2_tma_call_sites = 3
            l1_epilogue_tokens = {
                f"bar.sync 2, {audit_math_threads}": 9,
                "__expf(": 2,
                "cvt.rn.satfinite.e4m3x2.f32": 2,
                "red.release.gpu.global.or.b64": 1,
                "ld.shared.v4.b32": 1,
                "reinterpret_cast<int4*>": 1,
                "__float2bfloat16_rn(": 4,
            }
            missing_l1_epilogue = {
                token: expected
                for token, expected in l1_epilogue_tokens.items()
                if source.count(token) < expected
            }
            if missing_l1_epilogue:
                raise RuntimeError(
                    f"{label} scheduler-bound L1 epilogue is missing "
                    f"{missing_l1_epilogue}"
                )
            l1_tma_store_sites = source.count("tma_store_2d(") - 1
            if l1_tma_store_sites != 1:
                raise RuntimeError(
                    f"{label} scheduler-bound L1 output TMA sites mismatch: "
                    f"expected 1, got {l1_tma_store_sites}"
                )
            post_scatter_tokens = {
                "red.release.sys.global.add.s32": 3,
                "ld.acquire.sys.global.s32": 3,
                "atom.release.gpu.global.add.u32": 5,
                "ld.acquire.gpu.global.b32": 5,
                "redux.sync.add.u32": 3,
                f"barrier.sync 1, {64 + audit_math_threads}": 4,
                "bar.sync 0, 64": 11,
                # The wide scatter adds one staging-to-read sync per L2 task.
                f"bar.sync 2, {audit_math_threads}": 10 if audit_wide else 9,
            }
            mismatched_post_scatter = {
                token: (expected, source.count(token))
                for token, expected in post_scatter_tokens.items()
                if source.count(token) != expected
            }
            if mismatched_post_scatter:
                raise RuntimeError(
                    f"{label} post-scatter protocol mismatch: "
                    f"{mismatched_post_scatter}"
                )
            cleanup_tokens = {
                "cleanup_local_expert": 1,
                "cleanup_num_m_blocks": 1,
                "cleanup_pool_block_offset": 1,
                "cleanup_pool_block": 1,
            }
            missing_cleanup = {
                token: expected
                for token, expected in cleanup_tokens.items()
                if source.count(token) < expected
            }
            if missing_cleanup:
                raise RuntimeError(
                    f"{label} workspace cleanup is missing {missing_cleanup}"
                )
            combine_tokens = {
                (
                    f"// combine_full: {EPILOGUE_WARPS * 2} barriers, "
                    "init_count=1"
                ): 1,
                "ld.global.nc.b64": 3,
                "__ballot_sync": 3,
                "__float22bfloat162_rn": 1,
                "cp.async.bulk.global.shared::cta.bulk_group": 2,
            }
            mismatched_combine = {
                token: (expected, source.count(token))
                for token, expected in combine_tokens.items()
                if source.count(token) != expected
            }
            raw_combine_load_sites = (
                source.count("cp_async_bulk_gmem2smem(") - 1
            )
            if raw_combine_load_sites != 2:
                mismatched_combine["cp_async_bulk_gmem2smem call sites"] = (
                    2,
                    raw_combine_load_sites,
                )
            if mismatched_combine:
                raise RuntimeError(
                    f"{label} top-k combine mismatch: {mismatched_combine}"
                )
        else:
            nonswap_l1_tokens = {
                (
                    f"// l1_stage_full: {schedule.stages} barriers, "
                    "init_count=2"
                ): 1,
                (
                    f"// l1_stage_empty: {schedule.stages} barriers, "
                    "init_count=8"
                ): 1,
                f"if (l1_load_stage == {schedule.stages})": 2,
                f"if (l1_math_stage == {schedule.stages})": 1,
                "if (math_k_block_idx == 0)": 1,
                "ld.acquire.gpu.global.u32": 1,
                "ld.acquire.gpu.global.u64": 1,
                "ld.global.nc.b32": 2,
                f"barrier.sync 1, {64 + audit_math_threads}": 4,
                "wgmma.commit_group.sync.aligned": 2,
                "wgmma.wait_group.sync.aligned 0": 2,
                "bar.sync 3, 128": 2,
                "bar.sync 4, 128": 2,
                "bar.sync 2, 256": 8,
                "__expf(": 2,
                "cvt.rn.satfinite.e4m3x2.f32": 2,
                "__float22bfloat162_rn": 5,
                "red.release.gpu.global.or.b64": 1,
            }
            missing_nonswap_l1 = {
                token: expected
                for token, expected in nonswap_l1_tokens.items()
                if source.count(token) < expected
            }
            if missing_nonswap_l1:
                raise RuntimeError(
                    f"{label} scheduler-bound non-swap L1 body is missing "
                    f"{missing_nonswap_l1}"
                )
            # Both transported phases now consume the generated descriptor
            # branches: three L1 and three L2 TMA sites share the persistent
            # loader cursor exactly as in the source interleaved schedule.
            generated_task_tma_call_sites = (
                source.count("tma_2d_gmem2smem(") - 1
            )
            if generated_task_tma_call_sites != 6:
                raise RuntimeError(
                    f"{label} non-swap generated TMA sites mismatch: "
                    f"expected 6, got {generated_task_tma_call_sites}"
                )
            l1_tma_call_sites = 3
            l2_tma_call_sites = 3
            l1_tma_store_sites = source.count("tma_store_2d(") - 1
            if l1_tma_store_sites != 1:
                raise RuntimeError(
                    f"{label} scheduler-bound non-swap L1 output TMA sites "
                    f"mismatch: expected 1, got {l1_tma_store_sites}"
                )
            post_scatter_tokens = {
                "red.release.sys.global.add.s32": 3,
                "ld.acquire.sys.global.s32": 3,
                "atom.release.gpu.global.add.u32": 5,
                "ld.acquire.gpu.global.b32": 5,
                "redux.sync.add.u32": 3,
                f"barrier.sync 1, {64 + audit_math_threads}": 4,
                "bar.sync 0, 64": 11,
                "bar.sync 2, 256": 8,
            }
            mismatched_post_scatter = {
                token: (expected, source.count(token))
                for token, expected in post_scatter_tokens.items()
                if source.count(token) != expected
            }
            if mismatched_post_scatter:
                raise RuntimeError(
                    f"{label} non-swap post-scatter protocol mismatch: "
                    f"{mismatched_post_scatter}"
                )
            cleanup_tokens = {
                "cleanup_local_expert": 1,
                "cleanup_num_m_blocks": 1,
                "cleanup_pool_block_offset": 1,
                "cleanup_pool_block": 1,
            }
            missing_cleanup = {
                token: expected
                for token, expected in cleanup_tokens.items()
                if source.count(token) < expected
            }
            if missing_cleanup:
                raise RuntimeError(
                    f"{label} non-swap workspace cleanup is missing "
                    f"{missing_cleanup}"
                )
            combine_tokens = {
                (
                    f"// combine_full: {EPILOGUE_WARPS * 2} barriers, "
                    "init_count=1"
                ): 1,
                "ld.global.nc.b64": 3,
                "__ballot_sync": 3,
                "cp.async.bulk.global.shared::cta.bulk_group": 2,
            }
            mismatched_combine = {
                token: (expected, source.count(token))
                for token, expected in combine_tokens.items()
                if source.count(token) != expected
            }
            raw_combine_load_sites = (
                source.count("cp_async_bulk_gmem2smem(") - 1
            )
            if raw_combine_load_sites != 2:
                mismatched_combine["cp_async_bulk_gmem2smem call sites"] = (
                    2,
                    raw_combine_load_sites,
                )
            if mismatched_combine:
                raise RuntimeError(
                    f"{label} non-swap top-k combine mismatch: "
                    f"{mismatched_combine}"
                )
        forbidden = tuple(
            token
            for token in ("tcgen05", "tmem_", "qmul4")
            if token in source.lower()
        )
        if forbidden:
            raise RuntimeError(f"{label} generated CUDA contains {forbidden}")

        cubin = compile_cuda(
            source,
            options=["--use_fast_math"],
            include_dirs=_cuda_include_dirs(),
        )
        artifact = _remember_weave_artifact(
            schedule,
            arch=arch,
            kernel=kernel,
            generated_source=source,
            cubin=cubin,
        )
        strict_receipt = _weave_artifact_receipt(artifact)
        if tuple(strict_receipt) != (
            "source_family",
            "provider_kind",
            "real_weave_consumer",
            "provider_source",
            "provider_source_sha256",
            "generated_source_sha256",
            "cubin_sha256",
            "kernel_symbols",
            "forbidden_blackwell_absent",
        ):
            raise RuntimeError(f"{label} strict artifact receipt shape changed")
        receipts[label] = {
            "source_sha256": artifact.generated_source_sha256,
            "cubin_sha256": artifact.cubin_sha256,
            "kernel_symbol": artifact.kernel_symbol,
            "tma_descriptor_count": len(tma_buffers),
            "tma_l2_promotion": "l2_256b",
            "runtime_external_allocation_bytes": layout.symmetric_bytes,
            "runtime_external_tma_bindings": [
                plan._asdict() for plan in external_binding_plan
            ],
            "runtime_weight_tma_bindings": [
                plan._asdict() for plan in weight_binding_plan
            ],
            "runtime_peer_table_bindings": [
                "pg_flags",
                "sym_buffer_peers",
            ],
            "runtime_pg_flags_abi_only": True,
            "production_compile_sentinel_absent": True,
            "constant_lut_pointer_argument_absent": True,
            "constant_lut_payload_sha256": actual_lut_sha256,
            "constant_lut_copy_threads": 64,
            "constant_lut_copy_bytes_per_thread": 16,
            "register_plan": list(emitted_regs),
            "active_dispatch_warps": (
                1 if schedule.single_dispatch else 2
            ),
            "experts": schedule.experts,
            "experts_per_rank": schedule.experts // NUM_RANKS,
            "hidden": hidden,
            "block_m": schedule.block_m,
            "workspace_layout": _dispatch_layout(schedule)._asdict(),
            "scheduler_task_claim": uses_interleaved_scheduler,
            "scheduler_task_fields": 8 if uses_interleaved_scheduler else 0,
            "requested_interleaved_arm": (
                schedule.requested_interleaved_arm
            ),
            "requested_rs_arm": schedule.requested_rs_arm,
            "effective_rs_arm": schedule.swap_ab,
            "effective_interleaved_scheduler": uses_interleaved_scheduler,
            "scheduler_count_polls": scheduler_count_polls,
            "scheduler_task_ready_polls": expected_u32_polls,
            "scheduler_mailbox_transport": uses_interleaved_scheduler,
            "scheduler_invalid_terminator": uses_interleaved_scheduler,
            "scheduler_mailbox_stages": TASK_INFO_STAGES,
            "scheduler_mailbox_full_init_count": 1,
            "scheduler_mailbox_empty_init_count": 8,
            "scheduler_mailbox_consumer_warps": 9,
            "scheduler_mailbox_release_warps": 8,
            "scheduler_transport_downstream_fused": True,
            "scheduler_l1_task_body_fused": True,
            "scheduler_l1_task_body_rows": 1,
            "scheduler_l1_tma_call_sites": l1_tma_call_sites,
            "scheduler_l1_tma_store_sites": l1_tma_store_sites,
            "scheduler_l1_wgmma_sites": task_wgmma_sites,
            "scheduler_l1_nonswap_wgmma_sites": nonswap_wgmma_sites,
            "scheduler_l1_swiglu_fp8_publication": True,
            "scheduler_l1_task_release_boundary": (
                "first L1 full-stage wait"
            ),
            "scheduler_nonswap_l1_consumer": not schedule.swap_ab,
            "scheduler_nonswap_generated_tma_call_sites": (
                0 if schedule.swap_ab else generated_task_tma_call_sites
            ),
            "scheduler_l2_task_body_fused": True,
            "scheduler_l2_task_body_rows": 1,
            "scheduler_l2_tma_call_sites": l2_tma_call_sites,
            "scheduler_l2_wgmma_sites": (
                task_wgmma_sites
                if schedule.swap_ab
                else {128: 4}
            ),
            "scheduler_l2_expected_arrival_mask": l2_expected_arrival_mask,
            "scheduler_l2_bf16_remote_scatter": True,
            "scheduler_l2_scatter_vector_bytes": (
                16 if schedule.swap_ab else 4
            ),
            "scheduler_l2_scatter_metadata_words": 3,
            "scheduler_l2_task_release_boundary": (
                "first L2 full-stage wait"
            ),
            "scheduler_nonswap_l2_consumer": not schedule.swap_ab,
            "scheduler_nonswap_direct_scatter_store_width_bytes": (
                4 if not schedule.swap_ab else 0
            ),
            "scheduler_precombine_rendezvous": True,
            "scheduler_workspace_cleanup": True,
            "scheduler_after_cleanup_rendezvous": True,
            "scheduler_topk_combine": True,
            "scheduler_combine_num_chunks": combine_num_chunks,
            "scheduler_combine_chunk_bytes": combine_chunk_bytes,
            "scheduler_combine_load_stages_per_warp": 2,
            "scheduler_combine_store_slots_per_warp": 1,
            "scheduler_rendezvous_call_sites": 3,
            "scheduler_grid_sync_call_sites": 5,
            "scheduler_cleanup_cumulative_stats": "wrapper nullptr",
            "computed_smem_bytes": kernel.computed_smem_bytes,
            "forbidden_blackwell_absent": True,
            "launch_grid": [NUM_CTAS, 1, 1],
            "launch_block": [THREADS, 1, 1],
            "launch_cluster_dims": list(CLUSTER_DIMS),
            "launch_shared_mem_bytes": SMEM_BYTES,
            "launch_use_pdl": USE_PDL,
            "strict_artifact_receipt_fields": list(strict_receipt),
        }

    unique_cubins = len(
        {receipt["cubin_sha256"] for receipt in receipts.values()}
    )
    if unique_cubins < 6:
        raise RuntimeError(
            f"ten dispatch rows collapsed below six material variants: "
            f"{unique_cubins}"
        )

    # The source runtime rounds valid_m to one of these three static
    # register-source WGMMA shapes. The current wrapper's effective schedule
    # gives all three shapes the interleaved loader budget.
    rs_specs = {
        "m64n8k32_interleaved": (8, 64),
        "m64n16k32_interleaved": (16, 64),
        "m64n24k32_interleaved": (24, 64),
    }
    rs_receipts = {}
    for label, (n_swap, loader_registers) in rs_specs.items():
        kernel = _partial_l1_rs_kernel(
            n_swap=n_swap,
            loader_registers=loader_registers,
        )
        source = generate_kernel(kernel, validate=True)
        validate_constant_lut(label, kernel, source)
        emitted_regs = tuple(
            int(value)
            for value in re.findall(
                r"setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 (\d+)",
                source,
            )
        )
        expected_regs = (48, loader_registers, 208)
        if emitted_regs != expected_regs:
            raise RuntimeError(
                f"{label} register plan mismatch: expected {expected_regs}, "
                f"got {emitted_regs}"
            )
        wgmma_token = (
            "wgmma.mma_async.sync.aligned."
            f"m64n{n_swap}k32.f32.e4m3.e4m3"
        )
        if source.count(wgmma_token) != 4:
            raise RuntimeError(
                f"{label} has {source.count(wgmma_token)} K32 WGMMA sites, "
                "expected four"
            )
        required_rs_tokens = {
            f"barrier.sync {DISPATCH_WITH_MATH_BARRIER_ID}, "
            f"{DISPATCH_WITH_MATH_THREADS}": 2,
            "ld.acquire.gpu.global.u32": 1,
            "ld.shared.v2.b32": 1,
            "prmt.b32": 4,
            "wgmma.fence.sync.aligned": 1,
            "wgmma.commit_group.sync.aligned": 1,
            "wgmma.wait_group.sync.aligned 0": 1,
            'asm volatile("" : "+r"': 1,
            'asm volatile("" : "+f"': 1,
            "bar.sync 3, 128": 1,
            "bar.sync 4, 128": 1,
        }
        missing = {
            token: expected
            for token, expected in required_rs_tokens.items()
            if source.count(token) < expected
        }
        if missing:
            raise RuntimeError(
                f"{label} generated CUDA is missing typed RS ops {missing}"
            )
        forbidden = tuple(
            token
            for token in ("tcgen05", "tmem_", "qmul4")
            if token in source.lower()
        )
        if forbidden:
            raise RuntimeError(f"{label} generated CUDA contains {forbidden}")
        if kernel.computed_smem_bytes != SMEM_BYTES:
            raise RuntimeError(
                f"{label} SMEM mismatch: expected {SMEM_BYTES}, "
                f"got {kernel.computed_smem_bytes}"
            )

        cubin = compile_cuda(
            source,
            options=["--use_fast_math"],
            include_dirs=_cuda_include_dirs(),
        )
        rs_receipts[label] = {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "kernel_symbol": f"kernel_{kernel.symbol}",
            "n_swap": n_swap,
            "dispatch_math_handoff_threads": DISPATCH_WITH_MATH_THREADS,
            "l1_arrival_acquire": "u32 equality, device scope",
            "constant_lut_pointer_argument_absent": True,
            "constant_lut_payload_sha256": actual_lut_sha256,
            "k_slices": 4,
            "weight_halves": 2,
            "register_plan": list(emitted_regs),
            "computed_smem_bytes": kernel.computed_smem_bytes,
            "forbidden_blackwell_absent": True,
        }

    # The descriptor-bound L1 producer schedule follows the wrapper's
    # physical BLOCK_M, stage count, and effective interleaving choice. Keep
    # every material combination even when multiple workload rows select it.
    tma_specs = {
        (
            f"bm{schedule.block_m}_s{schedule.stages}_"
            f"{'interleaved' if _uses_current_interleaved_scheduler() else 'non_interleaved'}"
        ): (
            schedule.block_m,
            schedule.stages,
            64 if _uses_current_interleaved_scheduler() else 40,
        )
        for schedule in SMALL_RS_SCHEDULES.values()
    }
    tma_receipts = {}
    for label, (block_m, stages, loader_registers) in sorted(tma_specs.items()):
        kernel = _partial_l1_tma_kernel(
            block_m=block_m,
            stages=stages,
            loader_registers=loader_registers,
        )
        source = generate_kernel(kernel, validate=True)
        emitted_regs = tuple(
            int(value)
            for value in re.findall(
                r"setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 (\d+)",
                source,
            )
        )
        expected_regs = (48, loader_registers, 208)
        if emitted_regs != expected_regs:
            raise RuntimeError(
                f"{label} register plan mismatch: expected {expected_regs}, "
                f"got {emitted_regs}"
            )
        required_tma_tokens = {
            f"barrier.sync {DISPATCH_WITH_MATH_BARRIER_ID}, "
            f"{DISPATCH_WITH_MATH_THREADS}": 2,
            "ld.acquire.gpu.global.u32": 1,
            f"// stage_full: {stages} barriers, init_count=2": 1,
            f"// stage_empty: {stages} barriers, init_count=8": 1,
            f"if (load_stage == {stages})": 1,
            f"if (math_stage == {stages})": 1,
        }
        missing = {
            token: expected
            for token, expected in required_tma_tokens.items()
            if source.count(token) < expected
        }
        if missing:
            raise RuntimeError(
                f"{label} generated CUDA is missing typed L1 TMA ops {missing}"
            )
        # One helper definition plus exactly three source call sites: A, SFA,
        # and packed B. Likewise expect-tx has one helper plus one arrival from
        # each loader warp.
        tma_call_sites = source.count("tma_2d_gmem2smem(") - 1
        expect_tx_call_sites = source.count("mbarrier_arrive_expect_tx(") - 1
        if tma_call_sites != 3 or expect_tx_call_sites != 2:
            raise RuntimeError(
                f"{label} TMA sites mismatch: tma={tma_call_sites}, "
                f"expect_tx={expect_tx_call_sites}"
            )
        forbidden = tuple(
            token
            for token in ("tcgen05", "tmem_", "qmul4")
            if token in source.lower()
        )
        if forbidden:
            raise RuntimeError(f"{label} generated CUDA contains {forbidden}")
        if kernel.computed_smem_bytes != SMEM_BYTES:
            raise RuntimeError(
                f"{label} SMEM mismatch: expected {SMEM_BYTES}, "
                f"got {kernel.computed_smem_bytes}"
            )

        cubin = compile_cuda(
            source,
            options=["--use_fast_math"],
            include_dirs=_cuda_include_dirs(),
        )
        tma_receipts[label] = {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "kernel_symbol": f"kernel_{kernel.symbol}",
            "block_m": block_m,
            "stages": stages,
            "activation_tx_bytes": block_m * RS_ACT_ROW_BYTES,
            "activation_scale_tx_bytes": block_m * 4,
            "packed_weight_tx_bytes": RS_TMA_PACKED_B_BYTES,
            "full_barrier_init_count": 2,
            "empty_barrier_init_count": 8,
            "tma_call_sites": tma_call_sites,
            "expect_tx_call_sites": expect_tx_call_sites,
            "register_plan": list(emitted_regs),
            "computed_smem_bytes": kernel.computed_smem_bytes,
            "forbidden_blackwell_absent": True,
        }

    # Merge the two previously separate L1 boundaries for each material
    # swap_ab schedule arm. Runtime valid_m branches retain the source's
    # static N8/N16/N24 WGMMA instructions inside one staged TMA consumer.
    integrated_specs = {
        (
            f"bm{schedule.block_m}_s{schedule.stages}_"
            f"{'interleaved' if _uses_current_interleaved_scheduler() else 'non_interleaved'}"
        ): (
            schedule.block_m,
            schedule.stages,
            64 if _uses_current_interleaved_scheduler() else 40,
            _uses_current_interleaved_scheduler(),
        )
        for schedule in SMALL_RS_SCHEDULES.values()
        if schedule.swap_ab
    }
    integrated_receipts = {}
    for label, (
        block_m,
        stages,
        loader_registers,
        interleaved_scheduler,
    ) in sorted(
        integrated_specs.items()
    ):
        kernel = _partial_l1_tma_rs_kernel(
            block_m=block_m,
            stages=stages,
            loader_registers=loader_registers,
            interleaved_scheduler=interleaved_scheduler,
        )
        source = generate_kernel(kernel, validate=True)
        validate_constant_lut(label, kernel, source)
        emitted_regs = tuple(
            int(value)
            for value in re.findall(
                r"setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 (\d+)",
                source,
            )
        )
        expected_regs = (48, loader_registers, 208)
        if emitted_regs != expected_regs:
            raise RuntimeError(
                f"{label} integrated register plan mismatch: "
                f"expected {expected_regs}, got {emitted_regs}"
            )

        selected_shapes = tuple(
            n_swap for n_swap in (8, 16, 24) if n_swap <= block_m
        )
        wgmma_sites = {
            n_swap: source.count(
                "wgmma.mma_async.sync.aligned."
                f"m64n{n_swap}k32.f32.e4m3.e4m3"
            )
            for n_swap in (8, 16, 24)
        }
        # Rotated integrated body: two halves x four K32 slices per shape.
        expected_wgmma_sites = {
            n_swap: 8 if n_swap in selected_shapes else 0
            for n_swap in (8, 16, 24)
        }
        if wgmma_sites != expected_wgmma_sites:
            raise RuntimeError(
                f"{label} integrated WGMMA sites mismatch: "
                f"expected {expected_wgmma_sites}, got {wgmma_sites}"
            )

        required_integrated_tokens = {
            f"barrier.sync {DISPATCH_WITH_MATH_BARRIER_ID}, "
            f"{DISPATCH_WITH_MATH_THREADS}": 2,
            "ld.acquire.gpu.global.u32": 1,
            f"// stage_full: {stages} barriers, init_count=2": 1,
            f"// stage_empty: {stages} barriers, init_count=8": 1,
            f"if (load_stage == {stages})": 1,
            f"if (math_stage == {stages})": 1,
            "ld.shared.v2.b32": len(selected_shapes),
            "wgmma.commit_group.sync.aligned": len(selected_shapes),
            "wgmma.wait_group.sync.aligned 0": len(selected_shapes),
            "bar.sync 3, 128": 1,
            "bar.sync 4, 128": 1,
        }
        if interleaved_scheduler:
            required_integrated_tokens.update({
                (
                    f"// task_info_full: {TASK_INFO_STAGES} barriers, "
                    "init_count=1"
                ): 1,
                (
                    f"// task_info_empty: {TASK_INFO_STAGES} barriers, "
                    "init_count=8"
                ): 1,
                f"if (producer_task_stage == {TASK_INFO_STAGES})": 1,
                f"if (loader_task_stage == {TASK_INFO_STAGES})": 1,
                f"if (math_task_stage == {TASK_INFO_STAGES})": 1,
                "__threadfence_block();": 1,
                "unsigned int task_info_words[8]": 1,
                "unsigned int loader_task_words[8]": 1,
                "unsigned int math_task_words[8]": 1,
            })
        elif "task_info_full" in source or "task_info_empty" in source:
            raise RuntimeError(
                f"{label} non-interleaved helper arm unexpectedly contains "
                "the interleaved TaskInfo mailbox"
            )
        missing = {
            token: expected
            for token, expected in required_integrated_tokens.items()
            if source.count(token) < expected
        }
        if missing:
            raise RuntimeError(
                f"{label} integrated CUDA is missing typed ops {missing}"
            )
        tma_call_sites = source.count("tma_2d_gmem2smem(") - 1
        expect_tx_call_sites = source.count("mbarrier_arrive_expect_tx(") - 1
        if tma_call_sites != 3 or expect_tx_call_sites != 2:
            raise RuntimeError(
                f"{label} integrated TMA sites mismatch: "
                f"tma={tma_call_sites}, expect_tx={expect_tx_call_sites}"
            )
        forbidden = tuple(
            token
            for token in ("tcgen05", "tmem_", "qmul4")
            if token in source.lower()
        )
        if forbidden:
            raise RuntimeError(
                f"{label} integrated CUDA contains {forbidden}"
            )
        if kernel.computed_smem_bytes != SMEM_BYTES:
            raise RuntimeError(
                f"{label} integrated SMEM mismatch: expected {SMEM_BYTES}, "
                f"got {kernel.computed_smem_bytes}"
            )

        cubin = compile_cuda(
            source,
            options=["--use_fast_math"],
            include_dirs=_cuda_include_dirs(),
        )
        integrated_receipts[label] = {
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
            "kernel_symbol": f"kernel_{kernel.symbol}",
            "block_m": block_m,
            "stages": stages,
            "runtime_n_swap_shapes": list(selected_shapes),
            "constant_lut_pointer_argument_absent": True,
            "constant_lut_payload_sha256": actual_lut_sha256,
            "wgmma_sites": wgmma_sites,
            "tma_call_sites": tma_call_sites,
            "expect_tx_call_sites": expect_tx_call_sites,
            "full_barrier_init_count": 2,
            "empty_barrier_init_count": 8,
            "task_info_mailbox": interleaved_scheduler,
            "register_plan": list(emitted_regs),
            "computed_smem_bytes": kernel.computed_smem_bytes,
            "forbidden_blackwell_absent": True,
        }
        if interleaved_scheduler:
            integrated_receipts[label].update({
                "task_info_stages": TASK_INFO_STAGES,
                "task_info_payload_fields": TASK_INFO_FIELDS,
                "task_info_payload_bytes": TASK_INFO_BYTES,
                "task_info_full_init_count": 1,
                "task_info_empty_init_count": 8,
                "task_info_publication_fence": "__threadfence_block",
                "task_info_release_boundary": "first L1 full-stage wait",
            })

    return {
        "schema": "loom-sm90-megamoe-small-rs-partial-v20",
        "arch": arch,
        "threads": THREADS,
        "grid": NUM_CTAS,
        "cluster_dims": list(CLUSTER_DIMS),
        "use_pdl": USE_PDL,
        "top_k": TOP_K,
        "schedule_rows": len(SMALL_RS_SCHEDULES),
        "production_tma_l2_256b_rows": sum(
            receipt["tma_descriptor_count"] == 7
            and receipt["tma_l2_promotion"] == "l2_256b"
            for receipt in receipts.values()
        ),
        "production_compile_sentinel_absent_rows": sum(
            receipt["production_compile_sentinel_absent"]
            for receipt in receipts.values()
        ),
        "production_constant_lut_rows": sum(
            receipt["constant_lut_pointer_argument_absent"]
            and receipt["constant_lut_payload_sha256"]
            == RS_LUT_PAYLOAD_SHA256
            for receipt in receipts.values()
        ),
        "production_runtime_binding_plan_rows": sum(
            len(receipt["runtime_external_tma_bindings"]) == 5
            and len(receipt["runtime_weight_tma_bindings"]) == 2
            and receipt["runtime_pg_flags_abi_only"]
            for receipt in receipts.values()
        ),
        "constant_lut_payload_sha256": actual_lut_sha256,
        "scheduler_task_claim_rows": sum(
            _uses_current_interleaved_scheduler()
            for schedule in SMALL_RS_SCHEDULES.values()
        ),
        "requested_rs_rows": sum(
            schedule.requested_rs_arm
            for schedule in SMALL_RS_SCHEDULES.values()
        ),
        "requested_interleaved_rows": sum(
            schedule.requested_interleaved_arm
            for schedule in SMALL_RS_SCHEDULES.values()
        ),
        "scheduler_task_payload_fields": 8,
        "scheduler_mailbox_transport_rows": sum(
            receipt["scheduler_mailbox_transport"]
            for receipt in receipts.values()
        ),
        "scheduler_invalid_terminator_rows": sum(
            receipt["scheduler_invalid_terminator"]
            for receipt in receipts.values()
        ),
        "scheduler_l1_task_body_rows": sum(
            receipt["scheduler_l1_task_body_rows"]
            for receipt in receipts.values()
        ),
        "scheduler_l1_swiglu_fp8_publication_rows": sum(
            receipt["scheduler_l1_swiglu_fp8_publication"]
            for receipt in receipts.values()
        ),
        "scheduler_nonswap_l1_consumer_rows": sum(
            receipt["scheduler_nonswap_l1_consumer"]
            for receipt in receipts.values()
        ),
        "scheduler_l2_task_body_rows": sum(
            receipt["scheduler_l2_task_body_rows"]
            for receipt in receipts.values()
        ),
        "scheduler_l2_bf16_remote_scatter_rows": sum(
            receipt["scheduler_l2_bf16_remote_scatter"]
            for receipt in receipts.values()
        ),
        "scheduler_nonswap_l2_consumer_rows": sum(
            receipt["scheduler_nonswap_l2_consumer"]
            for receipt in receipts.values()
        ),
        "scheduler_precombine_rendezvous_rows": sum(
            receipt["scheduler_precombine_rendezvous"]
            for receipt in receipts.values()
        ),
        "scheduler_workspace_cleanup_rows": sum(
            receipt["scheduler_workspace_cleanup"]
            for receipt in receipts.values()
        ),
        "scheduler_after_cleanup_rendezvous_rows": sum(
            receipt["scheduler_after_cleanup_rendezvous"]
            for receipt in receipts.values()
        ),
        "scheduler_topk_combine_rows": sum(
            receipt["scheduler_topk_combine"]
            for receipt in receipts.values()
        ),
        "integrated_task_mailbox_arms": sum(
            receipt["task_info_mailbox"]
            for receipt in integrated_receipts.values()
        ),
        "integrated_task_mailbox_rows": sum(
            schedule.swap_ab
            and _uses_current_interleaved_scheduler()
            for schedule in SMALL_RS_SCHEDULES.values()
        ),
        "unique_dispatch_cubins": unique_cubins,
        "production_artifact_cache_rows": len(
            [
                key
                for key in _WEAVE_ARTIFACT_CACHE
                if key[0] == arch
            ]
        ),
        "production_launch_plan_rows": sum(
            receipt["launch_grid"] == [NUM_CTAS, 1, 1]
            and receipt["launch_block"] == [THREADS, 1, 1]
            and receipt["launch_cluster_dims"] == list(CLUSTER_DIMS)
            and receipt["launch_shared_mem_bytes"] == SMEM_BYTES
            and receipt["launch_use_pdl"] is USE_PDL
            for receipt in receipts.values()
        ),
        "strict_receipt_rows": sum(
            len(receipt["strict_artifact_receipt_fields"]) == 9
            for receipt in receipts.values()
        ),
        "active_provider_kind": PROVIDER_KIND,
        "real_weave_consumer": REAL_WEAVE_CONSUMER,
        "fragments": receipts,
        "l1_tma_rs_fragments": integrated_receipts,
        "l1_tma_fragments": tma_receipts,
        "l1_rs_fragments": rs_receipts,
    }


def _read_cpu_scalar(tensor, *, dtype, name: str) -> int:
    """Read one frozen-ABI CPU scalar without accepting implicit coercions."""

    import torch

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != dtype
        or tensor.numel() != 1
    ):
        raise ValueError(f"{name} must be a CPU {dtype} one-element tensor")
    return int(tensor.item())


def _validated_weave_run_e2e_identity(
    sym_ptrs,
    rank_scalar,
    state_generation,
    l1_mode2_flat,
    l2_mode2_flat,
    l1_global_scales,
    l2_global_scales,
    route_profile,
    m_scalar,
    h_scalar,
    y,
) -> tuple[SmallRSSchedule, int]:
    """Validate the exact frozen ABI before creating any runtime resource."""

    import torch

    if (
        not isinstance(sym_ptrs, torch.Tensor)
        or sym_ptrs.device.type != "cpu"
        or sym_ptrs.dtype != torch.int64
        or not sym_ptrs.is_contiguous()
        or tuple(sym_ptrs.shape) != (NUM_RANKS,)
    ):
        raise ValueError("sym_ptrs must be contiguous CPU int64[8]")
    if any(int(pointer) == 0 for pointer in sym_ptrs.tolist()):
        raise ValueError("sym_ptrs contains a null symmetric pointer")

    rank = _read_cpu_scalar(
        rank_scalar,
        dtype=torch.int32,
        name="rank_scalar",
    )
    if not 0 <= rank < NUM_RANKS:
        raise ValueError(f"rank_scalar={rank} is outside [0, {NUM_RANKS})")
    env_rank = os.environ.get("MEGAMOE_RANK")
    if env_rank is None or int(env_rank) != rank:
        raise ValueError("rank_scalar differs from MEGAMOE_RANK")
    if not torch.cuda.is_available() or torch.cuda.current_device() != rank:
        raise ValueError(
            f"current CUDA device must equal rank {rank} for the physical8 launch"
        )

    generation = _read_cpu_scalar(
        state_generation,
        dtype=torch.int64,
        name="state_generation",
    )
    if generation <= 0:
        raise ValueError("state_generation must be positive")
    route = _read_cpu_scalar(
        route_profile,
        dtype=torch.int32,
        name="route_profile",
    )
    if route != 0:
        raise ValueError("physical8 fused campaign requires route profile P0")
    m = _read_cpu_scalar(m_scalar, dtype=torch.int32, name="m_scalar")
    hidden = _read_cpu_scalar(h_scalar, dtype=torch.int32, name="h_scalar")
    try:
        schedule = SMALL_RS_SCHEDULES[(hidden, m)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported small-RS model/M schedule hidden={hidden}, M={m}"
        ) from exc

    weight_plan = {
        plan.resource: plan for plan in _weight_tma_binding_plan(schedule)
    }
    expected_tensors = (
        (
            "l1_mode2_flat",
            l1_mode2_flat,
            weight_plan["l1_packed_weights"].rows
            * weight_plan["l1_packed_weights"].cols,
        ),
        (
            "l2_mode2_flat",
            l2_mode2_flat,
            weight_plan["l2_packed_weights"].rows
            * weight_plan["l2_packed_weights"].cols,
        ),
    )
    for name, tensor, expected_numel in expected_tensors:
        if (
            not isinstance(tensor, torch.Tensor)
            or not tensor.is_cuda
            or tensor.dtype != torch.uint8
            or not tensor.is_contiguous()
            or tensor.numel() != expected_numel
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA uint8 with "
                f"{expected_numel} elements"
            )

    expected_scales = schedule.experts // NUM_RANKS
    for name, scales in (
        ("l1_global_scales", l1_global_scales),
        ("l2_global_scales", l2_global_scales),
    ):
        if (
            not isinstance(scales, torch.Tensor)
            or not scales.is_cuda
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or scales.numel() != expected_scales
        ):
            raise ValueError(
                f"{name} must be contiguous CUDA float32 with "
                f"{expected_scales} elements"
            )
    if (
        not isinstance(y, torch.Tensor)
        or not y.is_cuda
        or y.dtype != torch.bfloat16
        or not y.is_contiguous()
        or tuple(y.shape) != (m, hidden)
    ):
        raise ValueError(f"y must be contiguous CUDA bfloat16[{m}, {hidden}]")
    return schedule, rank


def _prepare_weave_bindings(
    sym_ptrs,
    rank_scalar,
    state_generation,
    l1_mode2_flat,
    l2_mode2_flat,
    l1_global_scales,
    l2_global_scales,
    route_profile,
    m_scalar,
    h_scalar,
    y,
) -> PreparedSmallRSWeaveBindings:
    """Bind the complete production IR without compiling or launching it."""

    from loom.runtime.external_memory import ExternalDeviceAllocation
    from loom.runtime.launch import (
        create_tma_from_external_slice,
        create_tma_from_spec,
        pack_kernel_args,
    )

    schedule, rank = _validated_weave_run_e2e_identity(
        sym_ptrs,
        rank_scalar,
        state_generation,
        l1_mode2_flat,
        l2_mode2_flat,
        l1_global_scales,
        l2_global_scales,
        route_profile,
        m_scalar,
        h_scalar,
        y,
    )
    layout = _dispatch_layout(schedule)
    kernel = _partial_dispatch_kernel(schedule)
    descriptors = {
        str(buffer.resource): buffer.tma
        for buffer in kernel.buffers
        if buffer.tma is not None
    }
    if tuple(descriptors) != PRODUCTION_TMA_RESOURCES:
        raise RuntimeError(
            "production tensor-map resources changed: "
            f"expected {PRODUCTION_TMA_RESOURCES}, got {tuple(descriptors)}"
        )

    allocation = ExternalDeviceAllocation.from_pointer(
        int(sym_ptrs[rank].item()),
        layout.symmetric_bytes,
        label="sym_buffer",
    )
    peer_ptrs = sym_ptrs.to(device="cuda")
    tensor_maps: dict[str, object] = {}
    for plan in _external_tma_binding_plan(schedule):
        window = allocation.slice(
            plan.byte_offset,
            dtype=plan.dtype,
            shape=(plan.rows, plan.cols),
            row_pitch_bytes=plan.row_pitch_bytes,
            label=plan.resource,
        )
        tensor_maps[plan.resource] = create_tma_from_external_slice(
            descriptors[plan.resource],
            window,
        )

    weight_sources = {
        "l1_packed_weights": l1_mode2_flat,
        "l2_packed_weights": l2_mode2_flat,
    }
    for plan in _weight_tma_binding_plan(schedule):
        tensor_maps[plan.resource] = create_tma_from_spec(
            descriptors[plan.resource],
            weight_sources[plan.resource].view(plan.rows, plan.cols),
        )

    bindings = {
        "num_tokens": schedule.m,
        **tensor_maps,
        "l1_global_scales": l1_global_scales,
        "l2_global_scales": l2_global_scales,
        "output_bf16": y,
        "pg_world": NUM_RANKS,
        "pg_rank": rank,
        # NvlinkRendezvous consumes sym_buffer_peers directly.  The generic
        # process-group flags parameter remains in the physical ABI but is
        # proven unused by compile_partial_weave, so the same valid pointer
        # table is its inert binding without allocating hidden state.
        "pg_flags": peer_ptrs,
        "sym_buffer": allocation,
        "sym_buffer_peers": peer_ptrs,
    }
    packed_args = pack_kernel_args(kernel, **bindings)
    import os as _os
    if _os.environ.get("WEAVE_DEBUG_ARG_ALIGN") and int(_os.environ.get("RANK", "0")) == 0:
        import ctypes as _ct
        _tmap_names = [str(a.name) for a in kernel.args]
        for _i, (_nm, _v) in enumerate(zip(_tmap_names, packed_args)):
            _kind = type(_v).__name__
            _ptr = None
            if hasattr(_v, "is_cuda") and getattr(_v, "is_cuda", False):
                _ptr = int(_v.data_ptr())
            elif isinstance(_v, _ct.c_void_p):
                _ptr = int(_v.value or 0)
            _a64 = (_ptr % 64) if _ptr is not None else "-"
            print(f"[argalign] {_i:2d} {_nm:22s} {_kind:16s} ptr={_ptr} %64={_a64}", flush=True)
    return PreparedSmallRSWeaveBindings(
        schedule=schedule,
        kernel=kernel,
        allocation=allocation,
        peer_ptrs=peer_ptrs,
        tensor_maps=tensor_maps,
        packed_args=packed_args,
        bindings=bindings,
        keepalive=(
            sym_ptrs,
            l1_mode2_flat,
            l2_mode2_flat,
            l1_global_scales,
            l2_global_scales,
            y,
            peer_ptrs,
            *tensor_maps.values(),
        ),
    )


def _remember_weave_artifact(
    schedule: SmallRSSchedule,
    *,
    arch: str,
    kernel: object,
    generated_source: str,
    cubin: bytes,
) -> CompiledSmallRSWeaveArtifact:
    """Install one immutable generated artifact in the host cache."""

    expected_schedule = SMALL_RS_SCHEDULES.get((schedule.hidden, schedule.m))
    if expected_schedule != schedule:
        raise ValueError("artifact schedule is not one of the ten source rows")
    if arch != "sm_90a":
        raise RuntimeError(f"small-RS artifact requires sm_90a, got {arch}")
    if kernel != _partial_dispatch_kernel(schedule):
        raise RuntimeError("artifact kernel differs from the material source schedule")
    generated_source_sha256 = hashlib.sha256(
        generated_source.encode("utf-8")
    ).hexdigest()
    cubin_sha256 = hashlib.sha256(cubin).hexdigest()
    forbidden_tokens = tuple(
        token
        for token in ("tcgen05", "tmem_", "qmul4")
        if token in generated_source.lower()
    )
    if forbidden_tokens:
        raise RuntimeError(
            f"small-RS generated CUDA contains forbidden tokens {forbidden_tokens}"
        )
    artifact = CompiledSmallRSWeaveArtifact(
        schedule=schedule,
        arch=arch,
        kernel=kernel,
        generated_source=generated_source,
        generated_source_sha256=generated_source_sha256,
        cubin=cubin,
        cubin_sha256=cubin_sha256,
        kernel_symbol=f"kernel_{kernel.symbol}",
        forbidden_blackwell_absent=True,
    )
    key = (arch, schedule.hidden, schedule.m)
    existing = _WEAVE_ARTIFACT_CACHE.get(key)
    if existing is not None and (
        existing.generated_source_sha256 != generated_source_sha256
        or existing.cubin_sha256 != cubin_sha256
        or existing.kernel_symbol != artifact.kernel_symbol
    ):
        raise RuntimeError(f"small-RS artifact cache identity changed for {key}")
    _WEAVE_ARTIFACT_CACHE[key] = artifact
    return artifact


def _compile_weave_artifact(
    schedule: SmallRSSchedule,
) -> CompiledSmallRSWeaveArtifact:
    """Generate and compile one selected production row, caching exact bytes."""

    from loom.codegen.kernel import generate_kernel
    from loom.examples.gemm_common import _cuda_include_dirs
    from loom.runtime.compiler import compile_cuda, detect_gpu_arch

    arch = detect_gpu_arch()
    key = (arch, schedule.hidden, schedule.m)
    cached = _WEAVE_ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = _partial_dispatch_kernel(schedule)
    generated_source = generate_kernel(kernel, validate=True)
    cubin = compile_cuda(
        generated_source,
        arch=arch,
        options=["--use_fast_math"],
        include_dirs=_cuda_include_dirs(),
    )
    return _remember_weave_artifact(
        schedule,
        arch=arch,
        kernel=kernel,
        generated_source=generated_source,
        cubin=cubin,
    )


def _weave_artifact_receipt(
    artifact: CompiledSmallRSWeaveArtifact,
) -> dict[str, object]:
    """Build the strict worktree-bound receipt for one compiled artifact."""

    provider_source = Path(__file__).resolve()
    return {
        "source_family": SOURCE_FAMILY,
        "provider_kind": PROVIDER_KIND,
        "real_weave_consumer": REAL_WEAVE_CONSUMER,
        "provider_source": str(provider_source),
        "provider_source_sha256": _sha256(provider_source),
        "generated_source_sha256": artifact.generated_source_sha256,
        "cubin_sha256": artifact.cubin_sha256,
        "kernel_symbols": [artifact.kernel_symbol],
        "forbidden_blackwell_absent": artifact.forbidden_blackwell_absent,
    }


def _weave_cuda_kernel(
    artifact: CompiledSmallRSWeaveArtifact,
) -> object:
    """Load one cached cubin symbol on the current rank-local CUDA device."""

    import torch

    from loom.runtime.kernel import CUDAKernel

    device = torch.cuda.current_device()
    key = (device, artifact.cubin_sha256, artifact.kernel_symbol)
    cached = _WEAVE_CUDA_KERNEL_CACHE.get(key)
    if cached is None:
        cached = CUDAKernel(artifact.cubin, artifact.kernel_symbol)
        _WEAVE_CUDA_KERNEL_CACHE[key] = cached
    return cached


def _prepare_weave_launch(*args) -> PreparedSmallRSWeaveLaunch:
    """Assemble the exact source launch without submitting GPU work."""

    import torch

    bindings = _prepare_weave_bindings(*args)
    artifact = _compile_weave_artifact(bindings.schedule)
    if artifact.kernel != bindings.kernel:
        raise RuntimeError("compiled artifact and launch bindings use different IR")
    cuda_kernel = _weave_cuda_kernel(artifact)
    stream = torch.cuda.current_stream()
    prepared_launch = cuda_kernel.prepare_launch(
        grid=(NUM_CTAS, 1, 1),
        block=(_kernel_threads_for(bindings.schedule), 1, 1),
        args=bindings.packed_args,
        shared_mem=SMEM_BYTES,
        stream=stream,
        use_pdl=USE_PDL,
    )
    return PreparedSmallRSWeaveLaunch(
        bindings=bindings,
        artifact=artifact,
        cuda_kernel=cuda_kernel,
        prepared_launch=prepared_launch,
        stream=stream,
        artifact_receipt=_weave_artifact_receipt(artifact),
    )


_PREPARED_LAUNCH_CACHE: dict = {}
_PREPARED_LAUNCH_CACHE_LIMIT = 64


def _prepared_launch_key(args) -> tuple:
    """Exact identity of one run_e2e call: same pointers, shapes, generation and stream."""

    import torch

    (
        sym_ptrs, rank_scalar, state_generation, l1_mode2_flat, l2_mode2_flat,
        l1_global_scales, l2_global_scales, route_profile, m_scalar, h_scalar, y,
    ) = args
    return (
        torch.cuda.current_device(),
        tuple(int(v) for v in sym_ptrs.tolist()),
        int(rank_scalar.item()),
        int(state_generation.item()),
        int(route_profile.item()),
        int(m_scalar.item()),
        int(h_scalar.item()),
        l1_mode2_flat.data_ptr(), l1_mode2_flat.numel(),
        l2_mode2_flat.data_ptr(), l2_mode2_flat.numel(),
        l1_global_scales.data_ptr(), l2_global_scales.data_ptr(),
        y.data_ptr(), tuple(y.shape),
        torch.cuda.current_stream().cuda_stream,
    )


def _submit_weave_launch(*args) -> None:
    """Submit the complete production Weave path on the caller stream.

    The prepared launch (validated ABI, bound descriptors, packed arguments,
    loaded cubin) is cached per exact call identity so repeated calls with the
    same tensors and state generation only re-issue the kernel launch, like the
    production host wrapper does.
    """

    global _LAST_WEAVE_ARTIFACT_RECEIPT
    _LAST_WEAVE_ARTIFACT_RECEIPT = None
    key = _prepared_launch_key(args)
    launch = _PREPARED_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = _prepare_weave_launch(*args)
        if len(_PREPARED_LAUNCH_CACHE) >= _PREPARED_LAUNCH_CACHE_LIMIT:
            _PREPARED_LAUNCH_CACHE.clear()
        _PREPARED_LAUNCH_CACHE[key] = launch
    launch.prepared_launch.launch()
    _LAST_WEAVE_ARTIFACT_RECEIPT = launch.artifact_receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_e2e(
    sym_ptrs,
    rank_scalar,
    state_generation,
    l1_mode2_flat,
    l2_mode2_flat,
    l1_global_scales,
    l2_global_scales,
    route_profile,
    m_scalar,
    h_scalar,
    y,
) -> None:
    """Launch the complete source-exact small-RS Weave provider."""

    _submit_weave_launch(
        sym_ptrs,
        rank_scalar,
        state_generation,
        l1_mode2_flat,
        l2_mode2_flat,
        l1_global_scales,
        l2_global_scales,
        route_profile,
        m_scalar,
        h_scalar,
        y,
    )


def artifact_receipt() -> dict:
    """Return the last successful submission's strict artifact identity."""

    if _LAST_WEAVE_ARTIFACT_RECEIPT is None:
        raise RuntimeError("artifact receipt is unavailable before run_e2e")
    receipt = dict(_LAST_WEAVE_ARTIFACT_RECEIPT)
    provider_source = Path(__file__).resolve()
    receipt["provider_source"] = str(provider_source)
    receipt["provider_source_sha256"] = _sha256(provider_source)
    return receipt


if __name__ == "__main__":
    print(
        "SM90_MEGAMOE_SMALL_RS_PARTIAL_JSON "
        + json.dumps(compile_partial_weave(), sort_keys=True),
        flush=True,
    )
