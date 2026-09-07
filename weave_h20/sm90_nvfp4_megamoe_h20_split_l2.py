"""H20 large-M split-arm L2 MegaMoE Weave kernel (SM90, 78 CTAs x 384 threads).

Source: ``megamoe_nvfp4_allm_opt_h20/deep_gemm/include/deep_gemm/impls/
sm90_nvfp4_mega_moe_split_l2_body.inl`` (production split arm, launched after
the split L1 kernel on the same symmetric workspace).  Minimum arch: sm_90a.

Retained structure (exact reproduction, not "inspired by"):

* 12 warps: warp 0 issues the activation + per-128 SFA TMA transactions, warp 1
  the packed Mode2 weight TMA transaction, warps 2-3 form one 64-thread
  in-place decode team (two 80-byte packed rows per thread expanded into the
  128-byte-swizzled FP8 operand of the same stage buffer, ``bar.sync 8, 64``
  read/write rendezvous), warps 4-11 are two M64N128 math warpgroups.
  Register plan 112 (non-epilogue) / 192 (epilogue) via setmaxnreg.
* One ``kNumStages``-deep ring with three mbarrier sets: ``full`` (two
  producer arrivals), ``dequant`` (decode-team leader arrival, waited by the
  math warps) and ``empty`` (eight math-warp arrivals).
* Static grid-stride scheduler ``for_each_linear2_block``: per-lane cached
  expert token counts (volatile spin until the L1 kernel finalized the
  ``recv_count_sum`` high word), linear (m_block, n_block) enumeration over
  the experts with stride ``kNumSMs``.  Wave boundaries have no effect on the
  split L2 enumeration (a failed fetch at a wave end is immediately retried
  on the next wave), so ``kNumExpertsPerWave`` is not a schedule axis here.
* Four M64N128K32 shared/shared WGMMA per BK128 stage, one activation scale
  per row per stage applied in registers, then the per-expert L2 global scale
  at the BF16 cast.
* Warp-private 8x(128+8) BF16 staging tile and 16-lane 16-byte remote scatter
  to the symmetric combine buffer of the destination rank/token/top-k slot.
* Pre-combine NVLink rendezvous bracketed by two epilogue grid syncs, the
  source top-k combine (two raw 1-D TMA load slots + one store slot per warp,
  one or two hidden chunks), the epilogue-owned workspace cleanup and the
  final dispatch-index NVLink rendezvous (prologue grid sync only).

Two documented deviations, both value-neutral: (1) the scatter metadata
shuffle uses a converged full-warp shuffle from the 16-lane group leader
(the source uses a 16-lane group mask), so metadata for rows beyond
``valid_m`` is read (in bounds, the pool is allocated to capacity) but never
stored; (2) ``cumulative_local_expert_recv_stats`` is specialized away
because the validated hosts pass ``nullptr``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import loom
from loom.codegen.weave_ir import LaneIndexSource
from loom.weave.types import lm as LM

from loom.examples.weave.sm90_nvfp4_megamoe_h20_small_rs import (
    MBARRIER_WAIT_SUSPEND_TICKS,
    RS_LUT_WORD_VALUES,
    RS_LUT_WORDS,
)

SOURCE_FAMILY = "split_l2"
PROVIDER_KIND = "weave"

THREADS = 384
NUM_CTAS = 78
CLUSTER_DIMS = (1, 1, 1)
USE_PDL = False
SMEM_BYTES = 232448
LOOM_CONTROL_BYTES = 1024
TOP_K = 6
NUM_RANKS = 8
MAX_TOKENS_PER_RANK = 8448
WORKSPACE_SIGNAL_BYTES = 128

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 128
WG_BLOCK_M = 64
WG_BLOCK_N = 128
B_ROW_BYTES = 80
A_STAGE_BYTES = BLOCK_M * BLOCK_K
B_STAGE_BYTES = BLOCK_N * BLOCK_K
B_LOAD_BYTES = BLOCK_N * B_ROW_BYTES
SFA_STAGE_BYTES = 512  # constexpr_align(BLOCK_M * 4, 128)
L2_STAGE_ROWS = 8
L2_STAGE_ROW_PAD = 8
L2_STAGE_ROW_STRIDE = WG_BLOCK_N + L2_STAGE_ROW_PAD
CD_L2_PER_WARP_BYTES = L2_STAGE_ROWS * L2_STAGE_ROW_STRIDE * 2
LUT_SMEM_BYTES = 1024

NON_EPILOGUE_WARPS = 4
EPILOGUE_WARPS = 8
EPILOGUE_THREADS = 256
DEQUANT_THREADS = 64
NON_EPILOGUE_REGISTERS = 112
EPILOGUE_REGISTERS = 192

CTA_BARRIER_ID = 0
DISPATCH_WITH_EPILOGUE_BARRIER_ID = 1
EPILOGUE_FULL_BARRIER_ID = 2
DEQUANT_BARRIER_ID = 8
DISPATCH_GRID_SYNC_INDEX = 0
EPILOGUE_GRID_SYNC_INDEX = 1
# Optional lm.timestamp sites (no IR in ordinary tracing; expanded only by
# loom.codegen.instrument_timestamps for attribution runs).
TS_BLOCK_CAPACITY = 512
TS_KBLOCK_CAPACITY = 8192
TS_WARP = 4


class SplitL2Schedule(NamedTuple):
    hidden: int
    intermediate: int
    experts: int
    stages: int
    max_pool_tokens: int
    padded_sf_tokens: int


class SplitLayout(NamedTuple):
    workspace_bytes: int
    grid_sync_offset: int
    rendezvous_counter_offset: int
    rendezvous_signal_offset: int
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


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def split_layout(schedule: SplitL2Schedule) -> SplitLayout:
    """Reproduce ``layout::Workspace`` plus the split-body buffer slicing."""

    experts_per_rank = schedule.experts // NUM_RANKS
    max_recv_tokens_per_expert = NUM_RANKS * MAX_TOKENS_PER_RANK
    # layout::Workspace sizes its own pool from the capacity formula; the
    # kernel template pool constants come from the same formula (heuristics
    # materialize_sm90_nvfp4_mega_moe_phase), so one value serves both.
    max_pool_blocks = schedule.max_pool_tokens // 8

    expert_send_count_offset = WORKSPACE_SIGNAL_BYTES
    expert_recv_count_offset = expert_send_count_offset + schedule.experts * 8
    expert_recv_sum_offset = expert_recv_count_offset + schedule.experts * 8
    l1_arrival_count_offset = expert_recv_sum_offset + experts_per_rank * 8
    l2_arrival_mask_offset = l1_arrival_count_offset + _align(max_pool_blocks, 2) * 4
    src_token_topk_offset = l2_arrival_mask_offset + max_pool_blocks * 8
    token_src_metadata_offset = (
        src_token_topk_offset
        + experts_per_rank * NUM_RANKS * max_recv_tokens_per_expert * 4
    )
    workspace_bytes = _align(token_src_metadata_offset + schedule.max_pool_tokens * 12, 16)

    input_token_offset = workspace_bytes
    input_sf_offset = input_token_offset + MAX_TOKENS_PER_RANK * schedule.hidden
    input_topk_idx_offset = input_sf_offset + MAX_TOKENS_PER_RANK * (schedule.hidden // 32)
    input_topk_weights_offset = input_topk_idx_offset + MAX_TOKENS_PER_RANK * TOP_K * 8
    l1_token_offset = input_topk_weights_offset + MAX_TOKENS_PER_RANK * TOP_K * 4
    l1_sf_offset = l1_token_offset + schedule.max_pool_tokens * schedule.hidden
    l1_topk_weights_offset = l1_sf_offset + schedule.padded_sf_tokens * (schedule.hidden // 32)
    l2_token_offset = l1_topk_weights_offset + schedule.max_pool_tokens * 4
    l2_sf_offset = l2_token_offset + schedule.max_pool_tokens * schedule.intermediate
    combine_token_offset = l2_sf_offset + schedule.padded_sf_tokens * (schedule.intermediate // 16)
    symmetric_bytes = combine_token_offset + TOP_K * MAX_TOKENS_PER_RANK * schedule.hidden * 2
    return SplitLayout(
        workspace_bytes=workspace_bytes,
        grid_sync_offset=0,
        rendezvous_counter_offset=16,
        rendezvous_signal_offset=20,
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


def max_pool_tokens_for(experts_per_rank: int) -> int:
    """``layout::get_num_max_pool_tokens`` (kMaxCandidateBlockM = 192, LCM 384)."""

    max_recv = NUM_RANKS * MAX_TOKENS_PER_RANK
    per_token = min(TOP_K, experts_per_rank)
    return _align(max_recv * per_token + experts_per_rank * (192 - 1), 384)


def select_split_l2_schedule(hidden: int, intermediate: int, experts: int) -> SplitL2Schedule:
    """Mirror ``get_sm90_nvfp4_mega_moe_pipeline_config`` for the L2 phase."""

    experts_per_rank = experts // NUM_RANKS
    max_pool_tokens = max_pool_tokens_for(experts_per_rank)
    padded_sf_tokens = (max_pool_tokens // 8) * 128
    smem_expert_count = _align(experts * 4, 1024)
    smem_cd = _align(EPILOGUE_WARPS * CD_L2_PER_WARP_BYTES, 1024)
    smem_fixed = smem_expert_count + LUT_SMEM_BYTES + smem_cd + 2 * EPILOGUE_WARPS * 8
    # The heuristic reserves the per-64 SFA capacity (1024 B) per stage.
    smem_per_stage = A_STAGE_BYTES + B_STAGE_BYTES + 1024 + 3 * 8
    stages = (SMEM_BYTES - smem_fixed) // smem_per_stage
    if stages < 2:
        raise ValueError("split L2 needs at least two stages")
    return SplitL2Schedule(hidden, intermediate, experts, stages, max_pool_tokens, padded_sf_tokens)


class SplitL2SmemLayout(NamedTuple):
    expert_count_bytes: int
    lut_offset: int
    cd_offset: int
    a_offset: int
    b_offset: int
    sfa_offset: int
    end: int


def split_l2_smem_layout(schedule: SplitL2Schedule) -> SplitL2SmemLayout:
    expert_count_bytes = _align(schedule.experts * 4, 1024)
    lut_offset = expert_count_bytes
    cd_offset = lut_offset + LUT_SMEM_BYTES
    a_offset = cd_offset + _align(EPILOGUE_WARPS * CD_L2_PER_WARP_BYTES, 1024)
    b_offset = a_offset + schedule.stages * A_STAGE_BYTES
    sfa_offset = b_offset + schedule.stages * B_STAGE_BYTES
    end = sfa_offset + schedule.stages * SFA_STAGE_BYTES
    if end > SMEM_BYTES - LOOM_CONTROL_BYTES:
        raise ValueError(f"split L2 SMEM {end} exceeds {SMEM_BYTES - LOOM_CONTROL_BYTES}")
    return SplitL2SmemLayout(expert_count_bytes, lut_offset, cd_offset, a_offset, b_offset, sfa_offset, end)


# ---------------------------------------------------------------------------
# Scheduler fragments (MegaMoEScheduler::for_each_linear2_block)
# ---------------------------------------------------------------------------

# state[]: 0 block_idx, 1 cur_expert, 2 cur_tokens, 3 cur_pool_block_offset,
#          4 found, 5 m_block, 6 n_block
STATE_BLOCK_IDX = 0
STATE_EXPERT = 1
STATE_TOKENS = 2
STATE_POOL_OFF = 3
STATE_FOUND = 4
STATE_M_BLOCK = 5
STATE_N_BLOCK = 6


@loom.weave_fragment
def _emit_l2_fetch_expert_counts(lm, *, recv_sum_u64, stored, experts_per_rank, experts_per_lane):
    """``fetch_expert_recv_count``: per-lane cached counts after the L1 finalization."""

    for expert_group in lm.range(0, experts_per_lane, constexpr=True):
        local_expert: lm.i32 = expert_group * 32 + lm.lane_id
        ready_status: lm.u64 = 0
        if local_expert < experts_per_rank:
            ready_status = lm.sys_volatile_load(recv_sum_u64, index=local_expert, dtype=lm.u64)
            while (ready_status >> 32).u32() != NUM_CTAS * NUM_RANKS:
                ready_status = lm.sys_volatile_load(recv_sum_u64, index=local_expert, dtype=lm.u64)
        stored[expert_group] = ready_status.u32()
    lm.sync_warp()


@loom.weave_fragment
def _emit_l2_num_tokens(lm, *, stored, expert, out, experts_per_lane):
    """``get_num_tokens(expert)``: select the owning lane's cached count."""

    lane_value: lm.u32 = 0
    for expert_group in lm.range(0, experts_per_lane, constexpr=True):
        if expert == (expert_group * 32 + lm.lane_id).u32():
            lane_value = stored[expert_group]
    out[0] = lm.shfl_sync(lane_value, (expert % 32).i32(), dtype=lm.u32)


@loom.weave_fragment
def _emit_l2_scheduler_init(lm, *, stored, state, experts_per_lane):
    state[STATE_BLOCK_IDX] = lm.bid.u32()
    state[STATE_EXPERT] = 0
    state[STATE_POOL_OFF] = 0
    first_tokens: lm.u32[1]
    _emit_l2_num_tokens(lm, stored=stored, expert=state[STATE_EXPERT], out=first_tokens,
                        experts_per_lane=experts_per_lane)
    state[STATE_TOKENS] = first_tokens[0]


@loom.weave_fragment
def _emit_l2_fetch_next_block(lm, *, stored, state, experts_per_rank, experts_per_lane, l2_n_blocks):
    """``fetch_next_l2_block`` across experts plus the linear2 index bookkeeping."""

    state[STATE_FOUND] = 0
    scanning: lm.i32 = 1
    while scanning != 0:
        if state[STATE_EXPERT] < experts_per_rank:
            num_m_blocks: lm.u32 = (state[STATE_TOKENS] + (BLOCK_M - 1)) // BLOCK_M
            if state[STATE_BLOCK_IDX] < num_m_blocks * l2_n_blocks:
                state[STATE_M_BLOCK] = state[STATE_BLOCK_IDX] // l2_n_blocks
                state[STATE_N_BLOCK] = state[STATE_BLOCK_IDX] - state[STATE_M_BLOCK] * l2_n_blocks
                state[STATE_BLOCK_IDX] = state[STATE_BLOCK_IDX] + NUM_CTAS
                state[STATE_FOUND] = 1
                scanning = 0
            else:
                state[STATE_BLOCK_IDX] = state[STATE_BLOCK_IDX] - num_m_blocks * l2_n_blocks
                state[STATE_POOL_OFF] = state[STATE_POOL_OFF] + num_m_blocks
                state[STATE_EXPERT] = state[STATE_EXPERT] + 1
                next_tokens: lm.u32[1]
                _emit_l2_num_tokens(lm, stored=stored, expert=state[STATE_EXPERT], out=next_tokens,
                                    experts_per_lane=experts_per_lane)
                state[STATE_TOKENS] = next_tokens[0]
        else:
            scanning = 0


@loom.weave_fragment
def _emit_l2_pool_block_offset(lm, *, stored, expert, out, experts_per_lane):
    """``get_pool_block_offset(expert)`` (kPoolBlockM = 128) from cached counts."""

    lane_blocks: lm.u32 = 0
    for expert_group in lm.range(0, experts_per_lane, constexpr=True):
        if (expert_group * 32 + lm.lane_id).u32() < expert:
            lane_blocks = lane_blocks + (stored[expert_group] + (BLOCK_M - 1)) // BLOCK_M
    out[0] = lm.warp_redux_u32(lane_blocks, op=lm.REDUCE_ADD)


# ---------------------------------------------------------------------------
# Decode team
# ---------------------------------------------------------------------------


@loom.weave_fragment
def _emit_l2_decode_stage(lm, *, packed_stage_addr, decoded_smem, decoded_stage_addr, lut_smem, dequant_tid):
    """``dequant_smem_b_inplace_two_rows_mode2_lop3<64, 8>`` for one stage."""

    row0: lm.i32 = dequant_tid
    row1: lm.i32 = dequant_tid + DEQUANT_THREADS
    row_addr0: lm.i32 = packed_stage_addr + row0 * B_ROW_BYTES
    row_addr1: lm.i32 = packed_stage_addr + row1 * B_ROW_BYTES
    scale_words0: lm.u32[2]
    scale_words1: lm.u32[2]
    lm.smem_load_vec(dst=scale_words0, src_addr=row_addr0 + 64, count=2)
    lm.smem_load_vec(dst=scale_words1, src_addr=row_addr1 + 64, count=2)
    quads0: lm.u32[16]
    quads1: lm.u32[16]
    for quad in lm.range(0, 4, constexpr=True):
        quad_words0: lm.u32[4]
        quad_words1: lm.u32[4]
        lm.smem_load_vec(dst=quad_words0, src_addr=row_addr0 + quad * 16, count=4)
        lm.smem_load_vec(dst=quad_words1, src_addr=row_addr1 + quad * 16, count=4)
        for word in lm.range(0, 4, constexpr=True):
            quads0[quad * 4 + word] = quad_words0[word]
            quads1[quad * 4 + word] = quad_words1[word]

    # Every packed row is read before any decoded row overwrites the buffer.
    lm.sync(bar_id=DEQUANT_BARRIER_ID, threads=DEQUANT_THREADS, aligned=True)

    for quad in lm.range(0, 4, constexpr=True):
        scale_i0 = quad * 2
        scale_i1 = scale_i0 + 1
        word0: lm.u32 = scale_words0[0]
        word1: lm.u32 = scale_words1[0]
        if quad >= 2:
            word0 = scale_words0[1]
            word1 = scale_words1[1]
        s00: lm.u32 = (word0 >> ((scale_i0 & 3) * 8)) & 0x7F
        s10: lm.u32 = (word1 >> ((scale_i0 & 3) * 8)) & 0x7F
        s01: lm.u32 = (word0 >> ((scale_i1 & 3) * 8)) & 0x7F
        s11: lm.u32 = (word1 >> ((scale_i1 & 3) * 8)) & 0x7F
        lut00: lm.u32[2]
        lut10: lm.u32[2]
        lut01: lm.u32[2]
        lut11: lm.u32[2]
        lm.smem_load_vec(dst=lut00, src_addr=lut_smem.addr + s00.i32() * 8, count=2)
        lm.smem_load_vec(dst=lut10, src_addr=lut_smem.addr + s10.i32() * 8, count=2)
        lm.smem_load_vec(dst=lut01, src_addr=lut_smem.addr + s01.i32() * 8, count=2)
        lm.smem_load_vec(dst=lut11, src_addr=lut_smem.addr + s11.i32() * 8, count=2)

        q0x_hi: lm.u32 = 0
        q0x_lo: lm.u32 = 0
        q0y_hi: lm.u32 = 0
        q0y_lo: lm.u32 = 0
        q1x_hi: lm.u32 = 0
        q1x_lo: lm.u32 = 0
        q1y_hi: lm.u32 = 0
        q1y_lo: lm.u32 = 0
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q0x_hi, dst_lo=q0x_lo, packed=quads0[quad * 4 + 0],
                                       lut_lo=lut00[0], lut_hi=lut00[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q0y_hi, dst_lo=q0y_lo, packed=quads0[quad * 4 + 1],
                                       lut_lo=lut00[0], lut_hi=lut00[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q1x_hi, dst_lo=q1x_lo, packed=quads1[quad * 4 + 0],
                                       lut_lo=lut10[0], lut_hi=lut10[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q1y_hi, dst_lo=q1y_lo, packed=quads1[quad * 4 + 1],
                                       lut_lo=lut10[0], lut_hi=lut10[1])
        decoded_smem.swizzled_vec_store(row=row0, col_bytes=scale_i0 * 16,
                                        values=(q0x_hi, q0x_lo, q0y_hi, q0y_lo), base_addr=decoded_stage_addr)
        decoded_smem.swizzled_vec_store(row=row1, col_bytes=scale_i0 * 16,
                                        values=(q1x_hi, q1x_lo, q1y_hi, q1y_lo), base_addr=decoded_stage_addr)

        q0z_hi: lm.u32 = 0
        q0z_lo: lm.u32 = 0
        q0w_hi: lm.u32 = 0
        q0w_lo: lm.u32 = 0
        q1z_hi: lm.u32 = 0
        q1z_lo: lm.u32 = 0
        q1w_hi: lm.u32 = 0
        q1w_lo: lm.u32 = 0
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q0z_hi, dst_lo=q0z_lo, packed=quads0[quad * 4 + 2],
                                       lut_lo=lut01[0], lut_hi=lut01[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q0w_hi, dst_lo=q0w_lo, packed=quads0[quad * 4 + 3],
                                       lut_lo=lut01[0], lut_hi=lut01[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q1z_hi, dst_lo=q1z_lo, packed=quads1[quad * 4 + 2],
                                       lut_lo=lut11[0], lut_hi=lut11[1])
        lm.nvfp4_mode2_lut_decode_word(dst_hi=q1w_hi, dst_lo=q1w_lo, packed=quads1[quad * 4 + 3],
                                       lut_lo=lut11[0], lut_hi=lut11[1])
        decoded_smem.swizzled_vec_store(row=row0, col_bytes=scale_i1 * 16,
                                        values=(q0z_hi, q0z_lo, q0w_hi, q0w_lo), base_addr=decoded_stage_addr)
        decoded_smem.swizzled_vec_store(row=row1, col_bytes=scale_i1 * 16,
                                        values=(q1z_hi, q1z_lo, q1w_hi, q1w_lo), base_addr=decoded_stage_addr)

    # Generic-proxy stores must be visible to the async proxy (WGMMA) before
    # the team leader publishes the stage.
    lm.fence_proxy_shared_cta()
    lm.sync(bar_id=DEQUANT_BARRIER_ID, threads=DEQUANT_THREADS, aligned=True)


# ---------------------------------------------------------------------------
# Math warpgroups
# ---------------------------------------------------------------------------


@loom.weave_fragment
def _emit_l2_math_stage(lm, *, act_stage, decoded_stage, sfa_stage_addr, empty_barrier, stage,
                        row_block_offset, wg_idx, final_accum):
    """One BK128 stage of ``run_default_gemm_loop`` after the dequant wait."""

    epilogue_warp_idx: lm.i32 = lm.warp_id - NON_EPILOGUE_WARPS
    warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
    row_idx: lm.i32 = lm.lane_id // 4
    r_0: lm.i32 = warp_idx_in_wg * 16 + row_idx
    r_1: lm.i32 = r_0 + 8
    scale_0: lm.f32[1]
    scale_1: lm.f32[1]
    lm.smem_load_vec(dst=scale_0, src_addr=sfa_stage_addr + (row_block_offset + r_0) * 4, count=1)
    lm.smem_load_vec(dst=scale_1, src_addr=sfa_stage_addr + (row_block_offset + r_1) * 4, count=1)
    accum: lm.f32[64]
    accum.fill_(0.0)
    for accum_idx in lm.range(0, 64, constexpr=True):
        lm.wgmma_fence_operand(accum[accum_idx])
    lm.wgmma_fence()
    for k_slice in range(BLOCK_K // 32):
        lm.mma(
            accum,
            act_stage.tile((wg_idx * WG_BLOCK_M, k_slice * 32), (WG_BLOCK_M, 32)),
            decoded_stage.tile((0, k_slice * 32), (BLOCK_N, 32)),
            init=(k_slice == 0),
        )
    lm.wgmma_commit_group()
    for accum_idx in lm.range(0, 64, constexpr=True):
        lm.wgmma_fence_operand(accum[accum_idx])
    lm.wgmma_wait_group(n=0)
    with lm.elected_thread():
        lm.arrive(empty_barrier, stage=stage)
    for accum_chunk in lm.range(0, 16, constexpr=True):
        base = accum_chunk * 4
        final_accum[base + 0] = final_accum[base + 0] + scale_0[0] * accum[base + 0]
        final_accum[base + 1] = final_accum[base + 1] + scale_0[0] * accum[base + 1]
        final_accum[base + 2] = final_accum[base + 2] + scale_1[0] * accum[base + 2]
        final_accum[base + 3] = final_accum[base + 3] + scale_1[0] * accum[base + 3]


@loom.weave_fragment
def _emit_l2_stage_rows(lm, *, final_accum, l2_global_scale, row_accum_offset, tile_addr):
    """``stage_rows``: cast one 8-row half into the warp-private staging tile."""

    row_idx: lm.i32 = lm.lane_id // 4
    col_idx: lm.i32 = lm.lane_id % 4
    for pair_idx in lm.range(0, 8, constexpr=True):
        chunk_lo = pair_idx * 2
        chunk_hi = chunk_lo + 1
        col_lo = chunk_lo * 8 + col_idx * 2
        col_hi = chunk_hi * 8 + col_idx * 2
        values_lo: lm.f32[2]
        values_hi: lm.f32[2]
        values_lo[0] = final_accum[chunk_lo * 4 + row_accum_offset + 0] * l2_global_scale
        values_lo[1] = final_accum[chunk_lo * 4 + row_accum_offset + 1] * l2_global_scale
        values_hi[0] = final_accum[chunk_hi * 4 + row_accum_offset + 0] * l2_global_scale
        values_hi[1] = final_accum[chunk_hi * 4 + row_accum_offset + 1] * l2_global_scale
        packed_lo: lm.u32[1]
        packed_hi: lm.u32[1]
        lm.reg_array_pack(values_lo, packed_lo, size=2, dtype=lm.bf16)
        lm.reg_array_pack(values_hi, packed_hi, size=2, dtype=lm.bf16)
        lm.smem_store_vec(dst_addr=tile_addr + (row_idx * L2_STAGE_ROW_STRIDE + col_lo) * 2, src=packed_lo[0])
        lm.smem_store_vec(dst_addr=tile_addr + (row_idx * L2_STAGE_ROW_STRIDE + col_hi) * 2, src=packed_hi[0])


@loom.weave_fragment
def _emit_l2_scatter_staged_rows(lm, *, row_base, valid_m, m_idx, n_idx, tile_addr, token_src_metadata,
                                 sym_buffer, combine_token_offset, hidden):
    """``scatter_staged_rows``: 16 lanes per row, one 16-byte remote store each."""

    scatter_row_in_pair: lm.i32 = lm.lane_id // 16
    lane_in_row: lm.i32 = lm.lane_id % 16
    group_leader: lm.i32 = lm.lane_id - lane_in_row
    for j in lm.range(0, L2_STAGE_ROWS // 2, constexpr=True):
        stage_row: lm.i32 = j * 2 + scatter_row_in_pair
        row_offset: lm.i32 = row_base + stage_row
        dst_rank: lm.i32 = 0
        dst_token: lm.i32 = 0
        dst_topk: lm.i32 = 0
        if lane_in_row == 0:
            metadata_base: lm.i32 = (m_idx + row_offset) * 3
            rank_word = lm.vec_load(token_src_metadata, index=metadata_base, count=1, dtype=lm.i32, dst_dtype=lm.i32)
            token_word = lm.vec_load(token_src_metadata, index=metadata_base + 1, count=1, dtype=lm.i32,
                                     dst_dtype=lm.i32)
            topk_word = lm.vec_load(token_src_metadata, index=metadata_base + 2, count=1, dtype=lm.i32,
                                    dst_dtype=lm.i32)
            dst_rank = rank_word[0]
            dst_token = token_word[0]
            dst_topk = topk_word[0]
        dst_rank = lm.shfl_sync(dst_rank, group_leader, dtype=lm.i32)
        dst_token = lm.shfl_sync(dst_token, group_leader, dtype=lm.i32)
        dst_topk = lm.shfl_sync(dst_topk, group_leader, dtype=lm.i32)
        packed: lm.u32[4]
        lm.smem_load_vec(dst=packed, src_addr=tile_addr + stage_row * (L2_STAGE_ROW_STRIDE * 2) + lane_in_row * 16,
                         count=4)
        if row_offset < valid_m:
            peer_combine_u32 = lm.ptr_to(lm.byte_ptr(sym_buffer.peer_ptr(dst_rank)) + combine_token_offset, lm.u32)
            dst_word_offset: lm.u64 = (
                ((dst_topk.u64() * MAX_TOKENS_PER_RANK + dst_token.u64()) * hidden + n_idx.u64() + lane_in_row.u64() * 8)
                // 2
            )
            lm.gmem_store_vec(dst=peer_combine_u32 + dst_word_offset, src=packed)


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict[SplitL2Schedule, object] = {}


def split_l2_kernel(schedule: SplitL2Schedule):
    kernel = _KERNEL_CACHE.get(schedule)
    if kernel is None:
        kernel = _build_split_l2_kernel(schedule)
        _KERNEL_CACHE[schedule] = kernel
    return kernel


def _build_split_l2_kernel(schedule: SplitL2Schedule):
    layout = split_layout(schedule)
    smem = split_l2_smem_layout(schedule)
    experts_per_rank = schedule.experts // NUM_RANKS
    experts_per_lane = (experts_per_rank + 31) // 32
    l2_n_blocks = schedule.hidden // BLOCK_N
    num_k_blocks = schedule.intermediate // BLOCK_K
    stages = schedule.stages
    hidden = schedule.hidden
    combine_num_chunks = (
        1
        if (
            3 * EPILOGUE_WARPS * hidden * 2
            <= smem.expert_count_bytes + LUT_SMEM_BYTES + stages * (A_STAGE_BYTES + B_STAGE_BYTES)
            and hidden <= 32 * 128
        )
        else 2
    )
    combine_chunk_bytes = hidden * 2 // combine_num_chunks
    combine_elems_per_lane = combine_chunk_bytes // 2 // 32
    if 3 * EPILOGUE_WARPS * combine_chunk_bytes > smem.end:
        raise ValueError("combine slots exceed the pre-barrier SMEM range")

    def split_l2_fragment(
        lm,
        num_tokens,
        l2_activation,
        l2_activation_scale,
        l2_packed_weights,
        l2_global_scales,
        output_bf16,
    ):
        lut = lm.constant_array("kE2M1AndUe4m3ToFp8Lut", dtype="u32", values=RS_LUT_WORD_VALUES, align=16)
        pg = lm.process_group("pg", world_size=NUM_RANKS)
        sym_buffer = lm.symmetric_memory("sym_buffer", shape=(layout.symmetric_bytes,), dtype="u8", group=pg)
        sym_bytes = lm.byte_ptr(sym_buffer)
        workspace_u32 = lm.ptr_to(sym_bytes, lm.u32)
        send_count_u64 = lm.ptr_to(sym_bytes + layout.expert_send_count_offset, lm.u64)
        recv_count_u64 = lm.ptr_to(sym_bytes + layout.expert_recv_count_offset, lm.u64)
        recv_sum_u64 = lm.ptr_to(sym_bytes + layout.expert_recv_sum_offset, lm.u64)
        token_src_metadata_i32 = lm.ptr_to(sym_bytes + layout.token_src_metadata_offset, lm.i32)
        input_topk_idx_i64 = lm.ptr_to(sym_bytes + layout.input_topk_idx_offset, lm.i64)
        l1_arrival_u32 = lm.ptr_to(sym_bytes + layout.l1_arrival_count_offset, lm.u32)

        pool = lm.smem(SMEM_BYTES - LOOM_CONTROL_BYTES)
        smem_expert_count = pool.view(offset=0, shape=(schedule.experts,), dtype=lm.u32)
        lut_smem = pool.view(offset=smem.lut_offset, shape=(RS_LUT_WORDS,), dtype=lm.u32, swizzle=None)
        cd_smem = pool.view(offset=smem.cd_offset, shape=(EPILOGUE_WARPS * CD_L2_PER_WARP_BYTES,), dtype=lm.u8,
                            swizzle=None)
        a_smem = pool.view(
            offset=smem.a_offset,
            shape=(BLOCK_M, BLOCK_K),
            dtype=lm.u8,
            logical_shape=(BLOCK_M, BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=stages,
            stride=A_STAGE_BYTES,
        )
        # The packed 80-byte rows land at the start of each B stage and are
        # expanded in place into the 128-byte-swizzled FP8 operand.
        b_packed_smem = pool.view(offset=smem.b_offset, shape=(B_LOAD_BYTES,), dtype=lm.u8, swizzle=None,
                                  stage=stages, stride=B_STAGE_BYTES)
        b_decoded_smem = pool.view(
            offset=smem.b_offset,
            shape=(BLOCK_N, BLOCK_K),
            dtype=lm.u8,
            logical_shape=(BLOCK_N, BLOCK_K),
            logical_dtype=lm.fp8_e4m3,
            swizzle=lm.swizzle_128b,
            stage=stages,
            stride=B_STAGE_BYTES,
        )
        sfa_smem = pool.view(offset=smem.sfa_offset, shape=(BLOCK_M,), dtype=lm.f32, swizzle=None, stage=stages,
                             stride=SFA_STAGE_BYTES)
        # After GEMM and scatter the source reuses the pre-barrier range for
        # two per-warp combine load slots and one store slot.
        combine_load_smem = pool.view(offset=0, shape=(combine_chunk_bytes,), dtype=lm.u8, swizzle=lm.swizzle_none,
                                      stage=EPILOGUE_WARPS * 2, stride=combine_chunk_bytes)
        combine_store_smem = pool.view(offset=EPILOGUE_WARPS * 2 * combine_chunk_bytes, shape=(combine_chunk_bytes,),
                                       dtype=lm.u8, swizzle=lm.swizzle_none, stage=EPILOGUE_WARPS,
                                       stride=combine_chunk_bytes)

        loader = lm.role("loader", warps=[0, 1])
        decode = lm.role("decode", warps=[2, 3])
        math = lm.role("math", warps=[4, 5, 6, 7, 8, 9, 10, 11])
        pipe = lm.pipeline(stages=stages)
        stage_full = lm.barrier(count=stages, prod=[loader], cons=[decode], init_count=2, pipeline=pipe)
        stage_empty = lm.barrier(count=stages, prod=[math], cons=[loader], init_count=EPILOGUE_WARPS, init_phase=1,
                                 pipeline=pipe)
        stage_dequant = lm.barrier(count=stages, prod=[decode], cons=[math], init_count=1, pipeline=pipe)
        combine_pipe = lm.pipeline(stages=EPILOGUE_WARPS * 2)
        combine_full = lm.barrier(count=EPILOGUE_WARPS * 2, prod=[math], cons=[math], init_count=1,
                                  pipeline=combine_pipe)

        with loader:
            lm.setmaxnreg_dealloc(NON_EPILOGUE_REGISTERS)
            # Source: threads < 64 stage the 1024-byte LUT, warp 0 clears the
            # expert-count scratch, then __syncthreads.
            lm.constant_array_copy16b(lut, dst_addr=lut_smem.addr + lm.tid * 16, chunk=lm.tid)
            if lm.warp_id == 0:
                for expert in lm.range(lm.lane_id, schedule.experts, 32, unroll=1, dtype=lm.i32):
                    lm.smem_store_vec(dst_addr=smem_expert_count.addr + expert * 4, src=0)
            lm.sync(bar_id=CTA_BARRIER_ID, threads=THREADS)

            stored: lm.u32[experts_per_lane]
            _emit_l2_fetch_expert_counts(lm, recv_sum_u64=recv_sum_u64, stored=stored,
                                         experts_per_rank=experts_per_rank, experts_per_lane=experts_per_lane)
            state: lm.u32[8]
            _emit_l2_scheduler_init(lm, stored=stored, state=state, experts_per_lane=experts_per_lane)
            load_stage: lm.u32 = 0
            while state[STATE_EXPERT] < experts_per_rank:
                _emit_l2_fetch_next_block(lm, stored=stored, state=state, experts_per_rank=experts_per_rank,
                                          experts_per_lane=experts_per_lane, l2_n_blocks=l2_n_blocks)
                if state[STATE_FOUND] != 0:
                    pool_token_idx: lm.i32 = ((state[STATE_POOL_OFF] + state[STATE_M_BLOCK]) * BLOCK_M).i32()
                    valid_m: lm.i32 = lm.min(
                        (state[STATE_TOKENS] - state[STATE_M_BLOCK] * BLOCK_M).i32(), BLOCK_M
                    )
                    weight_row: lm.i32 = (state[STATE_EXPERT] * hidden + state[STATE_N_BLOCK] * BLOCK_N).i32()
                    for k_block_idx in lm.range(0, num_k_blocks, unroll=1, dtype=lm.i32):
                        lm.wait(stage_empty, stage=load_stage)
                        if lm.warp_id == 0:
                            if valid_m > 0:
                                with lm.elected_thread():
                                    lm.tma_load(
                                        l2_activation,
                                        dst=a_smem.stage_addr(load_stage),
                                        coords=(k_block_idx * BLOCK_K, pool_token_idx),
                                        barrier=stage_full,
                                        stage=load_stage,
                                    )
                                    lm.tma_load(
                                        l2_activation_scale,
                                        dst=sfa_smem.stage_addr(load_stage),
                                        coords=(pool_token_idx, k_block_idx),
                                        barrier=stage_full,
                                        stage=load_stage,
                                    )
                                    lm.arrive_expect_tx(stage_full, tx_bytes=A_STAGE_BYTES + BLOCK_M * 4,
                                                        stage=load_stage)
                            else:
                                with lm.elected_thread():
                                    lm.arrive(stage_full, stage=load_stage)
                        else:
                            with lm.elected_thread():
                                lm.tma_load(
                                    l2_packed_weights,
                                    dst=b_packed_smem.stage_addr(load_stage),
                                    coords=(k_block_idx * B_ROW_BYTES, weight_row),
                                    barrier=stage_full,
                                    stage=load_stage,
                                )
                                lm.arrive_expect_tx(stage_full, tx_bytes=B_LOAD_BYTES, stage=load_stage)
                        lm.sync_warp()
                        lm.advance(load_stage, pipe)

        with decode:
            lm.setmaxnreg_dealloc(NON_EPILOGUE_REGISTERS)
            lm.sync(bar_id=CTA_BARRIER_ID, threads=THREADS)
            dequant_tid: lm.i32 = lm.tid - 64
            stored_d: lm.u32[experts_per_lane]
            _emit_l2_fetch_expert_counts(lm, recv_sum_u64=recv_sum_u64, stored=stored_d,
                                         experts_per_rank=experts_per_rank, experts_per_lane=experts_per_lane)
            state_d: lm.u32[8]
            _emit_l2_scheduler_init(lm, stored=stored_d, state=state_d, experts_per_lane=experts_per_lane)
            decode_stage: lm.u32 = 0
            while state_d[STATE_EXPERT] < experts_per_rank:
                _emit_l2_fetch_next_block(lm, stored=stored_d, state=state_d, experts_per_rank=experts_per_rank,
                                          experts_per_lane=experts_per_lane, l2_n_blocks=l2_n_blocks)
                if state_d[STATE_FOUND] != 0:
                    for _k_block_idx in lm.range(0, num_k_blocks, unroll=1, dtype=lm.i32):
                        lm.wait(stage_full, stage=decode_stage)
                        _emit_l2_decode_stage(
                            lm,
                            packed_stage_addr=b_packed_smem.stage_addr(decode_stage),
                            decoded_smem=b_decoded_smem,
                            decoded_stage_addr=b_decoded_smem.stage_addr(decode_stage),
                            lut_smem=lut_smem,
                            dequant_tid=dequant_tid,
                        )
                        if dequant_tid == 0:
                            lm.arrive(stage_dequant, stage=decode_stage)
                        lm.sync_warp()
                        lm.advance(decode_stage, pipe)

        with math:
            lm.setmaxnreg_alloc(EPILOGUE_REGISTERS)
            lm.sync(bar_id=CTA_BARRIER_ID, threads=THREADS)
            epilogue_warp_idx: lm.i32 = lm.warp_id - NON_EPILOGUE_WARPS
            epilogue_wg_idx: lm.i32 = epilogue_warp_idx // 4
            epilogue_thread_idx: lm.i32 = lm.tid - NON_EPILOGUE_WARPS * 32
            warp_idx_in_wg: lm.i32 = epilogue_warp_idx % 4
            row_block_offset: lm.i32 = epilogue_wg_idx * WG_BLOCK_M
            tile_addr: lm.i32 = cd_smem.addr + epilogue_warp_idx * CD_L2_PER_WARP_BYTES
            stored_m: lm.u32[experts_per_lane]
            _emit_l2_fetch_expert_counts(lm, recv_sum_u64=recv_sum_u64, stored=stored_m,
                                         experts_per_rank=experts_per_rank, experts_per_lane=experts_per_lane)
            state_m: lm.u32[8]
            _emit_l2_scheduler_init(lm, stored=stored_m, state=state_m, experts_per_lane=experts_per_lane)
            math_stage: lm.u32 = 0
            ts_block: lm.u32 = 0
            ts_kblock: lm.u32 = 0
            lm.sync(bar_id=DISPATCH_WITH_EPILOGUE_BARRIER_ID, threads=EPILOGUE_THREADS)
            lm.timestamp("math_loop", field="begin", record=0, capacity=1, warp=TS_WARP)
            while state_m[STATE_EXPERT] < experts_per_rank:
                _emit_l2_fetch_next_block(lm, stored=stored_m, state=state_m, experts_per_rank=experts_per_rank,
                                          experts_per_lane=experts_per_lane, l2_n_blocks=l2_n_blocks)
                if state_m[STATE_FOUND] != 0:
                    lm.timestamp("block", field="begin", record=ts_block, capacity=TS_BLOCK_CAPACITY, warp=TS_WARP)
                    local_expert_idx: lm.i32 = state_m[STATE_EXPERT].i32()
                    m_idx: lm.i32 = ((state_m[STATE_POOL_OFF] + state_m[STATE_M_BLOCK]) * BLOCK_M).i32()
                    valid_m_m: lm.i32 = lm.min(
                        (state_m[STATE_TOKENS] - state_m[STATE_M_BLOCK] * BLOCK_M).i32(), BLOCK_M
                    )
                    n_idx: lm.i32 = (state_m[STATE_N_BLOCK] * BLOCK_N).i32()
                    if row_block_offset >= valid_m_m:
                        for _k_skip in lm.range(0, num_k_blocks, unroll=1, dtype=lm.i32):
                            lm.wait(stage_dequant, stage=math_stage)
                            with lm.elected_thread():
                                lm.arrive(stage_empty, stage=math_stage)
                            lm.sync_warp()
                            lm.advance(math_stage, pipe)
                        lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
                    else:
                        final_accum: lm.f32[64]
                        final_accum.fill_(0.0)
                        for _k_block_idx in lm.range(0, num_k_blocks, unroll=1, dtype=lm.i32):
                            lm.timestamp("dequant_wait", field="begin", record=ts_kblock, capacity=TS_KBLOCK_CAPACITY,
                                         warp=TS_WARP)
                            lm.wait(stage_dequant, stage=math_stage)
                            lm.timestamp("dequant_wait", field="end", record=ts_kblock, capacity=TS_KBLOCK_CAPACITY,
                                         warp=TS_WARP)
                            lm.fence_proxy_shared_cta()
                            lm.timestamp("stage_body", field="begin", record=ts_kblock, capacity=TS_KBLOCK_CAPACITY,
                                         warp=TS_WARP)
                            _emit_l2_math_stage(
                                lm,
                                act_stage=a_smem[math_stage],
                                decoded_stage=b_decoded_smem[math_stage],
                                sfa_stage_addr=sfa_smem.stage_addr(math_stage),
                                empty_barrier=stage_empty,
                                stage=math_stage,
                                row_block_offset=row_block_offset,
                                wg_idx=epilogue_wg_idx,
                                final_accum=final_accum,
                            )
                            lm.timestamp("stage_body", field="end", record=ts_kblock, capacity=TS_KBLOCK_CAPACITY,
                                         warp=TS_WARP)
                            ts_kblock = ts_kblock + 1
                            lm.advance(math_stage, pipe)

                        lm.timestamp("epilogue", field="begin", record=ts_block, capacity=TS_BLOCK_CAPACITY, warp=TS_WARP)
                        global_scale_vec = lm.vec_load(l2_global_scales, index=local_expert_idx, count=1,
                                                       dtype=lm.f32, dst_dtype=lm.f32, cache_hint="read_only")
                        l2_global_scale: lm.f32 = global_scale_vec[0]
                        warp_row_base: lm.i32 = row_block_offset + warp_idx_in_wg * 16
                        row_idx_m: lm.i32 = lm.lane_id // 4
                        # The tile is warp-private: __syncwarp orders the
                        # staging stores against the 16-byte read-back.
                        if warp_row_base + row_idx_m < valid_m_m:
                            _emit_l2_stage_rows(lm, final_accum=final_accum, l2_global_scale=l2_global_scale,
                                                row_accum_offset=0, tile_addr=tile_addr)
                        lm.sync_warp()
                        _emit_l2_scatter_staged_rows(
                            lm, row_base=warp_row_base, valid_m=valid_m_m, m_idx=m_idx, n_idx=n_idx,
                            tile_addr=tile_addr, token_src_metadata=token_src_metadata_i32, sym_buffer=sym_buffer,
                            combine_token_offset=layout.combine_token_offset, hidden=hidden,
                        )
                        lm.sync_warp()
                        if warp_row_base + row_idx_m + 8 < valid_m_m:
                            _emit_l2_stage_rows(lm, final_accum=final_accum, l2_global_scale=l2_global_scale,
                                                row_accum_offset=2, tile_addr=tile_addr)
                        lm.sync_warp()
                        _emit_l2_scatter_staged_rows(
                            lm, row_base=warp_row_base + L2_STAGE_ROWS, valid_m=valid_m_m, m_idx=m_idx, n_idx=n_idx,
                            tile_addr=tile_addr, token_src_metadata=token_src_metadata_i32, sym_buffer=sym_buffer,
                            combine_token_offset=layout.combine_token_offset, hidden=hidden,
                        )
                        lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
                        lm.timestamp("epilogue", field="end", record=ts_block, capacity=TS_BLOCK_CAPACITY, warp=TS_WARP)
                    lm.timestamp("block", field="end", record=ts_block, capacity=TS_BLOCK_CAPACITY, warp=TS_WARP)
                    ts_block = ts_block + 1
            lm.timestamp("math_loop", field="end", record=0, capacity=1, warp=TS_WARP)

            # ---------------- COMBINE ----------------
            lm.timestamp("pre_combine_sync", field="begin", record=0, capacity=1, warp=TS_WARP)
            # kBeforeCombineReduceBarrierTag: epilogue grid sync, stateful
            # NVLink rendezvous, epilogue grid sync.
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            with lm.elected_thread(warp=4):
                lm.grid_sync_workspace(
                    counter=workspace_u32 + layout.grid_sync_offset // 4 + EPILOGUE_GRID_SYNC_INDEX,
                    num_ctas=NUM_CTAS,
                    leader=lm.bid == 0,
                )
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            if lm.logical_and(lm.bid == 0, lm.warp_id == 4):
                lm.nvlink_rendezvous(
                    counter=workspace_u32 + layout.rendezvous_counter_offset // 4,
                    signal_index=(layout.rendezvous_signal_offset - layout.rendezvous_counter_offset) // 4,
                    sym_buf=sym_buffer,
                    signal_offset=layout.rendezvous_signal_offset,
                    group=pg,
                )
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            with lm.elected_thread(warp=4):
                lm.grid_sync_workspace(
                    counter=workspace_u32 + layout.grid_sync_offset // 4 + EPILOGUE_GRID_SYNC_INDEX,
                    num_ctas=NUM_CTAS,
                    leader=lm.bid == 0,
                )
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            # Rendezvous before workspace cleanup (no dispatch warps in L2:
            # the 256 epilogue threads meet themselves on barrier 1).
            lm.sync(bar_id=DISPATCH_WITH_EPILOGUE_BARRIER_ID, threads=EPILOGUE_THREADS)
            lm.timestamp("pre_combine_sync", field="end", record=0, capacity=1, warp=TS_WARP)
            lm.timestamp("combine", field="begin", record=0, capacity=1, warp=TS_WARP)

            combine_phase: lm.u32 = 0
            combine_load_stage: lm.u32 = 0
            for combine_token_idx in lm.range(
                lm.bid * EPILOGUE_WARPS + epilogue_warp_idx, num_tokens, NUM_CTAS * EPILOGUE_WARPS, unroll=1,
                dtype=lm.u32,
            ):
                stored_topk_slot_idx: lm.i32 = -1
                if lm.lane_id < TOP_K:
                    stored_topk_slot = lm.vec_load(input_topk_idx_i64, index=combine_token_idx * TOP_K + lm.lane_id,
                                                   count=1, dtype=lm.i64, dst_dtype=lm.i64, cache_hint="read_only")
                    stored_topk_slot_idx = stored_topk_slot[0].i32()
                total_mask: lm.u32 = lm.warp_vote(vote=lm.VOTE_BALLOT, predicate=stored_topk_slot_idx >= 0,
                                                  mask=0xFFFFFFFF)
                for combine_chunk in lm.range(0, combine_num_chunks, constexpr=True):
                    chunk_byte_offset: lm.u32 = combine_chunk * combine_chunk_bytes
                    combine_mask: lm.u32 = total_mask
                    do_reduce: lm.i32 = 0
                    if combine_mask != 0:
                        slot_idx: lm.i32 = lm.ffs(combine_mask) - 1
                        combine_mask = combine_mask ^ (1 << slot_idx)
                        load_barrier_stage: lm.u32 = epilogue_warp_idx * 2 + combine_load_stage
                        with lm.elected_thread():
                            lm.arrive_expect_tx(combine_full, stage=load_barrier_stage, tx_bytes=combine_chunk_bytes)
                            lm.copy(
                                src=(
                                    sym_bytes
                                    + layout.combine_token_offset
                                    + (slot_idx.u64() * MAX_TOKENS_PER_RANK + combine_token_idx.u64()) * hidden * 2
                                    + chunk_byte_offset
                                ),
                                dst=combine_load_smem.stage_addr(load_barrier_stage),
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
                            next_slot_idx: lm.i32 = lm.ffs(combine_mask) - 1
                            combine_mask = combine_mask ^ (1 << next_slot_idx)
                            next_load_stage: lm.u32 = combine_load_stage ^ 1
                            next_barrier_stage: lm.u32 = epilogue_warp_idx * 2 + next_load_stage
                            with lm.elected_thread():
                                lm.arrive_expect_tx(combine_full, stage=next_barrier_stage,
                                                    tx_bytes=combine_chunk_bytes)
                                lm.copy(
                                    src=(
                                        sym_bytes
                                        + layout.combine_token_offset
                                        + (next_slot_idx.u64() * MAX_TOKENS_PER_RANK + combine_token_idx.u64())
                                        * hidden
                                        * 2
                                        + chunk_byte_offset
                                    ),
                                    dst=combine_load_smem.stage_addr(next_barrier_stage),
                                    bytes=combine_chunk_bytes,
                                    barrier=combine_full,
                                    stage=next_barrier_stage,
                                )
                            lm.sync_warp()
                            do_reduce = 1

                        current_barrier_stage: lm.u32 = epilogue_warp_idx * 2 + combine_load_stage
                        lm.wait(combine_full, stage=current_barrier_stage, phase=combine_phase)
                        lm.fence_proxy_shared_cta()
                        for combine_vec in lm.range(0, combine_elems_per_lane // 8, constexpr=True):
                            combine_packed: lm.u32[4]
                            lm.smem_load_vec(
                                dst=combine_packed,
                                src_addr=combine_load_smem.stage_addr(current_barrier_stage)
                                + (combine_vec * 32 + lm.lane_id) * 16,
                                count=4,
                            )
                            combine_values = combine_packed.f32()
                            for combine_elem in lm.range(0, 8, constexpr=True):
                                reduced[combine_vec * 8 + combine_elem] = (
                                    reduced[combine_vec * 8 + combine_elem] + combine_values[combine_elem]
                                )
                        combine_phase = combine_phase ^ combine_load_stage
                        combine_load_stage = combine_load_stage ^ 1

                    lm.bulk_wait(n=0)
                    lm.sync_warp()
                    for combine_vec in lm.range(0, combine_elems_per_lane // 8, constexpr=True):
                        cast_values: lm.f32[8]
                        for cast_elem in lm.range(0, 8, constexpr=True):
                            cast_values[cast_elem] = reduced[combine_vec * 8 + cast_elem]
                        casted: lm.u32[4]
                        lm.reg_array_pack(cast_values, casted, size=8, dtype=lm.bf16)
                        for cast_word in lm.range(0, 4, constexpr=True):
                            lm.smem_store_vec(
                                dst_addr=combine_store_smem.stage_addr(epilogue_warp_idx)
                                + (combine_vec * 32 + lm.lane_id) * 16
                                + cast_word * 4,
                                src=casted[cast_word],
                            )
                    lm.sync_warp()
                    with lm.elected_thread():
                        lm.fence_proxy_shared_cta()
                        lm.bulk_store(
                            dst=output_bf16 + combine_token_idx.u64() * hidden + chunk_byte_offset // 2,
                            src=combine_store_smem.stage_addr(epilogue_warp_idx),
                            bytes=combine_chunk_bytes,
                        )
                        lm.bulk_commit()
                    lm.sync_warp()

            # ---------------- finish_no_dispatch_cleanup ----------------
            lm.sync(bar_id=DISPATCH_WITH_EPILOGUE_BARRIER_ID, threads=EPILOGUE_THREADS)
            lm.timestamp("combine", field="end", record=0, capacity=1, warp=TS_WARP)
            lm.timestamp("cleanup_sync", field="begin", record=0, capacity=1, warp=TS_WARP)
            if lm.bid == 0:
                for expert in lm.range(epilogue_thread_idx, schedule.experts, EPILOGUE_THREADS, unroll=1,
                                       dtype=lm.i32):
                    lm.gmem_store(send_count_u64, 0, index=expert, dtype=lm.u64, src_dtype=lm.u64)
            else:
                for cleanup_expert in lm.range(lm.bid - 1, experts_per_rank, NUM_CTAS - 1, unroll=1, dtype=lm.i32):
                    cleanup_recv = lm.vec_load(recv_sum_u64, index=cleanup_expert, count=1, dtype=lm.u64,
                                               dst_dtype=lm.u64)
                    cleanup_num_tokens: lm.u32 = cleanup_recv[0].u32()
                    cleanup_num_m_blocks: lm.u32 = (cleanup_num_tokens + (BLOCK_M - 1)) // BLOCK_M
                    cleanup_pool_block_offset: lm.u32[1]
                    _emit_l2_pool_block_offset(lm, stored=stored_m, expert=cleanup_expert.u32(),
                                               out=cleanup_pool_block_offset, experts_per_lane=experts_per_lane)
                    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
                    if epilogue_thread_idx == 0:
                        lm.gmem_store(recv_sum_u64, 0, index=cleanup_expert, dtype=lm.u64, src_dtype=lm.u64)
                    if epilogue_thread_idx < NUM_RANKS:
                        lm.gmem_store(recv_count_u64, 0, index=epilogue_thread_idx * experts_per_rank + cleanup_expert,
                                      dtype=lm.u64, src_dtype=lm.u64)
                    for cleanup_block in lm.range(epilogue_thread_idx.u32(), cleanup_num_m_blocks, EPILOGUE_THREADS,
                                                  unroll=1, dtype=lm.u32):
                        lm.gmem_store(l1_arrival_u32, 0, index=cleanup_pool_block_offset[0] + cleanup_block,
                                      dtype=lm.u32, src_dtype=lm.u32)
                    lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)

            # kAfterWorkspaceCleanBarrierTag: dispatch-index grid sync, then
            # the stateful rendezvous; the source skips the epilogue grid sync.
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            with lm.elected_thread(warp=4):
                lm.grid_sync_workspace(
                    counter=workspace_u32 + layout.grid_sync_offset // 4 + DISPATCH_GRID_SYNC_INDEX,
                    num_ctas=NUM_CTAS,
                    leader=lm.bid == 0,
                )
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            if lm.logical_and(lm.bid == 0, lm.warp_id == 4):
                lm.nvlink_rendezvous(
                    counter=workspace_u32 + layout.rendezvous_counter_offset // 4,
                    signal_index=(layout.rendezvous_signal_offset - layout.rendezvous_counter_offset) // 4,
                    sym_buf=sym_buffer,
                    signal_offset=layout.rendezvous_signal_offset,
                    group=pg,
                )
            lm.sync(bar_id=EPILOGUE_FULL_BARRIER_ID, threads=EPILOGUE_THREADS, aligned=True)
            lm.timestamp("cleanup_sync", field="end", record=0, capacity=1, warp=TS_WARP)

    split_l2_fragment.__annotations__ = {
        "num_tokens": LM.i32,
        "l2_activation": LM.tma2d(dtype=LM.u8, box_shape=(BLOCK_K, BLOCK_M), swizzle=LM.swizzle_128b,
                                  l2_promotion="l2_256b"),
        "l2_activation_scale": LM.tma2d(dtype=LM.f32, box_shape=(BLOCK_M, 1), swizzle=LM.swizzle_none,
                                        l2_promotion="l2_256b"),
        "l2_packed_weights": LM.tma2d(dtype=LM.u8, box_shape=(B_ROW_BYTES, BLOCK_N), swizzle=LM.swizzle_none,
                                      l2_promotion="l2_256b"),
        "l2_global_scales": LM.ptr[LM.f32],
        "output_bf16": LM.ptr[LM.bf16],
    }
    return loom.weave(
        threads=THREADS,
        cluster_dims=CLUSTER_DIMS,
        mbarrier_wait_suspend_ticks=MBARRIER_WAIT_SUSPEND_TICKS,
        lane_index_source=LaneIndexSource.THREAD_INDEX,
    )(split_l2_fragment)


# ---------------------------------------------------------------------------
# Host path
# ---------------------------------------------------------------------------


class CompiledSplitL2Artifact(NamedTuple):
    schedule: SplitL2Schedule
    arch: str
    kernel: object
    generated_source: str
    generated_source_sha256: str
    cubin: bytes
    cubin_sha256: str
    kernel_symbol: str


_ARTIFACT_CACHE: dict[tuple, CompiledSplitL2Artifact] = {}
_CUDA_KERNEL_CACHE: dict[tuple, object] = {}


def generate_split_l2_source(schedule: SplitL2Schedule) -> str:
    from loom.codegen.kernel import generate_kernel

    return generate_kernel(split_l2_kernel(schedule), validate=True)


def compile_split_l2(schedule: SplitL2Schedule, *, arch: str | None = None) -> CompiledSplitL2Artifact:
    from loom.examples.gemm_common import _cuda_include_dirs
    from loom.runtime.compiler import compile_cuda, detect_gpu_arch

    arch = arch or detect_gpu_arch()
    key = (arch, schedule)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached
    kernel = split_l2_kernel(schedule)
    source = generate_split_l2_source(schedule)
    cubin = compile_cuda(source, arch=arch, options=["--use_fast_math"], include_dirs=_cuda_include_dirs())
    artifact = CompiledSplitL2Artifact(
        schedule=schedule,
        arch=arch,
        kernel=kernel,
        generated_source=source,
        generated_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        cubin=cubin,
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        kernel_symbol=f"kernel_{kernel.symbol}",
    )
    _ARTIFACT_CACHE[key] = artifact
    return artifact


class PreparedSplitL2Launch(NamedTuple):
    schedule: SplitL2Schedule
    layout: SplitLayout
    artifact: CompiledSplitL2Artifact
    bindings: dict[str, object]
    packed_args: list[object]
    prepared_launch: object
    keepalive: tuple[object, ...]


def prepare_split_l2_launch(
    *,
    schedule: SplitL2Schedule,
    local_base_ptr: int,
    peer_ptrs,
    rank: int,
    num_tokens: int,
    l2_packed_weights,
    l2_global_scales,
    y,
) -> PreparedSplitL2Launch:
    """Bind the L2 kernel to the production symmetric workspace of this rank."""

    import torch

    from loom.runtime.external_memory import ExternalDeviceAllocation
    from loom.runtime.kernel import CUDAKernel
    from loom.runtime.launch import create_tma_from_external_slice, create_tma_from_spec, pack_kernel_args

    layout = split_layout(schedule)
    artifact = compile_split_l2(schedule)
    kernel = artifact.kernel
    descriptors = {str(buffer.resource): buffer.tma for buffer in kernel.buffers if buffer.tma is not None}
    allocation = ExternalDeviceAllocation.from_pointer(int(local_base_ptr), layout.symmetric_bytes, label="sym_buffer")
    peer_ptrs = torch.as_tensor(peer_ptrs, dtype=torch.int64).to(device="cuda")
    experts_per_rank = schedule.experts // NUM_RANKS
    tensor_maps = {
        "l2_activation": create_tma_from_external_slice(
            descriptors["l2_activation"],
            allocation.slice(layout.l2_token_offset, dtype="u8", shape=(schedule.max_pool_tokens, schedule.intermediate),
                             row_pitch_bytes=schedule.intermediate, label="l2_activation"),
        ),
        "l2_activation_scale": create_tma_from_external_slice(
            descriptors["l2_activation_scale"],
            allocation.slice(layout.l2_sf_offset, dtype="f32",
                             shape=(schedule.intermediate // BLOCK_K, schedule.padded_sf_tokens),
                             row_pitch_bytes=schedule.padded_sf_tokens * 4, label="l2_activation_scale"),
        ),
        "l2_packed_weights": create_tma_from_spec(
            descriptors["l2_packed_weights"],
            l2_packed_weights.view(experts_per_rank * schedule.hidden, schedule.intermediate // BLOCK_K * B_ROW_BYTES),
        ),
    }
    bindings = {
        "num_tokens": int(num_tokens),
        **tensor_maps,
        "l2_global_scales": l2_global_scales,
        "output_bf16": y,
        "pg_world": NUM_RANKS,
        "pg_rank": int(rank),
        "pg_flags": peer_ptrs,
        "sym_buffer": allocation,
        "sym_buffer_peers": peer_ptrs,
    }
    packed_args = pack_kernel_args(kernel, **bindings)
    device = torch.cuda.current_device()
    ck_key = (device, artifact.cubin_sha256, artifact.kernel_symbol)
    cuda_kernel = _CUDA_KERNEL_CACHE.get(ck_key)
    if cuda_kernel is None:
        cuda_kernel = CUDAKernel(artifact.cubin, artifact.kernel_symbol)
        _CUDA_KERNEL_CACHE[ck_key] = cuda_kernel
    prepared = cuda_kernel.prepare_launch(
        grid=(NUM_CTAS, 1, 1),
        block=(THREADS, 1, 1),
        args=packed_args,
        shared_mem=SMEM_BYTES,
        stream=torch.cuda.current_stream(),
        use_pdl=USE_PDL,
    )
    return PreparedSplitL2Launch(
        schedule=schedule,
        layout=layout,
        artifact=artifact,
        bindings=bindings,
        packed_args=packed_args,
        prepared_launch=prepared,
        keepalive=(peer_ptrs, l2_packed_weights, l2_global_scales, y, allocation, *tensor_maps.values()),
    )


def compile_partial_weave() -> dict:
    """Headless generate + compile for both production model families."""

    from loom.runtime.compiler import detect_gpu_arch

    arch = detect_gpu_arch()
    receipts = {}
    for hidden, intermediate, experts in ((4096, 2048, 256), (7168, 3072, 384)):
        schedule = select_split_l2_schedule(hidden, intermediate, experts)
        artifact = compile_split_l2(schedule, arch=arch)
        receipts[f"h{hidden}"] = {
            "schedule": schedule._asdict(),
            "generated_source_sha256": artifact.generated_source_sha256,
            "cubin_sha256": artifact.cubin_sha256,
            "kernel_symbol": artifact.kernel_symbol,
            "generated_bytes": len(artifact.generated_source),
        }
    return receipts


if __name__ == "__main__":
    import json

    print(json.dumps(compile_partial_weave(), indent=2))
