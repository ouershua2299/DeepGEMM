typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) LoomTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) LoomTensorMapPack { LoomTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define LOOM_INF CUDART_INF_F
#define NUM_TASK_INFO_PIPE_STAGES 2
#define NUM_L1_PIPE_STAGES 6
#define NUM_COMBINE_PIPE_STAGES 16
#define SMEM_SMEM_EXPERT_COUNT_OFF 1024
#define SMEM_SMEM_EXPERT_COUNT_STAGE_BYTES 1536
#define SMEM_SMEM_EXPERT_COUNT_STRIDE 1536
#define SMEM_PULL_STAGE_OFF 3072
#define SMEM_PULL_STAGE_STAGE_BYTES 14336
#define SMEM_PULL_STAGE_STRIDE 14336
#define SMEM_TASK_INFO_SMEM_OFF 17408
#define SMEM_TASK_INFO_SMEM_STAGE_BYTES 32
#define SMEM_TASK_INFO_SMEM_STRIDE 32
#define SMEM_L1_ACT_SMEM_OFF 18432
#define SMEM_L1_ACT_SMEM_STAGE_BYTES 1024
#define SMEM_L1_ACT_SMEM_STRIDE 1024
#define SMEM_L1_SFA_SMEM_OFF 24576
#define SMEM_L1_SFA_SMEM_STAGE_BYTES 32
#define SMEM_L1_SFA_SMEM_STRIDE 128
#define SMEM_L1_PACKED_SMEM_OFF 25344
#define SMEM_L1_PACKED_SMEM_STAGE_BYTES 20480
#define SMEM_L1_PACKED_SMEM_STRIDE 20480
#define SMEM_L1_LUT_SMEM_OFF 148224
#define SMEM_L1_LUT_SMEM_STAGE_BYTES 1024
#define SMEM_L1_LUT_SMEM_STRIDE 1024
#define SMEM_L1_DECODED_SMEM_OFF 149504
#define SMEM_L1_DECODED_SMEM_STAGE_BYTES 32768
#define SMEM_L1_DECODED_SMEM_STRIDE 32768
#define SMEM_L1_OUTPUT_SMEM_OFF 149504
#define SMEM_L1_OUTPUT_SMEM_STAGE_BYTES 1024
#define SMEM_L1_OUTPUT_SMEM_STRIDE 1024
#define SMEM_L1_AMAX_SMEM_OFF 150528
#define SMEM_L1_AMAX_SMEM_STAGE_BYTES 256
#define SMEM_L1_AMAX_SMEM_STRIDE 256
#define SMEM_L2_OUTPUT_SMEM_OFF 149504
#define SMEM_L2_OUTPUT_SMEM_STAGE_BYTES 4096
#define SMEM_L2_OUTPUT_SMEM_STRIDE 4096
#define SMEM_COMBINE_LOAD_SMEM_OFF 1024
#define SMEM_COMBINE_LOAD_SMEM_STAGE_BYTES 7168
#define SMEM_COMBINE_LOAD_SMEM_STRIDE 7168
#define SMEM_COMBINE_STORE_SMEM_OFF 115712
#define SMEM_COMBINE_STORE_SMEM_STAGE_BYTES 7168
#define SMEM_COMBINE_STORE_SMEM_STRIDE 7168
#define SMEM_TOTAL 232448
#define THREADS 384

#include <math_constants.h>
static __device__ __constant__ __align__(16) const unsigned int kE2M1AndUe4m3ToFp8Lut[256] = {
    0x0u, 0x0u, 0x2010000u, 0x6040302u, 0x3020100u, 0xc080604u, 0x4030200u, 0x110c0906u,
    0x6040200u, 0x14100c08u, 0x8050200u, 0x17120f0au, 0x9060300u, 0x1914110cu, 0xa070400u, 0x1a16120eu,
    0xc080400u, 0x1c181410u, 0xe090400u, 0x1e191611u, 0xf0a0500u, 0x1f1a1712u, 0x100b0600u, 0x201b1813u,
    0x110c0600u, 0x211c1914u, 0x120d0600u, 0x221d1a15u, 0x120e0700u, 0x221e1a16u, 0x130f0800u, 0x231f1b17u,
    0x14100800u, 0x24201c18u, 0x16110900u, 0x26211e19u, 0x17120a00u, 0x27221f1au, 0x18130b00u, 0x2823201bu,
    0x19140c00u, 0x2924211cu, 0x1a150d00u, 0x2a25221du, 0x1a160e00u, 0x2a26221eu, 0x1b170f00u, 0x2b27231fu,
    0x1c181000u, 0x2c282420u, 0x1e191100u, 0x2e292621u, 0x1f1a1200u, 0x2f2a2722u, 0x201b1300u, 0x302b2823u,
    0x211c1400u, 0x312c2924u, 0x221d1500u, 0x322d2a25u, 0x221e1600u, 0x322e2a26u, 0x231f1700u, 0x332f2b27u,
    0x24201800u, 0x34302c28u, 0x26211900u, 0x36312e29u, 0x27221a00u, 0x37322f2au, 0x28231b00u, 0x3833302bu,
    0x29241c00u, 0x3934312cu, 0x2a251d00u, 0x3a35322du, 0x2a261e00u, 0x3a36322eu, 0x2b271f00u, 0x3b37332fu,
    0x2c282000u, 0x3c383430u, 0x2e292100u, 0x3e393631u, 0x2f2a2200u, 0x3f3a3732u, 0x302b2300u, 0x403b3833u,
    0x312c2400u, 0x413c3934u, 0x322d2500u, 0x423d3a35u, 0x322e2600u, 0x423e3a36u, 0x332f2700u, 0x433f3b37u,
    0x34302800u, 0x44403c38u, 0x36312900u, 0x46413e39u, 0x37322a00u, 0x47423f3au, 0x38332b00u, 0x4843403bu,
    0x39342c00u, 0x4944413cu, 0x3a352d00u, 0x4a45423du, 0x3a362e00u, 0x4a46423eu, 0x3b372f00u, 0x4b47433fu,
    0x3c383000u, 0x4c484440u, 0x3e393100u, 0x4e494641u, 0x3f3a3200u, 0x4f4a4742u, 0x403b3300u, 0x504b4843u,
    0x413c3400u, 0x514c4944u, 0x423d3500u, 0x524d4a45u, 0x423e3600u, 0x524e4a46u, 0x433f3700u, 0x534f4b47u,
    0x44403800u, 0x54504c48u, 0x46413900u, 0x56514e49u, 0x47423a00u, 0x57524f4au, 0x48433b00u, 0x5853504bu,
    0x49443c00u, 0x5954514cu, 0x4a453d00u, 0x5a55524du, 0x4a463e00u, 0x5a56524eu, 0x4b473f00u, 0x5b57534fu,
    0x4c484000u, 0x5c585450u, 0x4e494100u, 0x5e595651u, 0x4f4a4200u, 0x5f5a5752u, 0x504b4300u, 0x605b5853u,
    0x514c4400u, 0x615c5954u, 0x524d4500u, 0x625d5a55u, 0x524e4600u, 0x625e5a56u, 0x534f4700u, 0x635f5b57u,
    0x54504800u, 0x64605c58u, 0x56514900u, 0x66615e59u, 0x57524a00u, 0x67625f5au, 0x58534b00u, 0x6863605bu,
    0x59544c00u, 0x6964615cu, 0x5a554d00u, 0x6a65625du, 0x5a564e00u, 0x6a66625eu, 0x5b574f00u, 0x6b67635fu,
    0x5c585000u, 0x6c686460u, 0x5e595100u, 0x6e696661u, 0x5f5a5200u, 0x6f6a6762u, 0x605b5300u, 0x706b6863u,
    0x615c5400u, 0x716c6964u, 0x625d5500u, 0x726d6a65u, 0x625e5600u, 0x726e6a66u, 0x635f5700u, 0x736f6b67u,
    0x64605800u, 0x74706c68u, 0x66615900u, 0x76716e69u, 0x67625a00u, 0x77726f6au, 0x68635b00u, 0x7873706bu,
    0x69645c00u, 0x7974716cu, 0x6a655d00u, 0x7a75726du, 0x6a665e00u, 0x7a76726eu, 0x6b675f00u, 0x7b77736fu,
    0x6c686000u, 0x7c787470u, 0x6e696100u, 0x7e797671u, 0x6f6a6200u, 0x7f7a7772u, 0x706b6300u, 0x7f7b7873u,
    0x716c6400u, 0x7f7c7974u, 0x726d6500u, 0x7f7d7a75u, 0x726e6600u, 0x7f7e7a76u, 0x736f6700u, 0x7f7f7b77u,
    0x74706800u, 0x7f7f7c78u, 0x76716900u, 0x7f7f7e79u, 0x77726a00u, 0x7f7f7f7au, 0x78736b00u, 0x7f7f7f7bu,
    0x79746c00u, 0x7f7f7f7cu, 0x7a756d00u, 0x7f7f7f7du, 0x7a766e00u, 0x7f7f7f7eu, 0x7b776f00u, 0x7f7f7f7fu,
    0x7c787000u, 0x7f7f7f7fu, 0x7e797100u, 0x7f7f7f7fu, 0x7f7a7200u, 0x7f7f7f7fu, 0x7f7b7300u, 0x7f7f7f7fu,
    0x7f7c7400u, 0x7f7f7f7fu, 0x7f7d7500u, 0x7f7f7f7fu, 0x7f7e7600u, 0x7f7f7f7fu, 0x7f7e7600u, 0x7f7f7f7fu,
};

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384) void
kernel_small_rs_dispatch_fragment(int num_tokens, LoomTensorMap const* l1_activation, LoomTensorMap const* l1_activation_scale, LoomTensorMap const* l1_packed_weights, LoomTensorMap const* l1_output, LoomTensorMap const* l2_activation, LoomTensorMap const* l2_activation_scale, LoomTensorMap const* l2_packed_weights, float* __restrict__ l1_global_scales, float* __restrict__ l2_global_scales, __nv_bfloat16* __restrict__ output_bf16, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, uint8_t* __restrict__ sym_buffer, uint8_t* const* __restrict__ sym_buffer_peers)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = threadIdx.x % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define pull_full_addr (mbar_base + 0)
    #define task_info_full_addr (mbar_base + 16)
    #define task_info_empty_addr (mbar_base + 32)
    #define l1_stage_full_addr (mbar_base + 48)
    #define l1_stage_empty_addr (mbar_base + 96)
    #define combine_full_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l1_activation)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l1_activation_scale)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l1_packed_weights)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l1_output)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_activation)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_activation_scale)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_packed_weights)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    unsigned int* smem_expert_count = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_expert_count_addr = smem + 1024;
    uint8_t* pull_stage = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int pull_stage_addr = smem + 3072;
    unsigned int* task_info_smem = reinterpret_cast<unsigned int*>(smem_raw + 17408);
    const int task_info_smem_addr = smem + 17408;
    uint8_t* l1_act_smem = reinterpret_cast<uint8_t*>(smem_raw + 18432);
    const int l1_act_smem_addr = smem + 18432;
    float* l1_sfa_smem = reinterpret_cast<float*>(smem_raw + 24576);
    const int l1_sfa_smem_addr = smem + 24576;
    uint8_t* l1_packed_smem = reinterpret_cast<uint8_t*>(smem_raw + 25344);
    const int l1_packed_smem_addr = smem + 25344;
    unsigned int* l1_lut_smem = reinterpret_cast<unsigned int*>(smem_raw + 148224);
    const int l1_lut_smem_addr = smem + 148224;
    uint8_t* l1_decoded_smem = reinterpret_cast<uint8_t*>(smem_raw + 149504);
    const int l1_decoded_smem_addr = smem + 149504;
    uint8_t* l1_output_smem = reinterpret_cast<uint8_t*>(smem_raw + 149504);
    const int l1_output_smem_addr = smem + 149504;
    float* l1_amax_smem = reinterpret_cast<float*>(smem_raw + 150528);
    const int l1_amax_smem_addr = smem + 150528;
    __nv_bfloat16* l2_output_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int l2_output_smem_addr = smem + 149504;
    uint8_t* combine_load_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int combine_load_smem_addr = smem + 1024;
    uint8_t* combine_store_smem = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int combine_store_smem_addr = smem + 115712;

    // Mbarrier init (6 groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pull_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'task_info_pipe' ---
            // task_info_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // task_info_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            // --- pipeline 'l1_pipe' ---
            // l1_stage_full: 6 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // l1_stage_empty: 6 barriers, init_count=8
            mbarrier_init(smem + 96, 8);
            mbarrier_init(smem + 104, 8);
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // --- pipeline 'combine_pipe' ---
            // combine_full: 16 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: dispatch ----
    if (warp <= 1) {
        { // dispatch_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
            {
                const uint4 __cchunk = reinterpret_cast<const uint4*>(kE2M1AndUe4m3ToFp8Lut)[tid];
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(l1_lut_smem_addr + (unsigned int)(tid * 16)), "r"(__cchunk.x), "r"(__cchunk.y), "r"(__cchunk.z), "r"(__cchunk.w));
            }
            if (warp == 0) {
                #pragma unroll 1
                for (int expert = lane; expert < 384; expert += 32) {
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_expert_count_addr + (unsigned int)(expert * 4)), "r"((0)));
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (warp < 2) {
                int route_base = ((unsigned int)(bid * 2) + warp) * 5;
                int route_stride = 780;
                #pragma unroll 1
                for (int token_base = route_base; token_base < num_tokens; token_base += route_stride) {
                    int token_idx = (unsigned int)token_base + lane / 6;
                    int token_topk_idx = (unsigned int)(token_base * 6) + lane;
                    if (lane < 30 && token_idx < num_tokens) {
                        long long _vec_load_0[1];
                        {
                            uint64_t _scalar_bits_0;
                            asm volatile("ld.global.nc.b64 %0, [%1];"
                                : "=l"(_scalar_bits_0) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 171862016) + (token_topk_idx))) : "memory");
                            _vec_load_0[0] = (long long)_scalar_bits_0;
                        }
                        int expert_idx = (int)_vec_load_0[0];
                        if (expert_idx >= 0) {
                            unsigned int _smem_atomic_old_0;
                            asm volatile("atom.cta.shared.add.u32 %0, [%1], %2;" : "=r"(_smem_atomic_old_0) : "r"((unsigned int)((smem_expert_count_addr) + (expert_idx) * 4)), "r"((unsigned int)(1)) : "memory");
                            unsigned int _count_slot = _smem_atomic_old_0;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            #pragma unroll 1
            for (int expert_1 = tid; expert_1 < 384; expert_1 += 64) {
                unsigned int count_word[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&count_word[0])) : "r"(smem_expert_count_addr + (unsigned int)(expert_1 * 4)));
                unsigned long long send_value = 4294967296 + (unsigned long long)count_word[0];
                unsigned long long _atomic_old_0 = atomicAdd(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128)[expert_1], send_value);
                unsigned long long old_status = _atomic_old_0;
                asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_expert_count_addr + (unsigned int)(expert_1 * 4)), "r"(((unsigned int)old_status)));
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (warp < 2) {
                int route_base_1 = ((unsigned int)(bid * 2) + warp) * 5;
                int route_stride_1 = 780;
                #pragma unroll 1
                for (int token_base_1 = route_base_1; token_base_1 < num_tokens; token_base_1 += route_stride_1) {
                    int token_idx_1 = (unsigned int)token_base_1 + lane / 6;
                    int token_topk_idx_1 = (unsigned int)(token_base_1 * 6) + lane;
                    if (lane < 30 && token_idx_1 < num_tokens) {
                        long long _vec_load_1[1];
                        {
                            uint64_t _scalar_bits_1;
                            asm volatile("ld.global.nc.b64 %0, [%1];"
                                : "=l"(_scalar_bits_1) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 171862016) + (token_topk_idx_1))) : "memory");
                            _vec_load_1[0] = (long long)_scalar_bits_1;
                        }
                        int expert_idx_1 = (int)_vec_load_1[0];
                        if (expert_idx_1 >= 0) {
                            int dst_rank = expert_idx_1 / 48;
                            unsigned int _smem_atomic_old_1;
                            asm volatile("atom.cta.shared.add.u32 %0, [%1], %2;" : "=r"(_smem_atomic_old_1) : "r"((unsigned int)((smem_expert_count_addr) + (expert_idx_1) * 4)), "r"((unsigned int)(1)) : "memory");
                            unsigned int dst_slot = _smem_atomic_old_1;
                            unsigned long long dst_index = (unsigned long long)(expert_idx_1 % 48) * 540672 + (unsigned long long)pg_rank * 67584 + (unsigned long long)dst_slot;
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank]) + 628736)) + (dst_index)) = token_topk_idx_1;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_2 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 0);
                        const unsigned int _gs_add_2 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_2;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_2) : "l"(_gs_ptr_2), "r"(_gs_add_2) : "memory");
                        unsigned int _gs_new_2;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_2) : "l"(_gs_ptr_2) : "memory");
                        } while (((_gs_new_2 ^ _gs_old_2) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (bid == 0 && tid < 64) {
                #pragma unroll 1
                for (int expert_2 = tid; expert_2 < 384; expert_2 += 64) {
                    int dst_rank_1 = expert_2 / 48;
                    int dst_local_expert = expert_2 % 48;
                    unsigned long long _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128) + expert_2);
                    }
                    unsigned long long expert_status = _vec_load_2[0];
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_1]) + 3200)) + (pg_rank * 48 + dst_local_expert)) = expert_status & 4294967295;
                    unsigned long long _atomic_old_1;
                    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(_atomic_old_1) : "l"(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_1]) + 6272)[dst_local_expert]), "l"((unsigned long long)(expert_status)) : "memory");
                    unsigned long long _published = _atomic_old_1;
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (bid == 0 && warp == 0) {
                {
                    const unsigned int _rz_status_3 = (*reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)) & 3u;
                    const unsigned int _rz_phase_3 = _rz_status_3 & 1u;
                    const int _rz_sign_3 = (int)(_rz_status_3 >> 1);
                    if ((int)(lane) < (int)(pg_world)) {
                        int* _rz_peer_3 = reinterpret_cast<int*>(reinterpret_cast<char*>((sym_buffer_peers)[lane]) + (unsigned long long)(20)) + _rz_phase_3;
                        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(_rz_peer_3), "r"(_rz_sign_3 ? -1 : 1) : "memory");
                    }
                    __syncwarp();
                    if ((int)(lane) == 0) {
                        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)), "r"(1u) : "memory");
                        const int _rz_target_3 = _rz_sign_3 ? 0 : (int)(pg_world);
                        int* _rz_local_3 = reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4) + (unsigned int)(1)) + _rz_phase_3;
                        while (true) {
                            int _rz_seen_3;
                            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_rz_seen_3) : "l"(_rz_local_3) : "memory");
                            if (_rz_seen_3 == _rz_target_3) break;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_4 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 0);
                        const unsigned int _gs_add_4 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_4;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_4) : "l"(_gs_ptr_4), "r"(_gs_add_4) : "memory");
                        unsigned int _gs_new_4;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_4) : "l"(_gs_ptr_4) : "memory");
                        } while (((_gs_new_4 ^ _gs_old_4) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            unsigned int _phase_pull_full = 0;
            if (warp < 2) {
                unsigned long long ready0 = 0;
                unsigned long long ready1 = 0;
                unsigned int count0 = 0;
                unsigned int count1 = 0;
                if (lane < 48) {
                    while ((unsigned int)(ready0 >> 32) != 624) {
                        unsigned long long _sys_volatile_0;
                        asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_0) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[lane])) : "memory");
                        ready0 = _sys_volatile_0;
                    }
                    count0 = (unsigned int)ready0;
                }
                {
                    if (lane + 32 < 48) {
                        while ((unsigned int)(ready1 >> 32) != 624) {
                            unsigned long long _sys_volatile_1;
                            asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_1) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[lane + 32])) : "memory");
                            ready1 = _sys_volatile_1;
                        }
                        count1 = (unsigned int)ready1;
                    }
                }
                __syncwarp();
                int current_expert = -1;
                unsigned int expert_start = 0;
                unsigned int expert_end = 0;
                unsigned int expert_pool_block_offset = 0;
                unsigned int stored_rank_count = 0;
                unsigned int pull_token_idx = (unsigned int)(bid * 2) + warp;
                while (true) {
                    int old_expert = current_expert;
                    while (pull_token_idx >= expert_end) {
                        current_expert = current_expert + 1;
                        if (current_expert >= 48) {
                            break;
                        }
                        expert_pool_block_offset = expert_pool_block_offset + (expert_end - expert_start + 8 - 1) / 8;
                        expert_start = expert_end;
                        unsigned int lane_count = count0;
                        if (current_expert >= 32) {
                            lane_count = count1;
                        }
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_count, current_expert % 32);
                        unsigned int current_count = _shfl_0;
                        expert_end = expert_end + current_count;
                    }
                    if (current_expert >= 48) {
                        break;
                    }
                    if (old_expert != current_expert && lane < 8) {
                        unsigned long long _vec_load_3[1];
                        {
                            _vec_load_3[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3200) + lane * 48 + (unsigned int)current_expert);
                        }
                        stored_rank_count = (unsigned int)_vec_load_3[0];
                    }
                    unsigned int remaining = stored_rank_count;
                    unsigned int round_offset = 0;
                    unsigned int token_idx_in_expert = pull_token_idx - expert_start;
                    unsigned int slot_idx = token_idx_in_expert;
                    unsigned int selected_rank = 0;
                    unsigned int token_idx_in_rank = 0;
                    while (true) {
                        unsigned int lane_active = 0;
                        unsigned int lane_min = 4294967295;
                        if (remaining > 0) {
                            lane_active = 1;
                            lane_min = remaining;
                        }
                        unsigned int _warp_redux_u32_0;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(lane_active));
                        unsigned int num_active_ranks = _warp_redux_u32_0;
                        unsigned int _warp_redux_u32_1;
                        asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(lane_min));
                        unsigned int round_length = _warp_redux_u32_1;
                        unsigned int num_round_tokens = round_length * num_active_ranks;
                        if (slot_idx < num_round_tokens) {
                            unsigned int slot_in_round = slot_idx % num_active_ranks;
                            unsigned int _vote_0 = __ballot_sync(4294967295, remaining > 0);
                            unsigned int active_mask = _vote_0;
                            int _popc_0 = __popc(active_mask);
                            int num_active_lanes = _popc_0;
                            if (slot_in_round < (unsigned int)num_active_lanes) {
                                unsigned int _fns_0;
                                asm("fns.b32 %0, %1, %2, %3;" : "=r"(_fns_0) : "r"((unsigned int)(active_mask)), "r"((unsigned int)(0)), "r"((int)(slot_in_round + 1)));
                                selected_rank = _fns_0;
                            }
                            token_idx_in_rank = round_offset + slot_idx / num_active_ranks;
                            break;
                        }
                        slot_idx = slot_idx - num_round_tokens;
                        round_offset = round_offset + round_length;
                        if (remaining > round_length) {
                            remaining = remaining - round_length;
                        } else {
                            remaining = 0;
                        }
                    }
                    unsigned long long src_index = (unsigned long long)((unsigned int)(current_expert * 540672) + selected_rank * 67584 + token_idx_in_rank);
                    int _vec_load_4[1];
                    {
                        _vec_load_4[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 628736) + src_index);
                    }
                    unsigned int src_token_topk_idx = (unsigned int)_vec_load_4[0];
                    unsigned int src_token_idx = src_token_topk_idx / 6;
                    unsigned int src_topk_idx = src_token_topk_idx % 6;
                    unsigned int pool_token_idx = expert_pool_block_offset * 8 + token_idx_in_expert;
                    int pull_smem_addr = pull_stage_addr + warp * 7168;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(pull_full_addr + (warp) * 8, 7168);
                        // nvlink_pull: smem(pull_smem_addr) <- peers[selected_rank] + 109414400 + (unsigned long long)src_token_idx * 7168, 7168B
                        {
                            const void* __remote = (const void*)((const char*)((sym_buffer_peers)[selected_rank]) + (uint64_t)(109414400 + (unsigned long long)src_token_idx * 7168));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1], %2, [%3], %4;"
                                :: "r"(pull_smem_addr), "l"(__remote), "r"((uint32_t)(7168)), "r"(pull_full_addr + (warp) * 8), "l"(0x12F0000000000000ULL)
                                : "memory");
                        }
                    }
                    mbarrier_wait(pull_full_addr + (warp) * 8, _phase_pull_full);
                    _phase_pull_full ^= 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    for (int sf_group = 0; sf_group < 2; sf_group++) {
                        int sf_idx = (unsigned int)(sf_group * 32) + lane;
                        if (sf_idx < 56) {
                            float _vec_load_5[1];
                            {
                                _vec_load_5[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[selected_rank]) + 169969664) + src_token_idx * 56 + (unsigned int)sf_idx);
                            }
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3145183232)) + ((unsigned int)(sf_idx * 6635520) + pool_token_idx)) = _vec_load_5[0];
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        float _vec_load_6[1];
                        {
                            _vec_load_6[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[selected_rank]) + 172267520) + src_token_topk_idx);
                        }
                        *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4631539712)) + (pool_token_idx)) = _vec_load_6[0];
                        {
                            void* _cpbulk_dst_5 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(sym_buffer) + 172470272) + (pool_token_idx * 7168));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                :: "l"(_cpbulk_dst_5), "r"(pull_smem_addr), "r"((uint32_t)(7168)), "l"(0x1000000000000000ULL)
                                : "memory");
                        }
                        unsigned long long metadata_idx = (unsigned long long)pool_token_idx * 3;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760)) + (metadata_idx)) = (int)selected_rank;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760)) + (metadata_idx + 1)) = (int)src_token_idx;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760)) + (metadata_idx + 2)) = (int)src_topk_idx;
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(&reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6656)[expert_pool_block_offset + token_idx_in_expert / 8]), "r"((unsigned int)(1)) : "memory");
                    }
                    __syncwarp();
                    pull_token_idx = pull_token_idx + 156;
                }
            }
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int expert_3 = tid; expert_3 < 384; expert_3 += 64) {
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128)) + (expert_3)) = 0;
                }
                if (tid == 0) {
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 7) + (0)) = 0;
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 8) + (0)) = 0;
                }
            } else {
                int cleanup_local_expert = bid - 1;
                if (cleanup_local_expert < 48) {
                    unsigned long long _vec_load_7[1];
                    {
                        _vec_load_7[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272) + cleanup_local_expert);
                    }
                    unsigned int cleanup_num_tokens = (unsigned int)_vec_load_7[0];
                    unsigned int cleanup_num_m_blocks = (cleanup_num_tokens + 8 - 1) / 8;
                    unsigned int cleanup_prefix_lane = 0;
                    #pragma unroll
                    for (int expert_group = 0; expert_group < 2; expert_group++) {
                        int prefix_expert = (unsigned int)(expert_group * 32) + lane;
                        if (prefix_expert < cleanup_local_expert) {
                            unsigned long long _vec_load_8[1];
                            {
                                _vec_load_8[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272) + prefix_expert);
                            }
                            unsigned int prefix_num_tokens = (unsigned int)_vec_load_8[0];
                            cleanup_prefix_lane = cleanup_prefix_lane + (prefix_num_tokens + 8 - 1) / 8;
                        }
                    }
                    unsigned int _warp_redux_u32_2;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(cleanup_prefix_lane));
                    unsigned int cleanup_pool_block_offset = _warp_redux_u32_2;
                    asm volatile("bar.sync 0, 64;" ::: "memory");
                    if (warp == 0) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)) + (cleanup_local_expert)) = 0;
                    } else if (warp == 1) {
                        __syncwarp();
                    }
                    if (tid < 8) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3200)) + (tid * 48 + cleanup_local_expert)) = 0;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int cleanup_block = tid; cleanup_block < cleanup_num_m_blocks; cleanup_block += 64) {
                        unsigned int cleanup_pool_block = cleanup_pool_block_offset + (unsigned int)cleanup_block;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6656)) + (cleanup_pool_block)) = 0;
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 214016)) + (cleanup_pool_block)) = 0;
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_6 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 0);
                        const unsigned int _gs_add_6 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_6;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_6) : "l"(_gs_ptr_6), "r"(_gs_add_6) : "memory");
                        unsigned int _gs_new_6;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_6) : "l"(_gs_ptr_6) : "memory");
                        } while (((_gs_new_6 ^ _gs_old_6) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 0, 64;" ::: "memory");
            if (bid == 0 && warp == 0) {
                {
                    const unsigned int _rz_status_7 = (*reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)) & 3u;
                    const unsigned int _rz_phase_7 = _rz_status_7 & 1u;
                    const int _rz_sign_7 = (int)(_rz_status_7 >> 1);
                    if ((int)(lane) < (int)(pg_world)) {
                        int* _rz_peer_7 = reinterpret_cast<int*>(reinterpret_cast<char*>((sym_buffer_peers)[lane]) + (unsigned long long)(20)) + _rz_phase_7;
                        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(_rz_peer_7), "r"(_rz_sign_7 ? -1 : 1) : "memory");
                    }
                    __syncwarp();
                    if ((int)(lane) == 0) {
                        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)), "r"(1u) : "memory");
                        const int _rz_target_7 = _rz_sign_7 ? 0 : (int)(pg_world);
                        int* _rz_local_7 = reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4) + (unsigned int)(1)) + _rz_phase_7;
                        while (true) {
                            int _rz_seen_7;
                            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_rz_seen_7) : "l"(_rz_local_7) : "memory");
                            if (_rz_seen_7 == _rz_target_7) break;
                        }
                    }
                    __syncwarp();
                }
            }
        }
    }
    // ---- Role: loader ----
    if (warp >= 2 && warp <= 3) {
        { // loader_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
            unsigned int l1_load_stage = 0;
            unsigned int _phase_task_info_empty = 1;
            unsigned int _phase_l1_stage_empty = 1;
            unsigned int _phase_task_info_full = 0;
            if (warp == 3) {
                unsigned int stored_num_tokens[2];
                #pragma unroll
                for (int expert_group_1 = 0; expert_group_1 < 2; expert_group_1++) {
                    unsigned long long ready_status = 0;
                    int local_expert = (unsigned int)(expert_group_1 * 32) + lane;
                    if (local_expert < 48) {
                        while ((unsigned int)(ready_status >> 32) != 624) {
                            unsigned long long _sys_volatile_2;
                            asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_2) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert])) : "memory");
                            ready_status = _sys_volatile_2;
                        }
                    }
                    stored_num_tokens[expert_group_1] = (unsigned int)ready_status;
                }
                __syncwarp();
                unsigned int lane_num_m_blocks = 0;
                #pragma unroll
                for (int expert_group_2 = 0; expert_group_2 < 2; expert_group_2++) {
                    int local_expert_1 = (unsigned int)(expert_group_2 * 32) + lane;
                    if (local_expert_1 < 48) {
                        lane_num_m_blocks = lane_num_m_blocks + (stored_num_tokens[expert_group_2] + 8 - 1) / 8;
                    }
                }
                unsigned int _warp_redux_u32_3;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(lane_num_m_blocks));
                unsigned int num_total_m_blocks = _warp_redux_u32_3;
                unsigned int num_total_l1_tasks = num_total_m_blocks * 24;
                unsigned int num_total_l1_waves = (num_total_l1_tasks + 78 - 1) / 78;
                unsigned int num_l1_warmup_waves = 0;
                if (num_total_m_blocks > 0) {
                    int _min_0 = ((2) < (num_total_l1_waves) ? (2) : (num_total_l1_waves));
                    num_l1_warmup_waves = _min_0;
                }
                unsigned int l1_waves_done = 4294967295;
                unsigned int producer_task_stage = 0;
                unsigned int task_phase = 0;
                unsigned int task_idx = 0;
                unsigned int task_num_n_blocks = 0;
                unsigned int task_shape_n = 0;
                unsigned int task_shape_k = 0;
                unsigned int task_pool_block = 0;
                unsigned int task_local_expert = 0;
                unsigned int task_m_block = 0;
                unsigned int task_n_block = 0;
                unsigned int task_valid_m = 0;
                while (true) {
                    mbarrier_wait(task_info_empty_addr + (producer_task_stage) * 8, _phase_task_info_empty);
                    while (true) {
                        task_phase = 0;
                        task_idx = 0;
                        task_num_n_blocks = 0;
                        task_shape_n = 0;
                        task_shape_k = 0;
                        task_pool_block = 0;
                        task_local_expert = 0;
                        task_m_block = 0;
                        task_n_block = 0;
                        task_valid_m = 0;
                        if (num_l1_warmup_waves != l1_waves_done && num_l1_warmup_waves > 0) {
                            num_l1_warmup_waves = num_l1_warmup_waves - 1;
                            if (lane == 0) {
                                unsigned int _atomic_old_2 = atomicAdd(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 7, 1);
                                task_idx = _atomic_old_2;
                            }
                            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, task_idx, 0);
                            task_idx = _shfl_1;
                            if (task_idx >= num_total_l1_tasks) {
                                num_l1_warmup_waves = l1_waves_done;
                                continue;
                            }
                            task_phase = 1;
                            task_num_n_blocks = 24;
                            task_shape_n = 6144;
                            task_shape_k = 7168;
                        } else {
                            if (lane == 0) {
                                unsigned int _atomic_old_3 = atomicAdd(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 8, 1);
                                task_idx = _atomic_old_3;
                            }
                            unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, task_idx, 0);
                            task_idx = _shfl_2;
                            if (task_idx >= num_total_m_blocks * 28) {
                                break;
                            }
                            if (num_l1_warmup_waves != l1_waves_done) {
                                num_l1_warmup_waves = 1;
                            }
                            task_phase = 2;
                            task_num_n_blocks = 28;
                            task_shape_n = 7168;
                            task_shape_k = 3072;
                            unsigned int required_l1_tasks = (task_idx / 28 + 1) * 24;
                            unsigned int seen_l1_tasks = 0;
                            while (seen_l1_tasks < required_l1_tasks) {
                                unsigned int _sys_volatile_3;
                                asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(_sys_volatile_3) : "l"(reinterpret_cast<const unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 7)) : "memory");
                                seen_l1_tasks = _sys_volatile_3;
                            }
                        }
                        task_pool_block = task_idx / task_num_n_blocks;
                        task_n_block = task_idx % task_num_n_blocks;
                        unsigned int block_offset = 0;
                        #pragma unroll
                        for (int expert_group_3 = 0; expert_group_3 < 2; expert_group_3++) {
                            unsigned int expert_idx_2 = (unsigned int)(expert_group_3 * 32) + lane;
                            unsigned int expert_tokens = stored_num_tokens[expert_group_3];
                            unsigned int expert_m_blocks = (expert_tokens + 8 - 1) / 8;
                            unsigned int inclusive_m_blocks = expert_m_blocks;
                            unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, inclusive_m_blocks, 1, 32);
                            unsigned int scan_peer = _shfl_up_0;
                            if (lane >= 1) {
                                inclusive_m_blocks = inclusive_m_blocks + scan_peer;
                            }
                            unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, inclusive_m_blocks, 2, 32);
                            unsigned int scan_peer_0 = _shfl_up_1;
                            if (lane >= 2) {
                                inclusive_m_blocks = inclusive_m_blocks + scan_peer_0;
                            }
                            unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, inclusive_m_blocks, 4, 32);
                            unsigned int scan_peer_1 = _shfl_up_2;
                            if (lane >= 4) {
                                inclusive_m_blocks = inclusive_m_blocks + scan_peer_1;
                            }
                            unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, inclusive_m_blocks, 8, 32);
                            unsigned int scan_peer_2 = _shfl_up_3;
                            if (lane >= 8) {
                                inclusive_m_blocks = inclusive_m_blocks + scan_peer_2;
                            }
                            unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, inclusive_m_blocks, 16, 32);
                            unsigned int scan_peer_3 = _shfl_up_4;
                            if (lane >= 16) {
                                inclusive_m_blocks = inclusive_m_blocks + scan_peer_3;
                            }
                            unsigned int lane_pool_offset = block_offset + inclusive_m_blocks - expert_m_blocks;
                            unsigned int _vote_1 = __ballot_sync(4294967295, expert_idx_2 < 48 && (task_pool_block >= lane_pool_offset && task_pool_block < lane_pool_offset + expert_m_blocks));
                            unsigned int owner_mask = _vote_1;
                            if (owner_mask != 0) {
                                int _ffs_0 = __ffs(owner_mask);
                                int owner_lane = _ffs_0 - 1;
                                unsigned int owner_m_block = task_pool_block - lane_pool_offset;
                                unsigned int _min_1 = ((expert_tokens - owner_m_block * 8) < (8) ? (expert_tokens - owner_m_block * 8) : (8));
                                unsigned int owner_valid_m = _min_1;
                                unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, expert_idx_2, owner_lane);
                                task_local_expert = _shfl_3;
                                unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, owner_m_block, owner_lane);
                                task_m_block = _shfl_4;
                                unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, owner_valid_m, owner_lane);
                                task_valid_m = _shfl_5;
                            }
                            unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, inclusive_m_blocks, 31);
                            block_offset = block_offset + _shfl_6;
                        }
                        break;
                    }
                    if (elect_sync()) {
                        unsigned int task_info_words[8];
                        task_info_words[0] = task_phase;
                        task_info_words[1] = task_local_expert;
                        task_info_words[2] = task_m_block;
                        task_info_words[3] = task_n_block;
                        task_info_words[4] = task_pool_block;
                        task_info_words[5] = task_valid_m;
                        task_info_words[6] = task_shape_n;
                        task_info_words[7] = task_shape_k;
                        #pragma unroll
                        for (int task_word = 0; task_word < 8; task_word++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(task_info_smem_addr + producer_task_stage * 32 + (unsigned int)(task_word * 4)), "r"((task_info_words[task_word])));
                        }
                        __threadfence_block();
                        mbarrier_arrive(task_info_full_addr + (producer_task_stage) * 8);
                    }
                    __syncwarp();
                    producer_task_stage += 1;
                    if (producer_task_stage == 2) { producer_task_stage = 0; _phase_task_info_empty ^= 1; }
                    if (task_phase != 0) {
                        #pragma unroll 1
                        for (int k_block_idx = 0; k_block_idx < task_shape_k / 128; k_block_idx++) {
                            mbarrier_wait(l1_stage_empty_addr + (l1_load_stage) * 8, _phase_l1_stage_empty);
                            if (elect_sync()) {
                                if (task_phase == 1) {
                                    tma_2d_gmem2smem(l1_packed_smem_addr + l1_load_stage * 20480, l1_packed_weights, k_block_idx * 80, task_local_expert * task_shape_n + task_n_block * 256, l1_stage_full_addr + (l1_load_stage) * 8);
                                } else {
                                    tma_2d_gmem2smem(l1_packed_smem_addr + l1_load_stage * 20480, l2_packed_weights, k_block_idx * 80, task_local_expert * task_shape_n + task_n_block * 256, l1_stage_full_addr + (l1_load_stage) * 8);
                                }
                                mbarrier_arrive_expect_tx(l1_stage_full_addr + (l1_load_stage) * 8, 20480);
                            }
                            __syncwarp();
                            l1_load_stage += 1;
                            if (l1_load_stage == 6) { l1_load_stage = 0; _phase_l1_stage_empty ^= 1; }
                        }
                    }
                    if (task_phase == 0) {
                        break;
                    }
                }
            } else {
                unsigned int loader_task_stage = 0;
                unsigned int loader_task_phase = 0;
                while (true) {
                    mbarrier_wait(task_info_full_addr + (loader_task_stage) * 8, _phase_task_info_full);
                    unsigned int loader_task_words[8];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(0) + 3]))
                        : "r"(task_info_smem_addr + loader_task_stage * 32));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&loader_task_words[(4) + 3]))
                        : "r"(task_info_smem_addr + loader_task_stage * 32 + 16));
                    loader_task_phase = loader_task_words[0];
                    loader_task_stage += 1;
                    if (loader_task_stage == 2) { loader_task_stage = 0; _phase_task_info_empty ^= 1; _phase_task_info_full ^= 1; }
                    if (loader_task_phase == 0) {
                        break;
                    }
                    unsigned int loader_task_pool_block = loader_task_words[4];
                    unsigned int loader_task_valid_m = loader_task_words[5];
                    unsigned int loader_task_shape_k = loader_task_words[7];
                    if (loader_task_valid_m > 0) {
                        if (loader_task_phase == 1) {
                            {
                                unsigned int* _gwa_p_0 = reinterpret_cast<unsigned int*>(&reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6656)[loader_task_pool_block]);
                                while (true) {
                                    unsigned int _gwa_v_0;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gwa_v_0) : "l"(_gwa_p_0) : "memory");
                                    if (_gwa_v_0 == (unsigned int)(loader_task_valid_m)) break;
                                }
                            }
                        } else {
                            {
                                unsigned long long* _gwa_p_1 = reinterpret_cast<unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 214016)[loader_task_pool_block]);
                                while (true) {
                                    unsigned long long _gwa_v_1;
                                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_gwa_v_1) : "l"(_gwa_p_1) : "memory");
                                    if (_gwa_v_1 == (unsigned long long)(16777215)) break;
                                }
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int k_block_idx_1 = 0; k_block_idx_1 < loader_task_shape_k / 128; k_block_idx_1++) {
                        mbarrier_wait(l1_stage_empty_addr + (l1_load_stage) * 8, _phase_l1_stage_empty);
                        if (loader_task_valid_m > 0) {
                            if (elect_sync()) {
                                if (loader_task_phase == 1) {
                                    tma_2d_gmem2smem(l1_act_smem_addr + l1_load_stage * 1024, l1_activation, k_block_idx_1 * 128, loader_task_pool_block * 8, l1_stage_full_addr + (l1_load_stage) * 8);
                                    tma_2d_gmem2smem(l1_sfa_smem_addr + l1_load_stage * 128, l1_activation_scale, loader_task_pool_block * 8, k_block_idx_1, l1_stage_full_addr + (l1_load_stage) * 8);
                                } else {
                                    tma_2d_gmem2smem(l1_act_smem_addr + l1_load_stage * 1024, l2_activation, k_block_idx_1 * 128, loader_task_pool_block * 8, l1_stage_full_addr + (l1_load_stage) * 8);
                                    tma_2d_gmem2smem(l1_sfa_smem_addr + l1_load_stage * 128, l2_activation_scale, loader_task_pool_block * 8, k_block_idx_1, l1_stage_full_addr + (l1_load_stage) * 8);
                                }
                                mbarrier_arrive_expect_tx(l1_stage_full_addr + (l1_load_stage) * 8, 1056);
                            }
                        } else {
                            if (elect_sync()) {
                                mbarrier_arrive(l1_stage_full_addr + (l1_load_stage) * 8);
                            }
                        }
                        __syncwarp();
                        l1_load_stage += 1;
                        if (l1_load_stage == 6) { l1_load_stage = 0; _phase_l1_stage_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: math ----
    if (warp >= 4 && warp <= 11) {
        { // math_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            unsigned int math_task_stage = 0;
            unsigned int l1_math_stage = 0;
            unsigned int math_task_phase = 0;
            unsigned int _phase_task_info_full_1 = 0;
            unsigned int _phase_l1_stage_full = 0;
            while (true) {
                mbarrier_wait(task_info_full_addr + (math_task_stage) * 8, _phase_task_info_full_1);
                unsigned int consumed_task_stage = math_task_stage;
                unsigned int math_task_words[8];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(0) + 3]))
                    : "r"(task_info_smem_addr + math_task_stage * 32));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&math_task_words[(4) + 3]))
                    : "r"(task_info_smem_addr + math_task_stage * 32 + 16));
                math_task_phase = math_task_words[0];
                math_task_stage += 1;
                if (math_task_stage == 2) { math_task_stage = 0; _phase_task_info_full_1 ^= 1; }
                if (math_task_phase == 0) {
                    break;
                }
                if (math_task_phase != 0) {
                    int math_task_valid_m = (int)math_task_words[5];
                    int math_task_shape_k = (int)math_task_words[7];
                    float final_accum[64];
                    final_accum[0] = 0.0f;
                    final_accum[1] = 0.0f;
                    final_accum[2] = 0.0f;
                    final_accum[3] = 0.0f;
                    final_accum[4] = 0.0f;
                    final_accum[5] = 0.0f;
                    final_accum[6] = 0.0f;
                    final_accum[7] = 0.0f;
                    final_accum[8] = 0.0f;
                    final_accum[9] = 0.0f;
                    final_accum[10] = 0.0f;
                    final_accum[11] = 0.0f;
                    final_accum[12] = 0.0f;
                    final_accum[13] = 0.0f;
                    final_accum[14] = 0.0f;
                    final_accum[15] = 0.0f;
                    final_accum[16] = 0.0f;
                    final_accum[17] = 0.0f;
                    final_accum[18] = 0.0f;
                    final_accum[19] = 0.0f;
                    final_accum[20] = 0.0f;
                    final_accum[21] = 0.0f;
                    final_accum[22] = 0.0f;
                    final_accum[23] = 0.0f;
                    final_accum[24] = 0.0f;
                    final_accum[25] = 0.0f;
                    final_accum[26] = 0.0f;
                    final_accum[27] = 0.0f;
                    final_accum[28] = 0.0f;
                    final_accum[29] = 0.0f;
                    final_accum[30] = 0.0f;
                    final_accum[31] = 0.0f;
                    final_accum[32] = 0.0f;
                    final_accum[33] = 0.0f;
                    final_accum[34] = 0.0f;
                    final_accum[35] = 0.0f;
                    final_accum[36] = 0.0f;
                    final_accum[37] = 0.0f;
                    final_accum[38] = 0.0f;
                    final_accum[39] = 0.0f;
                    final_accum[40] = 0.0f;
                    final_accum[41] = 0.0f;
                    final_accum[42] = 0.0f;
                    final_accum[43] = 0.0f;
                    final_accum[44] = 0.0f;
                    final_accum[45] = 0.0f;
                    final_accum[46] = 0.0f;
                    final_accum[47] = 0.0f;
                    final_accum[48] = 0.0f;
                    final_accum[49] = 0.0f;
                    final_accum[50] = 0.0f;
                    final_accum[51] = 0.0f;
                    final_accum[52] = 0.0f;
                    final_accum[53] = 0.0f;
                    final_accum[54] = 0.0f;
                    final_accum[55] = 0.0f;
                    final_accum[56] = 0.0f;
                    final_accum[57] = 0.0f;
                    final_accum[58] = 0.0f;
                    final_accum[59] = 0.0f;
                    final_accum[60] = 0.0f;
                    final_accum[61] = 0.0f;
                    final_accum[62] = 0.0f;
                    final_accum[63] = 0.0f;
                    {
                        int math_num_k_blocks = math_task_shape_k / 128;
                        unsigned int a_frags_0[16];
                        unsigned int a_frags_1[16];
                        float swap_accum_0[4];
                        float swap_accum_1[4];
                        mbarrier_wait(l1_stage_full_addr + (l1_math_stage) * 8, _phase_l1_stage_full);
                        if (elect_sync()) {
                            mbarrier_arrive(task_info_empty_addr + (consumed_task_stage) * 8);
                        }
                        int math_wg = (warp - 4) / 4;
                        int warp_in_wg = (warp - 4) % 4;
                        int row_in_warp = lane / 4;
                        int word_sel = lane >> 1 & 1;
                        int decode_row = (unsigned int)(math_wg * 128 + warp_in_wg * 16 + row_in_warp) + ((lane & 1) << 3);
                        int packed_row_addr = l1_packed_smem_addr + l1_math_stage * 20480 + (unsigned int)(decode_row * 80);
                        unsigned int scale_words[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scale_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words[(0) + 1]))
                            : "r"(packed_row_addr + 64));
                        #pragma unroll
                        for (int k_slice = 0; k_slice < 4; k_slice++) {
                            unsigned int packed_lo[1];
                            unsigned int packed_hi[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_lo[0])) : "r"(packed_row_addr + k_slice * 16 + word_sel * 4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_hi[0])) : "r"(packed_row_addr + k_slice * 16 + 8 + word_sel * 4));
                            unsigned int scale_word = scale_words[0];
                            if (k_slice >= 2) {
                                scale_word = scale_words[1];
                            }
                            int scale_shift = (k_slice & 1) * 16;
                            unsigned int scale_lo = scale_word >> (unsigned int)scale_shift & 127;
                            unsigned int scale_hi = scale_word >> (unsigned int)(scale_shift + 8) & 127;
                            unsigned int lut_lo[2];
                            unsigned int lut_hi[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_lo[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_lo * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_hi[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_hi * 8)));
                            unsigned int decoded_lo_hi = 0;
                            unsigned int decoded_lo_lo = 0;
                            unsigned int decoded_hi_hi = 0;
                            unsigned int decoded_hi_lo = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_lo[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_lo[0])), "r"((uint32_t)(lut_lo[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_lo[0])), "r"((uint32_t)(lut_lo[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                decoded_lo_hi = _nv_hi;
                                decoded_lo_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_hi[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_hi[0])), "r"((uint32_t)(lut_hi[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_hi[0])), "r"((uint32_t)(lut_hi[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                decoded_hi_hi = _nv_hi;
                                decoded_hi_lo = _nv_lo;
                            }
                            unsigned int keep_lo = decoded_lo_hi;
                            unsigned int ship_lo = decoded_lo_lo;
                            unsigned int keep_hi = decoded_hi_hi;
                            unsigned int ship_hi = decoded_hi_lo;
                            if ((lane & 1) != 0) {
                                keep_lo = decoded_lo_lo;
                                ship_lo = decoded_lo_hi;
                                keep_hi = decoded_hi_lo;
                                ship_hi = decoded_hi_hi;
                            }
                            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, ship_lo, 1);
                            unsigned int recv_lo = _shfl_xor_0;
                            unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, ship_hi, 1);
                            unsigned int recv_hi = _shfl_xor_1;
                            int frag_base = k_slice * 4;
                            if ((lane & 1) == 0) {
                                a_frags_0[frag_base] = keep_lo;
                                a_frags_0[frag_base + 1] = recv_lo;
                                a_frags_0[frag_base + 2] = keep_hi;
                                a_frags_0[frag_base + 3] = recv_hi;
                            } else {
                                a_frags_0[frag_base] = recv_lo;
                                a_frags_0[frag_base + 1] = keep_lo;
                                a_frags_0[frag_base + 2] = recv_hi;
                                a_frags_0[frag_base + 3] = keep_hi;
                            }
                            #pragma unroll
                            for (int frag_word = 0; frag_word < 4; frag_word++) {
                                asm volatile("" : "+r"(a_frags_0[frag_base + frag_word]) :: "memory");
                            }
                        }
                        #pragma unroll 1
                        for (int k_block = 0; k_block < math_num_k_blocks; k_block++) {
                            int cur_packed_addr = l1_packed_smem_addr + l1_math_stage * 20480;
                            int cur_sfa_addr = l1_sfa_smem_addr + l1_math_stage * 128;
                            unsigned int cur_stage = l1_math_stage;
                            swap_accum_0[0] = 0.0f;
                            swap_accum_0[1] = 0.0f;
                            swap_accum_0[2] = 0.0f;
                            swap_accum_0[3] = 0.0f;
                            #pragma unroll
                            for (int accum_idx = 0; accum_idx < 4; accum_idx++) {
                                asm volatile("" : "+f"(swap_accum_0[accum_idx]) :: "memory");
                            }
                            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                            uint64_t _wgmma_desc_0 = (((uint64_t)(((l1_act_smem_addr + l1_math_stage * 1024)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                            uint64_t _wgmma_b_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1;\n}\n"
                                : "+f"(swap_accum_0[0]), "+f"(swap_accum_0[1]), "+f"(swap_accum_0[2]), "+f"(swap_accum_0[3])
                                : "r"(a_frags_0[0]), "r"(a_frags_0[1]), "r"(a_frags_0[2]), "r"(a_frags_0[3]), "l"(_wgmma_b_0_0)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_0[0]), "+f"(swap_accum_0[1]), "+f"(swap_accum_0[2]), "+f"(swap_accum_0[3])
                                : "r"(a_frags_0[4]), "r"(a_frags_0[(4) + 1]), "r"(a_frags_0[(4) + 2]), "r"(a_frags_0[(4) + 3]), "l"(_wgmma_b_0_0 + 2)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_0[0]), "+f"(swap_accum_0[1]), "+f"(swap_accum_0[2]), "+f"(swap_accum_0[3])
                                : "r"(a_frags_0[8]), "r"(a_frags_0[(8) + 1]), "r"(a_frags_0[(8) + 2]), "r"(a_frags_0[(8) + 3]), "l"(_wgmma_b_0_0 + 4)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_0[0]), "+f"(swap_accum_0[1]), "+f"(swap_accum_0[2]), "+f"(swap_accum_0[3])
                                : "r"(a_frags_0[12]), "r"(a_frags_0[(12) + 1]), "r"(a_frags_0[(12) + 2]), "r"(a_frags_0[(12) + 3]), "l"(_wgmma_b_0_0 + 6)
                                : "memory");
                            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                            int math_wg_0 = (warp - 4) / 4;
                            int warp_in_wg_1 = (warp - 4) % 4;
                            int row_in_warp_2 = lane / 4;
                            int word_sel_3 = lane >> 1 & 1;
                            int decode_row_4 = (unsigned int)(math_wg_0 * 128 + 64 + warp_in_wg_1 * 16 + row_in_warp_2) + ((lane & 1) << 3);
                            int packed_row_addr_5 = cur_packed_addr + decode_row_4 * 80;
                            unsigned int scale_words_6[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&scale_words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words_6[(0) + 1]))
                                : "r"(packed_row_addr_5 + 64));
                            #pragma unroll
                            for (int k_slice_1 = 0; k_slice_1 < 4; k_slice_1++) {
                                unsigned int packed_lo_1[1];
                                unsigned int packed_hi_1[1];
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_lo_1[0])) : "r"(packed_row_addr_5 + k_slice_1 * 16 + word_sel_3 * 4));
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_hi_1[0])) : "r"(packed_row_addr_5 + k_slice_1 * 16 + 8 + word_sel_3 * 4));
                                unsigned int scale_word_1 = scale_words_6[0];
                                if (k_slice_1 >= 2) {
                                    scale_word_1 = scale_words_6[1];
                                }
                                int scale_shift_1 = (k_slice_1 & 1) * 16;
                                unsigned int scale_lo_1 = scale_word_1 >> (unsigned int)scale_shift_1 & 127;
                                unsigned int scale_hi_1 = scale_word_1 >> (unsigned int)(scale_shift_1 + 8) & 127;
                                unsigned int lut_lo_1[2];
                                unsigned int lut_hi_1[2];
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&lut_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_lo_1[(0) + 1]))
                                    : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_lo_1 * 8)));
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&lut_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_hi_1[(0) + 1]))
                                    : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_hi_1 * 8)));
                                unsigned int decoded_lo_hi_1 = 0;
                                unsigned int decoded_lo_lo_1 = 0;
                                unsigned int decoded_hi_hi_1 = 0;
                                unsigned int decoded_hi_lo_1 = 0;
                                {
                                    const uint32_t _nv_packed = (uint32_t)(packed_lo_1[0]);
                                    const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                    uint32_t _nv_hi, _nv_lo;
                                    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_lo_1[0])), "r"((uint32_t)(lut_lo_1[1])), "r"(_nv_selectors));
                                    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_lo_1[0])), "r"((uint32_t)(lut_lo_1[1])), "r"(_nv_selectors >> 16));
                                    asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                    const uint32_t _nv_shifted = _nv_packed << 4;
                                    asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                    decoded_lo_hi_1 = _nv_hi;
                                    decoded_lo_lo_1 = _nv_lo;
                                }
                                {
                                    const uint32_t _nv_packed = (uint32_t)(packed_hi_1[0]);
                                    const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                    uint32_t _nv_hi, _nv_lo;
                                    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_hi_1[0])), "r"((uint32_t)(lut_hi_1[1])), "r"(_nv_selectors));
                                    asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_hi_1[0])), "r"((uint32_t)(lut_hi_1[1])), "r"(_nv_selectors >> 16));
                                    asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                    const uint32_t _nv_shifted = _nv_packed << 4;
                                    asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                    decoded_hi_hi_1 = _nv_hi;
                                    decoded_hi_lo_1 = _nv_lo;
                                }
                                unsigned int keep_lo_1 = decoded_lo_hi_1;
                                unsigned int ship_lo_1 = decoded_lo_lo_1;
                                unsigned int keep_hi_1 = decoded_hi_hi_1;
                                unsigned int ship_hi_1 = decoded_hi_lo_1;
                                if ((lane & 1) != 0) {
                                    keep_lo_1 = decoded_lo_lo_1;
                                    ship_lo_1 = decoded_lo_hi_1;
                                    keep_hi_1 = decoded_hi_lo_1;
                                    ship_hi_1 = decoded_hi_hi_1;
                                }
                                unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, ship_lo_1, 1);
                                unsigned int recv_lo_1 = _shfl_xor_2;
                                unsigned int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, ship_hi_1, 1);
                                unsigned int recv_hi_1 = _shfl_xor_3;
                                int frag_base_1 = k_slice_1 * 4;
                                if ((lane & 1) == 0) {
                                    a_frags_1[frag_base_1] = keep_lo_1;
                                    a_frags_1[frag_base_1 + 1] = recv_lo_1;
                                    a_frags_1[frag_base_1 + 2] = keep_hi_1;
                                    a_frags_1[frag_base_1 + 3] = recv_hi_1;
                                } else {
                                    a_frags_1[frag_base_1] = recv_lo_1;
                                    a_frags_1[frag_base_1 + 1] = keep_lo_1;
                                    a_frags_1[frag_base_1 + 2] = recv_hi_1;
                                    a_frags_1[frag_base_1 + 3] = keep_hi_1;
                                }
                                #pragma unroll
                                for (int frag_word_1 = 0; frag_word_1 < 4; frag_word_1++) {
                                    asm volatile("" : "+r"(a_frags_1[frag_base_1 + frag_word_1]) :: "memory");
                                }
                            }
                            swap_accum_1[0] = 0.0f;
                            swap_accum_1[1] = 0.0f;
                            swap_accum_1[2] = 0.0f;
                            swap_accum_1[3] = 0.0f;
                            #pragma unroll
                            for (int accum_idx_1 = 0; accum_idx_1 < 4; accum_idx_1++) {
                                asm volatile("" : "+f"(swap_accum_1[accum_idx_1]) :: "memory");
                            }
                            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1;\n}\n"
                                : "+f"(swap_accum_1[0]), "+f"(swap_accum_1[1]), "+f"(swap_accum_1[2]), "+f"(swap_accum_1[3])
                                : "r"(a_frags_1[0]), "r"(a_frags_1[1]), "r"(a_frags_1[2]), "r"(a_frags_1[3]), "l"(_wgmma_b_0_0)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_1[0]), "+f"(swap_accum_1[1]), "+f"(swap_accum_1[2]), "+f"(swap_accum_1[3])
                                : "r"(a_frags_1[4]), "r"(a_frags_1[(4) + 1]), "r"(a_frags_1[(4) + 2]), "r"(a_frags_1[(4) + 3]), "l"(_wgmma_b_0_0 + 2)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_1[0]), "+f"(swap_accum_1[1]), "+f"(swap_accum_1[2]), "+f"(swap_accum_1[3])
                                : "r"(a_frags_1[8]), "r"(a_frags_1[(8) + 1]), "r"(a_frags_1[(8) + 2]), "r"(a_frags_1[(8) + 3]), "l"(_wgmma_b_0_0 + 4)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1;\n}\n"
                                : "+f"(swap_accum_1[0]), "+f"(swap_accum_1[1]), "+f"(swap_accum_1[2]), "+f"(swap_accum_1[3])
                                : "r"(a_frags_1[12]), "r"(a_frags_1[(12) + 1]), "r"(a_frags_1[(12) + 2]), "r"(a_frags_1[(12) + 3]), "l"(_wgmma_b_0_0 + 6)
                                : "memory");
                            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                            #pragma unroll
                            for (int accum_idx_2 = 0; accum_idx_2 < 4; accum_idx_2++) {
                                asm volatile("" : "+f"(swap_accum_0[accum_idx_2]) :: "memory");
                            }
                            #pragma unroll
                            for (int frag_idx = 0; frag_idx < 16; frag_idx++) {
                                asm volatile("" : "+r"(a_frags_0[frag_idx]) :: "memory");
                            }
                            int col_idx = lane % 4;
                            #pragma unroll
                            for (int token_chunk = 0; token_chunk < 1; token_chunk++) {
                                int token_0 = token_chunk * 8 + col_idx * 2;
                                int token_1 = token_0 + 1;
                                int accum_offset = token_chunk * 4;
                                if (token_0 < math_task_valid_m) {
                                    float scale_0[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_0[0])) : "r"(cur_sfa_addr + token_0 * 4));
                                    final_accum[accum_offset] = final_accum[accum_offset] + scale_0[0] * swap_accum_0[token_chunk * 4];
                                    final_accum[accum_offset + 2] = final_accum[accum_offset + 2] + scale_0[0] * swap_accum_0[token_chunk * 4 + 2];
                                }
                                if (token_1 < math_task_valid_m) {
                                    float scale_1[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_1[0])) : "r"(cur_sfa_addr + token_1 * 4));
                                    final_accum[accum_offset + 1] = final_accum[accum_offset + 1] + scale_1[0] * swap_accum_0[token_chunk * 4 + 1];
                                    final_accum[accum_offset + 3] = final_accum[accum_offset + 3] + scale_1[0] * swap_accum_0[token_chunk * 4 + 3];
                                }
                            }
                            l1_math_stage += 1;
                            if (l1_math_stage == 6) { l1_math_stage = 0; _phase_l1_stage_full ^= 1; }
                            if (math_num_k_blocks > k_block + 1) {
                                mbarrier_wait(l1_stage_full_addr + (l1_math_stage) * 8, _phase_l1_stage_full);
                                int math_wg_1 = (warp - 4) / 4;
                                int warp_in_wg_2 = (warp - 4) % 4;
                                int row_in_warp_3 = lane / 4;
                                int word_sel_4 = lane >> 1 & 1;
                                int decode_row_5 = (unsigned int)(math_wg_1 * 128 + warp_in_wg_2 * 16 + row_in_warp_3) + ((lane & 1) << 3);
                                int packed_row_addr_6 = l1_packed_smem_addr + l1_math_stage * 20480 + (unsigned int)(decode_row_5 * 80);
                                unsigned int scale_words_7[2];
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&scale_words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words_7[(0) + 1]))
                                    : "r"(packed_row_addr_6 + 64));
                                #pragma unroll
                                for (int k_slice_2 = 0; k_slice_2 < 4; k_slice_2++) {
                                    unsigned int packed_lo_2[1];
                                    unsigned int packed_hi_2[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_lo_2[0])) : "r"(packed_row_addr_6 + k_slice_2 * 16 + word_sel_4 * 4));
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&packed_hi_2[0])) : "r"(packed_row_addr_6 + k_slice_2 * 16 + 8 + word_sel_4 * 4));
                                    unsigned int scale_word_2 = scale_words_7[0];
                                    if (k_slice_2 >= 2) {
                                        scale_word_2 = scale_words_7[1];
                                    }
                                    int scale_shift_2 = (k_slice_2 & 1) * 16;
                                    unsigned int scale_lo_2 = scale_word_2 >> (unsigned int)scale_shift_2 & 127;
                                    unsigned int scale_hi_2 = scale_word_2 >> (unsigned int)(scale_shift_2 + 8) & 127;
                                    unsigned int lut_lo_2[2];
                                    unsigned int lut_hi_2[2];
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&lut_lo_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_lo_2[(0) + 1]))
                                        : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_lo_2 * 8)));
                                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&lut_hi_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut_hi_2[(0) + 1]))
                                        : "r"(l1_lut_smem_addr + (unsigned int)((int)scale_hi_2 * 8)));
                                    unsigned int decoded_lo_hi_2 = 0;
                                    unsigned int decoded_lo_lo_2 = 0;
                                    unsigned int decoded_hi_hi_2 = 0;
                                    unsigned int decoded_hi_lo_2 = 0;
                                    {
                                        const uint32_t _nv_packed = (uint32_t)(packed_lo_2[0]);
                                        const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                        uint32_t _nv_hi, _nv_lo;
                                        asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_lo_2[0])), "r"((uint32_t)(lut_lo_2[1])), "r"(_nv_selectors));
                                        asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_lo_2[0])), "r"((uint32_t)(lut_lo_2[1])), "r"(_nv_selectors >> 16));
                                        asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                        const uint32_t _nv_shifted = _nv_packed << 4;
                                        asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                        decoded_lo_hi_2 = _nv_hi;
                                        decoded_lo_lo_2 = _nv_lo;
                                    }
                                    {
                                        const uint32_t _nv_packed = (uint32_t)(packed_hi_2[0]);
                                        const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                        uint32_t _nv_hi, _nv_lo;
                                        asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut_hi_2[0])), "r"((uint32_t)(lut_hi_2[1])), "r"(_nv_selectors));
                                        asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut_hi_2[0])), "r"((uint32_t)(lut_hi_2[1])), "r"(_nv_selectors >> 16));
                                        asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                        const uint32_t _nv_shifted = _nv_packed << 4;
                                        asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                        decoded_hi_hi_2 = _nv_hi;
                                        decoded_hi_lo_2 = _nv_lo;
                                    }
                                    unsigned int keep_lo_2 = decoded_lo_hi_2;
                                    unsigned int ship_lo_2 = decoded_lo_lo_2;
                                    unsigned int keep_hi_2 = decoded_hi_hi_2;
                                    unsigned int ship_hi_2 = decoded_hi_lo_2;
                                    if ((lane & 1) != 0) {
                                        keep_lo_2 = decoded_lo_lo_2;
                                        ship_lo_2 = decoded_lo_hi_2;
                                        keep_hi_2 = decoded_hi_lo_2;
                                        ship_hi_2 = decoded_hi_hi_2;
                                    }
                                    unsigned int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, ship_lo_2, 1);
                                    unsigned int recv_lo_2 = _shfl_xor_4;
                                    unsigned int _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, ship_hi_2, 1);
                                    unsigned int recv_hi_2 = _shfl_xor_5;
                                    int frag_base_2 = k_slice_2 * 4;
                                    if ((lane & 1) == 0) {
                                        a_frags_0[frag_base_2] = keep_lo_2;
                                        a_frags_0[frag_base_2 + 1] = recv_lo_2;
                                        a_frags_0[frag_base_2 + 2] = keep_hi_2;
                                        a_frags_0[frag_base_2 + 3] = recv_hi_2;
                                    } else {
                                        a_frags_0[frag_base_2] = recv_lo_2;
                                        a_frags_0[frag_base_2 + 1] = keep_lo_2;
                                        a_frags_0[frag_base_2 + 2] = recv_hi_2;
                                        a_frags_0[frag_base_2 + 3] = keep_hi_2;
                                    }
                                    #pragma unroll
                                    for (int frag_word_2 = 0; frag_word_2 < 4; frag_word_2++) {
                                        asm volatile("" : "+r"(a_frags_0[frag_base_2 + frag_word_2]) :: "memory");
                                    }
                                }
                            }
                            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                            #pragma unroll
                            for (int accum_idx_3 = 0; accum_idx_3 < 4; accum_idx_3++) {
                                asm volatile("" : "+f"(swap_accum_1[accum_idx_3]) :: "memory");
                            }
                            #pragma unroll
                            for (int frag_idx_1 = 0; frag_idx_1 < 16; frag_idx_1++) {
                                asm volatile("" : "+r"(a_frags_1[frag_idx_1]) :: "memory");
                            }
                            int col_idx_7 = lane % 4;
                            #pragma unroll
                            for (int token_chunk_1 = 0; token_chunk_1 < 1; token_chunk_1++) {
                                int token_0_1 = token_chunk_1 * 8 + col_idx_7 * 2;
                                int token_1_1 = token_0_1 + 1;
                                int accum_offset_1 = 32 + token_chunk_1 * 4;
                                if (token_0_1 < math_task_valid_m) {
                                    float scale_0_1[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_0_1[0])) : "r"(cur_sfa_addr + token_0_1 * 4));
                                    final_accum[accum_offset_1] = final_accum[accum_offset_1] + scale_0_1[0] * swap_accum_1[token_chunk_1 * 4];
                                    final_accum[accum_offset_1 + 2] = final_accum[accum_offset_1 + 2] + scale_0_1[0] * swap_accum_1[token_chunk_1 * 4 + 2];
                                }
                                if (token_1_1 < math_task_valid_m) {
                                    float scale_1_1[1];
                                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_1_1[0])) : "r"(cur_sfa_addr + token_1_1 * 4));
                                    final_accum[accum_offset_1 + 1] = final_accum[accum_offset_1 + 1] + scale_1_1[0] * swap_accum_1[token_chunk_1 * 4 + 1];
                                    final_accum[accum_offset_1 + 3] = final_accum[accum_offset_1 + 3] + scale_1_1[0] * swap_accum_1[token_chunk_1 * 4 + 3];
                                }
                            }
                            if (elect_sync()) {
                                mbarrier_arrive(l1_stage_empty_addr + (cur_stage) * 8);
                            }
                        }
                    }
                    if (math_task_phase == 1) {
                        int epilogue_warp_idx = warp - 4;
                        int epilogue_wg_idx = epilogue_warp_idx / 4;
                        int warp_idx_in_wg = epilogue_warp_idx % 4;
                        int epilogue_thread_idx = (unsigned int)(epilogue_warp_idx * 32) + lane;
                        int row_idx = lane / 4;
                        int col_idx_1 = lane % 4;
                        int wg_l1_out_n_idx = epilogue_wg_idx * 64;
                        int m_idx = (int)math_task_words[4] * 8;
                        float _vec_load_9[1];
                        {
                            uint32_t _scalar_bits_1;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_1) : "l"((const void*)(l1_global_scales + ((int)math_task_words[1]))) : "memory");
                            _vec_load_9[0] = __uint_as_float(_scalar_bits_1);
                        }
                        float l1_global_scale = _vec_load_9[0];
                        float swap_v0[2];
                        float swap_v1[2];
                        swap_v0[0] = 0.0f;
                        swap_v0[1] = 0.0f;
                        swap_v1[0] = 0.0f;
                        swap_v1[1] = 0.0f;
                        #pragma unroll
                        for (int token_chunk_2 = 0; token_chunk_2 < 1; token_chunk_2++) {
                            int token_0_2 = token_chunk_2 * 8 + col_idx_1 * 2;
                            int token_1_2 = token_0_2 + 1;
                            float v0_amax = 0.0f;
                            float v1_amax = 0.0f;
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
                                if (token_0_2 < math_task_valid_m) {
                                    float _min_2 = fminf(final_accum[half * 32 + token_chunk_2 * 4] * l1_global_scale, 10.0f);
                                    float gate_0 = _min_2;
                                    float _min_3 = fminf(final_accum[half * 32 + token_chunk_2 * 4 + 2] * l1_global_scale, 10.0f);
                                    float _max_0 = max_noftz(_min_3, -10.0f);
                                    float up_0 = _max_0;
                                    float _vec_load_10[1];
                                    {
                                        _vec_load_10[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4631539712) + m_idx + token_0_2);
                                    }
                                    float _expf_0 = __expf(-gate_0);
                                    float _rcp_0 = approx_rcp(1.0f + _expf_0);
                                    float sigmoid_0 = _rcp_0;
                                    float value_0 = gate_0 * sigmoid_0 * up_0 * _vec_load_10[0];
                                    swap_v0[half + token_chunk_2] = value_0;
                                    float _fabs_0 = fabsf(value_0);
                                    float _max_1 = max_noftz(v0_amax, _fabs_0);
                                    v0_amax = _max_1;
                                }
                                if (token_1_2 < math_task_valid_m) {
                                    float _min_4 = fminf(final_accum[half * 32 + token_chunk_2 * 4 + 1] * l1_global_scale, 10.0f);
                                    float gate_1 = _min_4;
                                    float _min_5 = fminf(final_accum[half * 32 + token_chunk_2 * 4 + 3] * l1_global_scale, 10.0f);
                                    float _max_2 = max_noftz(_min_5, -10.0f);
                                    float up_1 = _max_2;
                                    float _vec_load_11[1];
                                    {
                                        _vec_load_11[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4631539712) + m_idx + token_1_2);
                                    }
                                    float _expf_1 = __expf(-gate_1);
                                    float _rcp_1 = approx_rcp(1.0f + _expf_1);
                                    float sigmoid_1 = _rcp_1;
                                    float value_1 = gate_1 * sigmoid_1 * up_1 * _vec_load_11[0];
                                    swap_v1[half + token_chunk_2] = value_1;
                                    float _fabs_1 = fabsf(value_1);
                                    float _max_3 = max_noftz(v1_amax, _fabs_1);
                                    v1_amax = _max_3;
                                }
                            }
                            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, v0_amax, 4);
                            float peer_0 = _shfl_xor_10;
                            float _max_4 = max_noftz(v0_amax, peer_0);
                            v0_amax = _max_4;
                            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, v1_amax, 4);
                            float peer_1 = _shfl_xor_11;
                            float _max_5 = max_noftz(v1_amax, peer_1);
                            v1_amax = _max_5;
                            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, v0_amax, 8);
                            peer_0 = _shfl_xor_12;
                            float _max_6 = max_noftz(v0_amax, peer_0);
                            v0_amax = _max_6;
                            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, v1_amax, 8);
                            peer_1 = _shfl_xor_13;
                            float _max_7 = max_noftz(v1_amax, peer_1);
                            v1_amax = _max_7;
                            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, v0_amax, 16);
                            peer_0 = _shfl_xor_14;
                            float _max_8 = max_noftz(v0_amax, peer_0);
                            v0_amax = _max_8;
                            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, v1_amax, 16);
                            peer_1 = _shfl_xor_15;
                            float _max_9 = max_noftz(v1_amax, peer_1);
                            v1_amax = _max_9;
                            if (row_idx == 0) {
                                if (token_0_2 < math_task_valid_m) {
                                    l1_amax_smem[token_0_2 * 8 + epilogue_warp_idx] = v0_amax;
                                }
                                if (token_1_2 < math_task_valid_m) {
                                    l1_amax_smem[token_1_2 * 8 + epilogue_warp_idx] = v1_amax;
                                }
                            }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (epilogue_thread_idx < math_task_valid_m) {
                            float amax = 0.0f;
                            #pragma unroll
                            for (int source_warp = 0; source_warp < 8; source_warp++) {
                                float _max_10 = max_noftz(amax, l1_amax_smem[epilogue_thread_idx * 8 + source_warp]);
                                amax = _max_10;
                            }
                            float scaled_amax = amax * 0.002232142857142857f;
                            unsigned int scaled_bits = 0;
                            scaled_bits = reinterpret_cast<unsigned int*>(&scaled_amax)[0];
                            int exponent = (int)(scaled_bits >> 23);
                            unsigned int mantissa = scaled_bits & 8388607;
                            int scale_exponent = exponent - 127;
                            if (mantissa != 0) {
                                scale_exponent = scale_exponent + 1;
                            }
                            unsigned int scale_bits = (unsigned int)(scale_exponent + 127) << 23;
                            unsigned int scale_inv_bits = (unsigned int)(-scale_exponent + 127) << 23;
                            float scale = 0.0f;
                            float scale_inv = 0.0f;
                            scale = reinterpret_cast<float*>(&scale_bits)[0];
                            scale_inv = reinterpret_cast<float*>(&scale_inv_bits)[0];
                            int sf_index = (int)math_task_words[3] * 6635520 + m_idx + epilogue_thread_idx;
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 5907218432)) + (sf_index)) = scale;
                            l1_amax_smem[epilogue_thread_idx * 8] = scale_inv;
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        #pragma unroll
                        for (int token_chunk_3 = 0; token_chunk_3 < 1; token_chunk_3++) {
                            int token_0_3 = token_chunk_3 * 8 + col_idx_1 * 2;
                            int token_1_3 = token_0_3 + 1;
                            #pragma unroll
                            for (int half_1 = 0; half_1 < 2; half_1++) {
                                int out_col = wg_l1_out_n_idx + half_1 * 32 + warp_idx_in_wg * 8 + row_idx;
                                float quant_pair[4];
                                quant_pair[0] = 0.0f;
                                quant_pair[1] = 0.0f;
                                quant_pair[2] = 0.0f;
                                quant_pair[3] = 0.0f;
                                if (token_0_3 < math_task_valid_m) {
                                    float sf_inv_0 = l1_amax_smem[token_0_3 * 8];
                                    quant_pair[0] = swap_v0[half_1 + token_chunk_3] * sf_inv_0;
                                }
                                if (token_1_3 < math_task_valid_m) {
                                    float sf_inv_1 = l1_amax_smem[token_1_3 * 8];
                                    quant_pair[1] = swap_v1[half_1 + token_chunk_3] * sf_inv_1;
                                }
                                uint32_t _fp8_0[1];
                                {
                                    uint32_t _packed;
                                    asm volatile("{\n\t"
                                        ".reg .b16 _lo;\n\t"
                                        ".reg .b16 _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}"
                                        : "=r"(_packed) : "f"(quant_pair[0]), "f"(quant_pair[1]),
                                                           "f"(quant_pair[2]), "f"(quant_pair[3]));
                                    _fp8_0[0] = _packed;
                                }
                                if (token_0_3 < math_task_valid_m) {
                                    l1_output_smem[token_0_3 * 128 + out_col] = _fp8_0[0] & 255;
                                }
                                if (token_1_3 < math_task_valid_m) {
                                    l1_output_smem[token_1_3 * 128 + out_col] = _fp8_0[0] >> 8 & 255;
                                }
                            }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                tma_store_2d(l1_output, (int)math_task_words[3] * 128, m_idx, l1_output_smem_addr);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        __syncwarp();
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                unsigned long long ready_bit = 1;
                                ready_bit = ready_bit << (unsigned long long)(int)math_task_words[3];
                                asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 214016)[(int)math_task_words[4]]), "l"((unsigned long long)(ready_bit)) : "memory");
                            }
                        }
                        __syncwarp();
                    } else {
                        float _vec_load_12[1];
                        {
                            uint32_t _scalar_bits_2;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_2) : "l"((const void*)(l2_global_scales + ((int)math_task_words[1]))) : "memory");
                            _vec_load_12[0] = __uint_as_float(_scalar_bits_2);
                        }
                        float l2_global_scale = _vec_load_12[0];
                        int epilogue_warp_idx_1 = warp - 4;
                        int epilogue_wg_idx_1 = epilogue_warp_idx_1 / 4;
                        int warp_idx_in_wg_1 = epilogue_warp_idx_1 % 4;
                        int row_idx_1 = lane / 4;
                        int col_idx_2 = lane % 4;
                        int r_0 = warp_idx_in_wg_1 * 16 + row_idx_1;
                        int r_1 = r_0 + 8;
                        int wg_n_idx = epilogue_wg_idx_1 * 128;
                        int num_swap_token_chunks = (math_task_valid_m + 7) / 8;
                        #pragma unroll
                        for (int token_chunk_4 = 0; token_chunk_4 < 1; token_chunk_4++) {
                            if (num_swap_token_chunks > token_chunk_4) {
                                int token_0_4 = token_chunk_4 * 8 + col_idx_2 * 2;
                                int token_1_4 = token_0_4 + 1;
                                #pragma unroll
                                for (int half_2 = 0; half_2 < 2; half_2++) {
                                    int accum_offset_2 = half_2 * 32 + token_chunk_4 * 4;
                                    int col_offset = half_2 * 64;
                                    if (token_0_4 < math_task_valid_m) {
                                        {
                                            __nv_bfloat16 _bval_3 = __float2bfloat16_rn(final_accum[accum_offset_2] * l2_global_scale);
                                            uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                                            uint32_t _addr_3 = static_cast<uint32_t>((l2_output_smem_addr + (unsigned int)(token_0_4 * 512 + (wg_n_idx + col_offset + r_0) * 2)));
                                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
                                        }
                                        {
                                            __nv_bfloat16 _bval_4 = __float2bfloat16_rn(final_accum[accum_offset_2 + 2] * l2_global_scale);
                                            uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                                            uint32_t _addr_4 = static_cast<uint32_t>((l2_output_smem_addr + (unsigned int)(token_0_4 * 512 + (wg_n_idx + col_offset + r_1) * 2)));
                                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4) : "memory");
                                        }
                                    }
                                    if (token_1_4 < math_task_valid_m) {
                                        {
                                            __nv_bfloat16 _bval_5 = __float2bfloat16_rn(final_accum[accum_offset_2 + 1] * l2_global_scale);
                                            uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                                            uint32_t _addr_5 = static_cast<uint32_t>((l2_output_smem_addr + (unsigned int)(token_1_4 * 512 + (wg_n_idx + col_offset + r_0) * 2)));
                                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
                                        }
                                        {
                                            __nv_bfloat16 _bval_6 = __float2bfloat16_rn(final_accum[accum_offset_2 + 3] * l2_global_scale);
                                            uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                                            uint32_t _addr_6 = static_cast<uint32_t>((l2_output_smem_addr + (unsigned int)(token_1_4 * 512 + (wg_n_idx + col_offset + r_1) * 2)));
                                            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                        if (warp < 8) {
                            asm volatile("bar.sync 3, 128;" ::: "memory");
                        } else {
                            asm volatile("bar.sync 4, 128;" ::: "memory");
                        }
                        int row_in_warp_block = lane / 16;
                        int lane_in_row = lane % 16;
                        int n_idx = (int)math_task_words[3] * 256 + wg_n_idx;
                        #pragma unroll
                        for (int row_iter = 0; row_iter < ((1) ? 4 : 8); row_iter++) {
                            int token = warp_idx_in_wg_1 * 16 + row_iter * 2 + row_in_warp_block;
                            if (token >= math_task_valid_m) {
                                break;
                            }
                            int metadata_base = ((int)math_task_words[4] * 8 + token) * 3;
                            int _vec_load_13[1];
                            {
                                _vec_load_13[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base);
                            }
                            int _vec_load_14[1];
                            {
                                _vec_load_14[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base + 1);
                            }
                            int _vec_load_15[1];
                            {
                                _vec_load_15[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base + 2);
                            }
                            int dst_rank_2 = _vec_load_13[0];
                            int dst_token = _vec_load_14[0];
                            int dst_topk = _vec_load_15[0];
                            unsigned int packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(l2_output_smem_addr + (unsigned int)((token * 256 + wg_n_idx + lane_in_row * 8) * 2)));
                            unsigned long long dst_word_offset = (((unsigned long long)dst_topk * 8448 + (unsigned long long)dst_token) * 7168 + (unsigned long long)n_idx + (unsigned long long)lane_in_row * 8) / 2;
                            reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_2]) + 7181238272) + dst_word_offset)[0] = reinterpret_cast<int4*>(packed)[0];
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_7 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
                        const unsigned int _gs_add_7 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_7;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_7) : "l"(_gs_ptr_7), "r"(_gs_add_7) : "memory");
                        unsigned int _gs_new_7;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_7) : "l"(_gs_ptr_7) : "memory");
                        } while (((_gs_new_7 ^ _gs_old_7) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (bid == 0 && warp == 4) {
                {
                    const unsigned int _rz_status_8 = (*reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)) & 3u;
                    const unsigned int _rz_phase_8 = _rz_status_8 & 1u;
                    const int _rz_sign_8 = (int)(_rz_status_8 >> 1);
                    if ((int)(lane) < (int)(pg_world)) {
                        int* _rz_peer_8 = reinterpret_cast<int*>(reinterpret_cast<char*>((sym_buffer_peers)[lane]) + (unsigned long long)(20)) + _rz_phase_8;
                        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(_rz_peer_8), "r"(_rz_sign_8 ? -1 : 1) : "memory");
                    }
                    __syncwarp();
                    if ((int)(lane) == 0) {
                        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)), "r"(1u) : "memory");
                        const int _rz_target_8 = _rz_sign_8 ? 0 : (int)(pg_world);
                        int* _rz_local_8 = reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4) + (unsigned int)(1)) + _rz_phase_8;
                        while (true) {
                            int _rz_seen_8;
                            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_rz_seen_8) : "l"(_rz_local_8) : "memory");
                            if (_rz_seen_8 == _rz_target_8) break;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_9 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
                        const unsigned int _gs_add_9 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_9;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_9) : "l"(_gs_ptr_9), "r"(_gs_add_9) : "memory");
                        unsigned int _gs_new_9;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_9) : "l"(_gs_ptr_9) : "memory");
                        } while (((_gs_new_9 ^ _gs_old_9) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            int epilogue_warp_idx_2 = warp - 4;
            unsigned int combine_phase = 0;
            unsigned int combine_load_stage = 0;
            #pragma unroll 1
            for (unsigned int combine_token_idx = bid * 8 + epilogue_warp_idx_2; combine_token_idx < num_tokens; combine_token_idx += 624) {
                int stored_topk_slot_idx = -1;
                if (lane < 6) {
                    long long _vec_load_16[1];
                    {
                        uint64_t _scalar_bits_10;
                        asm volatile("ld.global.nc.b64 %0, [%1];"
                            : "=l"(_scalar_bits_10) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 171862016) + (combine_token_idx * 6 + lane))) : "memory");
                        _vec_load_16[0] = (long long)_scalar_bits_10;
                    }
                    stored_topk_slot_idx = (int)_vec_load_16[0];
                }
                unsigned int _vote_2 = __ballot_sync(4294967295, stored_topk_slot_idx >= 0);
                unsigned int total_mask = _vote_2;
                #pragma unroll
                for (int combine_chunk = 0; combine_chunk < 2; combine_chunk++) {
                    unsigned int chunk_byte_offset = combine_chunk * 7168;
                    unsigned int combine_mask = total_mask;
                    int do_reduce = 0;
                    if (combine_mask != 0) {
                        int _ffs_1 = __ffs(combine_mask);
                        int slot_idx_1 = _ffs_1 - 1;
                        combine_mask = combine_mask ^ (unsigned int)(1 << slot_idx_1);
                        unsigned int load_barrier_stage = (unsigned int)(epilogue_warp_idx_2 * 2) + combine_load_stage;
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(combine_full_addr + (load_barrier_stage) * 8, 7168);
                            cp_async_bulk_gmem2smem(combine_load_smem_addr + load_barrier_stage * 7168, reinterpret_cast<uint8_t*>(sym_buffer) + (7181238272 + ((unsigned long long)slot_idx_1 * 8448 + (unsigned long long)combine_token_idx) * 7168 * 2 + chunk_byte_offset), 7168, combine_full_addr + (load_barrier_stage) * 8);
                        }
                        __syncwarp();
                        do_reduce = 1;
                    }
                    float reduced[112];
                    reduced[0] = 0.0f;
                    reduced[1] = 0.0f;
                    reduced[2] = 0.0f;
                    reduced[3] = 0.0f;
                    reduced[4] = 0.0f;
                    reduced[5] = 0.0f;
                    reduced[6] = 0.0f;
                    reduced[7] = 0.0f;
                    reduced[8] = 0.0f;
                    reduced[9] = 0.0f;
                    reduced[10] = 0.0f;
                    reduced[11] = 0.0f;
                    reduced[12] = 0.0f;
                    reduced[13] = 0.0f;
                    reduced[14] = 0.0f;
                    reduced[15] = 0.0f;
                    reduced[16] = 0.0f;
                    reduced[17] = 0.0f;
                    reduced[18] = 0.0f;
                    reduced[19] = 0.0f;
                    reduced[20] = 0.0f;
                    reduced[21] = 0.0f;
                    reduced[22] = 0.0f;
                    reduced[23] = 0.0f;
                    reduced[24] = 0.0f;
                    reduced[25] = 0.0f;
                    reduced[26] = 0.0f;
                    reduced[27] = 0.0f;
                    reduced[28] = 0.0f;
                    reduced[29] = 0.0f;
                    reduced[30] = 0.0f;
                    reduced[31] = 0.0f;
                    reduced[32] = 0.0f;
                    reduced[33] = 0.0f;
                    reduced[34] = 0.0f;
                    reduced[35] = 0.0f;
                    reduced[36] = 0.0f;
                    reduced[37] = 0.0f;
                    reduced[38] = 0.0f;
                    reduced[39] = 0.0f;
                    reduced[40] = 0.0f;
                    reduced[41] = 0.0f;
                    reduced[42] = 0.0f;
                    reduced[43] = 0.0f;
                    reduced[44] = 0.0f;
                    reduced[45] = 0.0f;
                    reduced[46] = 0.0f;
                    reduced[47] = 0.0f;
                    reduced[48] = 0.0f;
                    reduced[49] = 0.0f;
                    reduced[50] = 0.0f;
                    reduced[51] = 0.0f;
                    reduced[52] = 0.0f;
                    reduced[53] = 0.0f;
                    reduced[54] = 0.0f;
                    reduced[55] = 0.0f;
                    reduced[56] = 0.0f;
                    reduced[57] = 0.0f;
                    reduced[58] = 0.0f;
                    reduced[59] = 0.0f;
                    reduced[60] = 0.0f;
                    reduced[61] = 0.0f;
                    reduced[62] = 0.0f;
                    reduced[63] = 0.0f;
                    reduced[64] = 0.0f;
                    reduced[65] = 0.0f;
                    reduced[66] = 0.0f;
                    reduced[67] = 0.0f;
                    reduced[68] = 0.0f;
                    reduced[69] = 0.0f;
                    reduced[70] = 0.0f;
                    reduced[71] = 0.0f;
                    reduced[72] = 0.0f;
                    reduced[73] = 0.0f;
                    reduced[74] = 0.0f;
                    reduced[75] = 0.0f;
                    reduced[76] = 0.0f;
                    reduced[77] = 0.0f;
                    reduced[78] = 0.0f;
                    reduced[79] = 0.0f;
                    reduced[80] = 0.0f;
                    reduced[81] = 0.0f;
                    reduced[82] = 0.0f;
                    reduced[83] = 0.0f;
                    reduced[84] = 0.0f;
                    reduced[85] = 0.0f;
                    reduced[86] = 0.0f;
                    reduced[87] = 0.0f;
                    reduced[88] = 0.0f;
                    reduced[89] = 0.0f;
                    reduced[90] = 0.0f;
                    reduced[91] = 0.0f;
                    reduced[92] = 0.0f;
                    reduced[93] = 0.0f;
                    reduced[94] = 0.0f;
                    reduced[95] = 0.0f;
                    reduced[96] = 0.0f;
                    reduced[97] = 0.0f;
                    reduced[98] = 0.0f;
                    reduced[99] = 0.0f;
                    reduced[100] = 0.0f;
                    reduced[101] = 0.0f;
                    reduced[102] = 0.0f;
                    reduced[103] = 0.0f;
                    reduced[104] = 0.0f;
                    reduced[105] = 0.0f;
                    reduced[106] = 0.0f;
                    reduced[107] = 0.0f;
                    reduced[108] = 0.0f;
                    reduced[109] = 0.0f;
                    reduced[110] = 0.0f;
                    reduced[111] = 0.0f;
                    while (do_reduce != 0) {
                        do_reduce = 0;
                        if (combine_mask != 0) {
                            int _ffs_2 = __ffs(combine_mask);
                            int next_slot_idx = _ffs_2 - 1;
                            combine_mask = combine_mask ^ (unsigned int)(1 << next_slot_idx);
                            unsigned int next_load_stage = combine_load_stage ^ 1;
                            unsigned int next_barrier_stage = (unsigned int)(epilogue_warp_idx_2 * 2) + next_load_stage;
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(combine_full_addr + (next_barrier_stage) * 8, 7168);
                                cp_async_bulk_gmem2smem(combine_load_smem_addr + next_barrier_stage * 7168, reinterpret_cast<uint8_t*>(sym_buffer) + (7181238272 + ((unsigned long long)next_slot_idx * 8448 + (unsigned long long)combine_token_idx) * 7168 * 2 + chunk_byte_offset), 7168, combine_full_addr + (next_barrier_stage) * 8);
                            }
                            __syncwarp();
                            do_reduce = 1;
                        }
                        unsigned int current_barrier_stage = (unsigned int)(epilogue_warp_idx_2 * 2) + combine_load_stage;
                        mbarrier_wait(combine_full_addr + (current_barrier_stage) * 8, combine_phase);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        #pragma unroll
                        for (int combine_vec = 0; combine_vec < 14; combine_vec++) {
                            unsigned int combine_packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 3]))
                                : "r"(combine_load_smem_addr + current_barrier_stage * 7168 + ((unsigned int)(combine_vec * 32) + lane) * 16));
                            float combine_packed_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&combine_packed_f32[_pair * 2])[0]), "=f"((&combine_packed_f32[_pair * 2])[1])
                                    : "r"(combine_packed[_pair]));
                            }
                            #pragma unroll
                            for (int combine_elem = 0; combine_elem < 8; combine_elem++) {
                                reduced[combine_vec * 8 + combine_elem] = reduced[combine_vec * 8 + combine_elem] + combine_packed_f32[combine_elem];
                            }
                        }
                        combine_phase = combine_phase ^ combine_load_stage;
                        combine_load_stage = combine_load_stage ^ 1;
                    }
                    asm volatile("cp.async.bulk.wait_group 0;");
                    __syncwarp();
                    #pragma unroll
                    for (int combine_vec_1 = 0; combine_vec_1 < 14; combine_vec_1++) {
                        float cast_values[8];
                        #pragma unroll
                        for (int cast_elem = 0; cast_elem < 8; cast_elem++) {
                            cast_values[cast_elem] = reduced[combine_vec_1 * 8 + cast_elem];
                        }
                        unsigned int casted[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(cast_values[_lp*2 + 0], cast_values[_lp*2+1 + 0]));
                            casted[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int cast_word = 0; cast_word < 4; cast_word++) {
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx_2 * 7168) + ((unsigned int)(combine_vec_1 * 32) + lane) * 16 + (unsigned int)(cast_word * 4)), "r"((casted[cast_word])));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            void* _cpbulk_dst_11 = reinterpret_cast<void*>(output_bf16 + ((unsigned long long)combine_token_idx * 7168 + (unsigned long long)(chunk_byte_offset / 2)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_11), "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx_2 * 7168)), "r"((uint32_t)(7168))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    __syncwarp();
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

