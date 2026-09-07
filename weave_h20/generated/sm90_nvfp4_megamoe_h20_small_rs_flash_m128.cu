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
#define NUM_L1_PIPE_STAGES 3
#define NUM_COMBINE_PIPE_STAGES 16
#define SMEM_SMEM_EXPERT_COUNT_OFF 1024
#define SMEM_SMEM_EXPERT_COUNT_STAGE_BYTES 1024
#define SMEM_SMEM_EXPERT_COUNT_STRIDE 1024
#define SMEM_PULL_STAGE_OFF 2048
#define SMEM_PULL_STAGE_STAGE_BYTES 8192
#define SMEM_PULL_STAGE_STRIDE 8192
#define SMEM_TASK_INFO_SMEM_OFF 10240
#define SMEM_TASK_INFO_SMEM_STAGE_BYTES 32
#define SMEM_TASK_INFO_SMEM_STRIDE 32
#define SMEM_L1_ACT_SMEM_OFF 11264
#define SMEM_L1_ACT_SMEM_STAGE_BYTES 8192
#define SMEM_L1_ACT_SMEM_STRIDE 8192
#define SMEM_L1_SFA_SMEM_OFF 35840
#define SMEM_L1_SFA_SMEM_STAGE_BYTES 256
#define SMEM_L1_SFA_SMEM_STRIDE 256
#define SMEM_L1_PACKED_SMEM_OFF 36608
#define SMEM_L1_PACKED_SMEM_STAGE_BYTES 20480
#define SMEM_L1_PACKED_SMEM_STRIDE 20480
#define SMEM_L1_LUT_SMEM_OFF 98048
#define SMEM_L1_LUT_SMEM_STAGE_BYTES 1024
#define SMEM_L1_LUT_SMEM_STRIDE 1024
#define SMEM_L1_DECODED_SMEM_OFF 99328
#define SMEM_L1_DECODED_SMEM_STAGE_BYTES 32768
#define SMEM_L1_DECODED_SMEM_STRIDE 32768
#define SMEM_L1_OUTPUT_SMEM_OFF 197632
#define SMEM_L1_OUTPUT_SMEM_STAGE_BYTES 8192
#define SMEM_L1_OUTPUT_SMEM_STRIDE 8192
#define SMEM_L1_AMAX_SMEM_OFF 205824
#define SMEM_L1_AMAX_SMEM_STAGE_BYTES 2048
#define SMEM_L1_AMAX_SMEM_STRIDE 2048
#define SMEM_L2_OUTPUT_SMEM_OFF 197632
#define SMEM_L2_OUTPUT_SMEM_STAGE_BYTES 32768
#define SMEM_L2_OUTPUT_SMEM_STRIDE 32768
#define SMEM_COMBINE_LOAD_SMEM_OFF 1024
#define SMEM_COMBINE_LOAD_SMEM_STAGE_BYTES 8192
#define SMEM_COMBINE_LOAD_SMEM_STRIDE 8192
#define SMEM_COMBINE_STORE_SMEM_OFF 132096
#define SMEM_COMBINE_STORE_SMEM_STAGE_BYTES 8192
#define SMEM_COMBINE_STORE_SMEM_STRIDE 8192
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
    #define l1_stage_empty_addr (mbar_base + 72)
    #define combine_full_addr (mbar_base + 96)

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
    uint8_t* pull_stage = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int pull_stage_addr = smem + 2048;
    unsigned int* task_info_smem = reinterpret_cast<unsigned int*>(smem_raw + 10240);
    const int task_info_smem_addr = smem + 10240;
    uint8_t* l1_act_smem = reinterpret_cast<uint8_t*>(smem_raw + 11264);
    const int l1_act_smem_addr = smem + 11264;
    float* l1_sfa_smem = reinterpret_cast<float*>(smem_raw + 35840);
    const int l1_sfa_smem_addr = smem + 35840;
    uint8_t* l1_packed_smem = reinterpret_cast<uint8_t*>(smem_raw + 36608);
    const int l1_packed_smem_addr = smem + 36608;
    unsigned int* l1_lut_smem = reinterpret_cast<unsigned int*>(smem_raw + 98048);
    const int l1_lut_smem_addr = smem + 98048;
    uint8_t* l1_decoded_smem = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int l1_decoded_smem_addr = smem + 99328;
    uint8_t* l1_output_smem = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int l1_output_smem_addr = smem + 197632;
    float* l1_amax_smem = reinterpret_cast<float*>(smem_raw + 205824);
    const int l1_amax_smem_addr = smem + 205824;
    __nv_bfloat16* l2_output_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int l2_output_smem_addr = smem + 197632;
    uint8_t* combine_load_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int combine_load_smem_addr = smem + 1024;
    uint8_t* combine_store_smem = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int combine_store_smem_addr = smem + 132096;

    // Mbarrier init (6 groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

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
            // l1_stage_full: 3 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            // l1_stage_empty: 3 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // --- pipeline 'combine_pipe' ---
            // combine_full: 16 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
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
                for (int expert = lane; expert < 256; expert += 32) {
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
                                : "=l"(_scalar_bits_0) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 110452096) + (token_topk_idx))) : "memory");
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
            for (int expert_1 = tid; expert_1 < 256; expert_1 += 64) {
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
                                : "=l"(_scalar_bits_1) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 110452096) + (token_topk_idx_1))) : "memory");
                            _vec_load_1[0] = (long long)_scalar_bits_1;
                        }
                        int expert_idx_1 = (int)_vec_load_1[0];
                        if (expert_idx_1 >= 0) {
                            int dst_rank = expert_idx_1 / 32;
                            unsigned int _smem_atomic_old_1;
                            asm volatile("atom.cta.shared.add.u32 %0, [%1], %2;" : "=r"(_smem_atomic_old_1) : "r"((unsigned int)((smem_expert_count_addr) + (expert_idx_1) * 4)), "r"((unsigned int)(1)) : "memory");
                            unsigned int dst_slot = _smem_atomic_old_1;
                            unsigned long long dst_index = (unsigned long long)(expert_idx_1 % 32) * 540672 + (unsigned long long)pg_rank * 67584 + (unsigned long long)dst_slot;
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank]) + 621952)) + (dst_index)) = token_topk_idx_1;
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
                for (int expert_2 = tid; expert_2 < 256; expert_2 += 64) {
                    int dst_rank_1 = expert_2 / 32;
                    int dst_local_expert = expert_2 % 32;
                    unsigned long long _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128) + expert_2);
                    }
                    unsigned long long expert_status = _vec_load_2[0];
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_1]) + 2176)) + (pg_rank * 32 + dst_local_expert)) = expert_status & 4294967295;
                    unsigned long long _atomic_old_1;
                    asm volatile("atom.sys.global.add.u64 %0, [%1], %2;" : "=l"(_atomic_old_1) : "l"(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_1]) + 4224)[dst_local_expert]), "l"((unsigned long long)(expert_status)) : "memory");
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
                if (lane < 32) {
                    while ((unsigned int)(ready0 >> 32) != 624) {
                        unsigned long long _sys_volatile_0;
                        asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_0) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4224)[lane])) : "memory");
                        ready0 = _sys_volatile_0;
                    }
                    count0 = (unsigned int)ready0;
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
                        if (current_expert >= 32) {
                            break;
                        }
                        expert_pool_block_offset = expert_pool_block_offset + (expert_end - expert_start + 64 - 1) / 64;
                        expert_start = expert_end;
                        unsigned int lane_count = count0;
                        if (current_expert >= 32) {
                            lane_count = count1;
                        }
                        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_count, current_expert % 32);
                        unsigned int current_count = _shfl_0;
                        expert_end = expert_end + current_count;
                    }
                    if (current_expert >= 32) {
                        break;
                    }
                    if (old_expert != current_expert && lane < 8) {
                        unsigned long long _vec_load_3[1];
                        {
                            _vec_load_3[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 2176) + lane * 32 + (unsigned int)current_expert);
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
                        _vec_load_4[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 621952) + src_index);
                    }
                    unsigned int src_token_topk_idx = (unsigned int)_vec_load_4[0];
                    unsigned int src_token_idx = src_token_topk_idx / 6;
                    unsigned int src_topk_idx = src_token_topk_idx % 6;
                    unsigned int pool_token_idx = expert_pool_block_offset * 64 + token_idx_in_expert;
                    int pull_smem_addr = pull_stage_addr + warp * 4096;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(pull_full_addr + (warp) * 8, 4096);
                        // nvlink_pull: smem(pull_smem_addr) <- peers[selected_rank] + 74767744 + (unsigned long long)src_token_idx * 4096, 4096B
                        {
                            const void* __remote = (const void*)((const char*)((sym_buffer_peers)[selected_rank]) + (uint64_t)(74767744 + (unsigned long long)src_token_idx * 4096));
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1], %2, [%3], %4;"
                                :: "r"(pull_smem_addr), "l"(__remote), "r"((uint32_t)(4096)), "r"(pull_full_addr + (warp) * 8), "l"(0x12F0000000000000ULL)
                                : "memory");
                        }
                    }
                    mbarrier_wait(pull_full_addr + (warp) * 8, _phase_pull_full);
                    _phase_pull_full ^= 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    for (int sf_group = 0; sf_group < 1; sf_group++) {
                        int sf_idx = (unsigned int)(sf_group * 32) + lane;
                        if (sf_idx < 32) {
                            float _vec_load_5[1];
                            {
                                _vec_load_5[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[selected_rank]) + 109370752) + src_token_idx * 32 + (unsigned int)sf_idx);
                            }
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 1797170560)) + ((unsigned int)(sf_idx * 6586368) + pool_token_idx)) = _vec_load_5[0];
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        float _vec_load_6[1];
                        {
                            _vec_load_6[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[selected_rank]) + 110857600) + src_token_topk_idx);
                        }
                        *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 2640225664)) + (pool_token_idx)) = _vec_load_6[0];
                        {
                            void* _cpbulk_dst_5 = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(reinterpret_cast<uint8_t*>(sym_buffer) + 111060352) + (pool_token_idx * 4096));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;"
                                :: "l"(_cpbulk_dst_5), "r"(pull_smem_addr), "r"((uint32_t)(4096)), "l"(0x1000000000000000ULL)
                                : "memory");
                        }
                        unsigned long long metadata_idx = (unsigned long long)pool_token_idx * 3;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968)) + (metadata_idx)) = (int)selected_rank;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968)) + (metadata_idx + 1)) = (int)src_token_idx;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968)) + (metadata_idx + 2)) = (int)src_topk_idx;
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group 0;");
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(&reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4480)[expert_pool_block_offset + token_idx_in_expert / 64]), "r"((unsigned int)(1)) : "memory");
                    }
                    __syncwarp();
                    pull_token_idx = pull_token_idx + 156;
                }
            }
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int expert_3 = tid; expert_3 < 256; expert_3 += 64) {
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128)) + (expert_3)) = 0;
                }
                if (tid == 0) {
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 7) + (0)) = 0;
                    *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 8) + (0)) = 0;
                }
            } else {
                int cleanup_local_expert = bid - 1;
                if (cleanup_local_expert < 32) {
                    unsigned long long _vec_load_7[1];
                    {
                        _vec_load_7[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4224) + cleanup_local_expert);
                    }
                    unsigned int cleanup_num_tokens = (unsigned int)_vec_load_7[0];
                    unsigned int cleanup_num_m_blocks = (cleanup_num_tokens + 64 - 1) / 64;
                    unsigned int cleanup_prefix_lane = 0;
                    #pragma unroll
                    for (int expert_group = 0; expert_group < 1; expert_group++) {
                        int prefix_expert = (unsigned int)(expert_group * 32) + lane;
                        if (prefix_expert < cleanup_local_expert) {
                            unsigned long long _vec_load_8[1];
                            {
                                _vec_load_8[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4224) + prefix_expert);
                            }
                            unsigned int prefix_num_tokens = (unsigned int)_vec_load_8[0];
                            cleanup_prefix_lane = cleanup_prefix_lane + (prefix_num_tokens + 64 - 1) / 64;
                        }
                    }
                    unsigned int _warp_redux_u32_2;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(cleanup_prefix_lane));
                    unsigned int cleanup_pool_block_offset = _warp_redux_u32_2;
                    asm volatile("bar.sync 0, 64;" ::: "memory");
                    if (warp == 0) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4224)) + (cleanup_local_expert)) = 0;
                    } else if (warp == 1) {
                        __syncwarp();
                    }
                    if (tid < 8) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 2176)) + (tid * 32 + cleanup_local_expert)) = 0;
                    }
                    __syncwarp();
                    #pragma unroll 1
                    for (int cleanup_block = tid; cleanup_block < cleanup_num_m_blocks; cleanup_block += 64) {
                        unsigned int cleanup_pool_block = cleanup_pool_block_offset + (unsigned int)cleanup_block;
                        *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4480)) + (cleanup_pool_block)) = 0;
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 210304)) + (cleanup_pool_block)) = 0;
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
                unsigned int stored_num_tokens[1];
                #pragma unroll
                for (int expert_group_1 = 0; expert_group_1 < 1; expert_group_1++) {
                    unsigned long long ready_status = 0;
                    int local_expert = (unsigned int)(expert_group_1 * 32) + lane;
                    if (local_expert < 32) {
                        while ((unsigned int)(ready_status >> 32) != 624) {
                            unsigned long long _sys_volatile_2;
                            asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_2) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4224)[local_expert])) : "memory");
                            ready_status = _sys_volatile_2;
                        }
                    }
                    stored_num_tokens[expert_group_1] = (unsigned int)ready_status;
                }
                __syncwarp();
                unsigned int lane_num_m_blocks = 0;
                #pragma unroll
                for (int expert_group_2 = 0; expert_group_2 < 1; expert_group_2++) {
                    int local_expert_1 = (unsigned int)(expert_group_2 * 32) + lane;
                    if (local_expert_1 < 32) {
                        lane_num_m_blocks = lane_num_m_blocks + (stored_num_tokens[expert_group_2] + 64 - 1) / 64;
                    }
                }
                unsigned int _warp_redux_u32_3;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(lane_num_m_blocks));
                unsigned int num_total_m_blocks = _warp_redux_u32_3;
                unsigned int num_total_l1_tasks = num_total_m_blocks * 16;
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
                            task_num_n_blocks = 16;
                            task_shape_n = 4096;
                            task_shape_k = 4096;
                        } else {
                            if (lane == 0) {
                                unsigned int _atomic_old_3 = atomicAdd(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 8, 1);
                                task_idx = _atomic_old_3;
                            }
                            unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, task_idx, 0);
                            task_idx = _shfl_2;
                            if (task_idx >= num_total_m_blocks * 16) {
                                break;
                            }
                            if (num_l1_warmup_waves != l1_waves_done) {
                                num_l1_warmup_waves = 1;
                            }
                            task_phase = 2;
                            task_num_n_blocks = 16;
                            task_shape_n = 4096;
                            task_shape_k = 2048;
                            unsigned int required_l1_tasks = (task_idx / 16 + 1) * 16;
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
                        for (int expert_group_3 = 0; expert_group_3 < 1; expert_group_3++) {
                            unsigned int expert_idx_2 = (unsigned int)(expert_group_3 * 32) + lane;
                            unsigned int expert_tokens = stored_num_tokens[expert_group_3];
                            unsigned int expert_m_blocks = (expert_tokens + 64 - 1) / 64;
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
                            unsigned int _vote_1 = __ballot_sync(4294967295, expert_idx_2 < 32 && (task_pool_block >= lane_pool_offset && task_pool_block < lane_pool_offset + expert_m_blocks));
                            unsigned int owner_mask = _vote_1;
                            if (owner_mask != 0) {
                                int _ffs_0 = __ffs(owner_mask);
                                int owner_lane = _ffs_0 - 1;
                                unsigned int owner_m_block = task_pool_block - lane_pool_offset;
                                unsigned int _min_1 = ((expert_tokens - owner_m_block * 64) < (64) ? (expert_tokens - owner_m_block * 64) : (64));
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
                            if (l1_load_stage == 3) { l1_load_stage = 0; _phase_l1_stage_empty ^= 1; }
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
                                unsigned int* _gwa_p_0 = reinterpret_cast<unsigned int*>(&reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 4480)[loader_task_pool_block]);
                                while (true) {
                                    unsigned int _gwa_v_0;
                                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_gwa_v_0) : "l"(_gwa_p_0) : "memory");
                                    if (_gwa_v_0 == (unsigned int)(loader_task_valid_m)) break;
                                }
                            }
                        } else {
                            {
                                unsigned long long* _gwa_p_1 = reinterpret_cast<unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 210304)[loader_task_pool_block]);
                                while (true) {
                                    unsigned long long _gwa_v_1;
                                    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(_gwa_v_1) : "l"(_gwa_p_1) : "memory");
                                    if (_gwa_v_1 == (unsigned long long)(65535)) break;
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
                                    tma_2d_gmem2smem(l1_act_smem_addr + l1_load_stage * 8192, l1_activation, k_block_idx_1 * 128, loader_task_pool_block * 64, l1_stage_full_addr + (l1_load_stage) * 8);
                                    tma_2d_gmem2smem(l1_sfa_smem_addr + l1_load_stage * 256, l1_activation_scale, loader_task_pool_block * 64, k_block_idx_1, l1_stage_full_addr + (l1_load_stage) * 8);
                                } else {
                                    tma_2d_gmem2smem(l1_act_smem_addr + l1_load_stage * 8192, l2_activation, k_block_idx_1 * 128, loader_task_pool_block * 64, l1_stage_full_addr + (l1_load_stage) * 8);
                                    tma_2d_gmem2smem(l1_sfa_smem_addr + l1_load_stage * 256, l2_activation_scale, loader_task_pool_block * 64, k_block_idx_1, l1_stage_full_addr + (l1_load_stage) * 8);
                                }
                                mbarrier_arrive_expect_tx(l1_stage_full_addr + (l1_load_stage) * 8, 8448);
                            }
                        } else {
                            if (elect_sync()) {
                                mbarrier_arrive(l1_stage_full_addr + (l1_load_stage) * 8);
                            }
                        }
                        __syncwarp();
                        l1_load_stage += 1;
                        if (l1_load_stage == 3) { l1_load_stage = 0; _phase_l1_stage_empty ^= 1; }
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
                if (math_task_phase == 1) {
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
                    #pragma unroll 1
                    for (int math_k_block_idx = 0; math_k_block_idx < math_task_shape_k / 128; math_k_block_idx++) {
                        mbarrier_wait(l1_stage_full_addr + (l1_math_stage) * 8, _phase_l1_stage_full);
                        if (math_k_block_idx == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(task_info_empty_addr + (consumed_task_stage) * 8);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int epilogue_warp_idx = warp - 4;
                        int epilogue_thread_idx = (unsigned int)(epilogue_warp_idx * 32) + lane;
                        int warp_idx_in_wg = epilogue_warp_idx % 4;
                        int row_idx = lane / 4;
                        int r_0 = warp_idx_in_wg * 16 + row_idx;
                        int r_1 = r_0 + 8;
                        int packed_row_addr = l1_packed_smem_addr + l1_math_stage * 20480 + (unsigned int)(epilogue_thread_idx * 80);
                        unsigned int scale_words[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scale_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words[(0) + 1]))
                            : "r"(packed_row_addr + 64));
                        float scale_0[1];
                        float scale_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_0[0])) : "r"(l1_sfa_smem_addr + l1_math_stage * 256 + (unsigned int)(r_0 * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_1[0])) : "r"(l1_sfa_smem_addr + l1_math_stage * 256 + (unsigned int)(r_1 * 4)));
                        float accum[64];
                        accum[0] = 0.0f;
                        accum[1] = 0.0f;
                        accum[2] = 0.0f;
                        accum[3] = 0.0f;
                        accum[4] = 0.0f;
                        accum[5] = 0.0f;
                        accum[6] = 0.0f;
                        accum[7] = 0.0f;
                        accum[8] = 0.0f;
                        accum[9] = 0.0f;
                        accum[10] = 0.0f;
                        accum[11] = 0.0f;
                        accum[12] = 0.0f;
                        accum[13] = 0.0f;
                        accum[14] = 0.0f;
                        accum[15] = 0.0f;
                        accum[16] = 0.0f;
                        accum[17] = 0.0f;
                        accum[18] = 0.0f;
                        accum[19] = 0.0f;
                        accum[20] = 0.0f;
                        accum[21] = 0.0f;
                        accum[22] = 0.0f;
                        accum[23] = 0.0f;
                        accum[24] = 0.0f;
                        accum[25] = 0.0f;
                        accum[26] = 0.0f;
                        accum[27] = 0.0f;
                        accum[28] = 0.0f;
                        accum[29] = 0.0f;
                        accum[30] = 0.0f;
                        accum[31] = 0.0f;
                        accum[32] = 0.0f;
                        accum[33] = 0.0f;
                        accum[34] = 0.0f;
                        accum[35] = 0.0f;
                        accum[36] = 0.0f;
                        accum[37] = 0.0f;
                        accum[38] = 0.0f;
                        accum[39] = 0.0f;
                        accum[40] = 0.0f;
                        accum[41] = 0.0f;
                        accum[42] = 0.0f;
                        accum[43] = 0.0f;
                        accum[44] = 0.0f;
                        accum[45] = 0.0f;
                        accum[46] = 0.0f;
                        accum[47] = 0.0f;
                        accum[48] = 0.0f;
                        accum[49] = 0.0f;
                        accum[50] = 0.0f;
                        accum[51] = 0.0f;
                        accum[52] = 0.0f;
                        accum[53] = 0.0f;
                        accum[54] = 0.0f;
                        accum[55] = 0.0f;
                        accum[56] = 0.0f;
                        accum[57] = 0.0f;
                        accum[58] = 0.0f;
                        accum[59] = 0.0f;
                        accum[60] = 0.0f;
                        accum[61] = 0.0f;
                        accum[62] = 0.0f;
                        accum[63] = 0.0f;
                        #pragma unroll
                        for (int accum_idx = 0; accum_idx < 64; accum_idx++) {
                            asm volatile("" : "+f"(accum[accum_idx]) :: "memory");
                        }
                        #pragma unroll
                        for (int quad_in_half = 0; quad_in_half < 2; quad_in_half++) {
                            unsigned int packed_quad[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad[(0) + 3]))
                                : "r"(packed_row_addr + quad_in_half * 16));
                            unsigned int scale_word = scale_words[0];
                            if (quad_in_half >= 2) {
                                scale_word = scale_words[1];
                            }
                            unsigned int scale0 = scale_word >> (unsigned int)((quad_in_half * 2 & 3) * 8) & 127;
                            unsigned int scale1 = scale_word >> (unsigned int)((quad_in_half * 2 + 1 & 3) * 8) & 127;
                            unsigned int lut0[2];
                            unsigned int lut1[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut0[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut0[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale0 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut1[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale1 * 8)));
                            unsigned int q0_hi = 0;
                            unsigned int q0_lo = 0;
                            unsigned int q1_hi = 0;
                            unsigned int q1_lo = 0;
                            unsigned int q2_hi = 0;
                            unsigned int q2_lo = 0;
                            unsigned int q3_hi = 0;
                            unsigned int q3_lo = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0[0])), "r"((uint32_t)(lut0[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0[0])), "r"((uint32_t)(lut0[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0_hi = _nv_hi;
                                q0_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad[1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0[0])), "r"((uint32_t)(lut0[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0[0])), "r"((uint32_t)(lut0[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1_hi = _nv_hi;
                                q1_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad[2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1[0])), "r"((uint32_t)(lut1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1[0])), "r"((uint32_t)(lut1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q2_hi = _nv_hi;
                                q2_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad[3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1[0])), "r"((uint32_t)(lut1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1[0])), "r"((uint32_t)(lut1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q3_hi = _nv_hi;
                                q3_lo = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx * 128 + quad_in_half * 2 * 16 ^ (epilogue_thread_idx * 128 + quad_in_half * 2 * 16 >> 7 & 7) << 4))), "r"(q0_hi), "r"(q0_lo), "r"(q1_hi), "r"(q1_lo) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx * 128 + (quad_in_half * 2 + 1) * 16 ^ (epilogue_thread_idx * 128 + (quad_in_half * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q2_hi), "r"(q2_lo), "r"(q3_hi), "r"(q3_lo) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (warp < 8) {
                            asm volatile("bar.sync 3, 128;" ::: "memory");
                        } else {
                            asm volatile("bar.sync 4, 128;" ::: "memory");
                        }
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_0 = (((uint64_t)(((l1_act_smem_addr + l1_math_stage * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                        uint64_t _wgmma_desc_1 = (((uint64_t)(((l1_decoded_smem_addr + l1_math_stage * 32768 + (warp - 4) / 4 * 128 * 128)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int quad_in_half_1 = 0; quad_in_half_1 < 2; quad_in_half_1++) {
                            unsigned int packed_quad_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_1[(0) + 3]))
                                : "r"(packed_row_addr + (2 + quad_in_half_1) * 16));
                            unsigned int scale_word_1 = scale_words[0];
                            if (2 + quad_in_half_1 >= 2) {
                                scale_word_1 = scale_words[1];
                            }
                            unsigned int scale0_1 = scale_word_1 >> (unsigned int)(((2 + quad_in_half_1) * 2 & 3) * 8) & 127;
                            unsigned int scale1_1 = scale_word_1 >> (unsigned int)(((2 + quad_in_half_1) * 2 + 1 & 3) * 8) & 127;
                            unsigned int lut0_1[2];
                            unsigned int lut1_1[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut0_1[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale0_1 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut1_1[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale1_1 * 8)));
                            unsigned int q0_hi_1 = 0;
                            unsigned int q0_lo_1 = 0;
                            unsigned int q1_hi_1 = 0;
                            unsigned int q1_lo_1 = 0;
                            unsigned int q2_hi_1 = 0;
                            unsigned int q2_lo_1 = 0;
                            unsigned int q3_hi_1 = 0;
                            unsigned int q3_lo_1 = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_1[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_1[0])), "r"((uint32_t)(lut0_1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_1[0])), "r"((uint32_t)(lut0_1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0_hi_1 = _nv_hi;
                                q0_lo_1 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_1[1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_1[0])), "r"((uint32_t)(lut0_1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_1[0])), "r"((uint32_t)(lut0_1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1_hi_1 = _nv_hi;
                                q1_lo_1 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_1[2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_1[0])), "r"((uint32_t)(lut1_1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_1[0])), "r"((uint32_t)(lut1_1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q2_hi_1 = _nv_hi;
                                q2_lo_1 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_1[3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_1[0])), "r"((uint32_t)(lut1_1[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_1[0])), "r"((uint32_t)(lut1_1[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q3_hi_1 = _nv_hi;
                                q3_lo_1 = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx * 128 + (2 + quad_in_half_1) * 2 * 16 ^ (epilogue_thread_idx * 128 + (2 + quad_in_half_1) * 2 * 16 >> 7 & 7) << 4))), "r"(q0_hi_1), "r"(q0_lo_1), "r"(q1_hi_1), "r"(q1_lo_1) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx * 128 + ((2 + quad_in_half_1) * 2 + 1) * 16 ^ (epilogue_thread_idx * 128 + ((2 + quad_in_half_1) * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q2_hi_1), "r"(q2_lo_1), "r"(q3_hi_1), "r"(q3_lo_1) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (warp < 8) {
                            asm volatile("bar.sync 3, 128;" ::: "memory");
                        } else {
                            asm volatile("bar.sync 4, 128;" ::: "memory");
                        }
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int accum_idx_1 = 0; accum_idx_1 < 64; accum_idx_1++) {
                            asm volatile("" : "+f"(accum[accum_idx_1]) :: "memory");
                        }
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(l1_stage_empty_addr + (l1_math_stage) * 8);
                        }
                        #pragma unroll
                        for (int accum_chunk = 0; accum_chunk < 16; accum_chunk++) {
                            final_accum[accum_chunk * 4] = final_accum[accum_chunk * 4] + scale_0[0] * accum[accum_chunk * 4];
                            final_accum[accum_chunk * 4 + 1] = final_accum[accum_chunk * 4 + 1] + scale_0[0] * accum[accum_chunk * 4 + 1];
                            final_accum[accum_chunk * 4 + 2] = final_accum[accum_chunk * 4 + 2] + scale_1[0] * accum[accum_chunk * 4 + 2];
                            final_accum[accum_chunk * 4 + 3] = final_accum[accum_chunk * 4 + 3] + scale_1[0] * accum[accum_chunk * 4 + 3];
                        }
                        l1_math_stage += 1;
                        if (l1_math_stage == 3) { l1_math_stage = 0; _phase_l1_stage_full ^= 1; }
                    }
                    int epilogue_warp_idx_1 = warp - 4;
                    int epilogue_wg_idx = epilogue_warp_idx_1 / 4;
                    int warp_idx_in_wg_1 = epilogue_warp_idx_1 % 4;
                    int row_idx_1 = lane / 4;
                    int col_idx = lane % 4;
                    int r_0_1 = warp_idx_in_wg_1 * 16 + row_idx_1;
                    int r_1_1 = r_0_1 + 8;
                    int wg_l1_out_n_idx = epilogue_wg_idx * 64;
                    int m_idx = (int)math_task_words[4] * 64;
                    float _vec_load_9[1];
                    {
                        uint32_t _scalar_bits_2;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_2) : "l"((const void*)(l1_global_scales + ((int)math_task_words[1]))) : "memory");
                        _vec_load_9[0] = __uint_as_float(_scalar_bits_2);
                    }
                    float l1_global_scale = _vec_load_9[0];
                    float weight_0 = 0.0f;
                    float weight_1 = 0.0f;
                    if (r_0_1 < math_task_valid_m) {
                        float _vec_load_10[1];
                        {
                            _vec_load_10[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 2640225664) + m_idx + r_0_1);
                        }
                        weight_0 = _vec_load_10[0];
                    }
                    if (r_1_1 < math_task_valid_m) {
                        float _vec_load_11[1];
                        {
                            _vec_load_11[0] = *reinterpret_cast<const float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 2640225664) + m_idx + r_1_1);
                        }
                        weight_1 = _vec_load_11[0];
                    }
                    float swiglu_0[16];
                    float swiglu_1[16];
                    swiglu_0[0] = 0.0f;
                    swiglu_0[1] = 0.0f;
                    swiglu_0[2] = 0.0f;
                    swiglu_0[3] = 0.0f;
                    swiglu_0[4] = 0.0f;
                    swiglu_0[5] = 0.0f;
                    swiglu_0[6] = 0.0f;
                    swiglu_0[7] = 0.0f;
                    swiglu_0[8] = 0.0f;
                    swiglu_0[9] = 0.0f;
                    swiglu_0[10] = 0.0f;
                    swiglu_0[11] = 0.0f;
                    swiglu_0[12] = 0.0f;
                    swiglu_0[13] = 0.0f;
                    swiglu_0[14] = 0.0f;
                    swiglu_0[15] = 0.0f;
                    swiglu_1[0] = 0.0f;
                    swiglu_1[1] = 0.0f;
                    swiglu_1[2] = 0.0f;
                    swiglu_1[3] = 0.0f;
                    swiglu_1[4] = 0.0f;
                    swiglu_1[5] = 0.0f;
                    swiglu_1[6] = 0.0f;
                    swiglu_1[7] = 0.0f;
                    swiglu_1[8] = 0.0f;
                    swiglu_1[9] = 0.0f;
                    swiglu_1[10] = 0.0f;
                    swiglu_1[11] = 0.0f;
                    swiglu_1[12] = 0.0f;
                    swiglu_1[13] = 0.0f;
                    swiglu_1[14] = 0.0f;
                    swiglu_1[15] = 0.0f;
                    float amax_0 = 0.0f;
                    float amax_1 = 0.0f;
                    #pragma unroll
                    for (int pair_idx = 0; pair_idx < 8; pair_idx++) {
                        if (r_0_1 < math_task_valid_m) {
                            float _min_2 = fminf(final_accum[pair_idx * 8] * l1_global_scale, 10.0f);
                            float gate_00 = _min_2;
                            float _min_3 = fminf(final_accum[pair_idx * 8 + 1] * l1_global_scale, 10.0f);
                            float gate_01 = _min_3;
                            float _min_4 = fminf(final_accum[pair_idx * 8 + 4] * l1_global_scale, 10.0f);
                            float _max_0 = max_noftz(_min_4, -10.0f);
                            float up_00 = _max_0;
                            float _min_5 = fminf(final_accum[pair_idx * 8 + 4 + 1] * l1_global_scale, 10.0f);
                            float _max_1 = max_noftz(_min_5, -10.0f);
                            float up_01 = _max_1;
                            float _expf_0 = __expf(-gate_00);
                            float _rcp_0 = approx_rcp(1.0f + _expf_0);
                            float value_00 = gate_00 * _rcp_0 * up_00;
                            float _expf_1 = __expf(-gate_01);
                            float _rcp_1 = approx_rcp(1.0f + _expf_1);
                            float value_01 = gate_01 * _rcp_1 * up_01;
                            swiglu_0[pair_idx * 2] = value_00;
                            swiglu_0[pair_idx * 2 + 1] = value_01;
                            float _fabs_0 = fabsf(value_00);
                            float _fabs_1 = fabsf(value_01);
                            float _max_2 = max_noftz(_fabs_0, _fabs_1);
                            float _max_3 = max_noftz(amax_0, _max_2);
                            amax_0 = _max_3;
                        }
                        if (r_1_1 < math_task_valid_m) {
                            float _min_6 = fminf(final_accum[pair_idx * 8 + 2] * l1_global_scale, 10.0f);
                            float gate_10 = _min_6;
                            float _min_7 = fminf(final_accum[pair_idx * 8 + 3] * l1_global_scale, 10.0f);
                            float gate_11 = _min_7;
                            float _min_8 = fminf(final_accum[pair_idx * 8 + 4 + 2] * l1_global_scale, 10.0f);
                            float _max_4 = max_noftz(_min_8, -10.0f);
                            float up_10 = _max_4;
                            float _min_9 = fminf(final_accum[pair_idx * 8 + 4 + 3] * l1_global_scale, 10.0f);
                            float _max_5 = max_noftz(_min_9, -10.0f);
                            float up_11 = _max_5;
                            float _expf_2 = __expf(-gate_10);
                            float _rcp_2 = approx_rcp(1.0f + _expf_2);
                            float value_10 = gate_10 * _rcp_2 * up_10;
                            float _expf_3 = __expf(-gate_11);
                            float _rcp_3 = approx_rcp(1.0f + _expf_3);
                            float value_11 = gate_11 * _rcp_3 * up_11;
                            swiglu_1[pair_idx * 2] = value_10;
                            swiglu_1[pair_idx * 2 + 1] = value_11;
                            float _fabs_2 = fabsf(value_10);
                            float _fabs_3 = fabsf(value_11);
                            float _max_6 = max_noftz(_fabs_2, _fabs_3);
                            float _max_7 = max_noftz(amax_1, _max_6);
                            amax_1 = _max_7;
                        }
                    }
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 16; value_idx++) {
                        swiglu_0[value_idx] = swiglu_0[value_idx] * weight_0;
                        swiglu_1[value_idx] = swiglu_1[value_idx] * weight_1;
                    }
                    float _fabs_4 = fabsf(weight_0);
                    amax_0 = amax_0 * _fabs_4;
                    float _fabs_5 = fabsf(weight_1);
                    amax_1 = amax_1 * _fabs_5;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, amax_0, 1);
                    float _max_8 = max_noftz(amax_0, _shfl_xor_0);
                    amax_0 = _max_8;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, amax_1, 1);
                    float _max_9 = max_noftz(amax_1, _shfl_xor_1);
                    amax_1 = _max_9;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, amax_0, 2);
                    float _max_10 = max_noftz(amax_0, _shfl_xor_2);
                    amax_0 = _max_10;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, amax_1, 2);
                    float _max_11 = max_noftz(amax_1, _shfl_xor_3);
                    amax_1 = _max_11;
                    if (col_idx == 0) {
                        l1_amax_smem[epilogue_wg_idx * 64 + r_0_1] = amax_0;
                        l1_amax_smem[epilogue_wg_idx * 64 + r_1_1] = amax_1;
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    float _max_12 = max_noftz(l1_amax_smem[r_0_1], l1_amax_smem[64 + r_0_1]);
                    amax_0 = _max_12;
                    float _max_13 = max_noftz(l1_amax_smem[r_1_1], l1_amax_smem[64 + r_1_1]);
                    amax_1 = _max_13;
                    float scaled_amax_0 = amax_0 * 0.002232142857142857f;
                    float scaled_amax_1 = amax_1 * 0.002232142857142857f;
                    unsigned int scaled_bits_0 = 0;
                    unsigned int scaled_bits_1 = 0;
                    scaled_bits_0 = reinterpret_cast<unsigned int*>(&scaled_amax_0)[0];
                    scaled_bits_1 = reinterpret_cast<unsigned int*>(&scaled_amax_1)[0];
                    int exponent_0 = (int)(scaled_bits_0 >> 23);
                    int exponent_1 = (int)(scaled_bits_1 >> 23);
                    int scale_exponent_0 = exponent_0 - 127;
                    int scale_exponent_1 = exponent_1 - 127;
                    if ((scaled_bits_0 & 8388607) != 0) {
                        scale_exponent_0 = scale_exponent_0 + 1;
                    }
                    if ((scaled_bits_1 & 8388607) != 0) {
                        scale_exponent_1 = scale_exponent_1 + 1;
                    }
                    unsigned int scale_bits_0 = (unsigned int)(scale_exponent_0 + 127) << 23;
                    unsigned int scale_bits_1 = (unsigned int)(scale_exponent_1 + 127) << 23;
                    unsigned int inv_bits_0 = (unsigned int)(-scale_exponent_0 + 127) << 23;
                    unsigned int inv_bits_1 = (unsigned int)(-scale_exponent_1 + 127) << 23;
                    float scale_0_1 = 0.0f;
                    float scale_1_1 = 0.0f;
                    float scale_inv_0 = 0.0f;
                    float scale_inv_1 = 0.0f;
                    scale_0_1 = reinterpret_cast<float*>(&scale_bits_0)[0];
                    scale_1_1 = reinterpret_cast<float*>(&scale_bits_1)[0];
                    scale_inv_0 = reinterpret_cast<float*>(&inv_bits_0)[0];
                    scale_inv_1 = reinterpret_cast<float*>(&inv_bits_1)[0];
                    if (epilogue_wg_idx == 0 && col_idx == 0) {
                        int sf_base = (int)math_task_words[3] * 6586368 + m_idx;
                        if (r_0_1 < math_task_valid_m) {
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3484927360)) + (sf_base + r_0_1)) = scale_0_1;
                        }
                        if (r_1_1 < math_task_valid_m) {
                            *(reinterpret_cast<float*>(reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3484927360)) + (sf_base + r_1_1)) = scale_1_1;
                        }
                    }
                    #pragma unroll
                    for (int pair_idx_1 = 0; pair_idx_1 < 8; pair_idx_1++) {
                        float quant_values[4];
                        quant_values[0] = swiglu_0[pair_idx_1 * 2] * scale_inv_0;
                        quant_values[1] = swiglu_0[pair_idx_1 * 2 + 1] * scale_inv_0;
                        quant_values[2] = swiglu_1[pair_idx_1 * 2] * scale_inv_1;
                        quant_values[3] = swiglu_1[pair_idx_1 * 2 + 1] * scale_inv_1;
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
                                : "=r"(_packed) : "f"(quant_values[0]), "f"(quant_values[1]),
                                                   "f"(quant_values[2]), "f"(quant_values[3]));
                            _fp8_0[0] = _packed;
                        }
                        int out_col = wg_l1_out_n_idx + pair_idx_1 * 8 + col_idx * 2;
                        if (r_0_1 < math_task_valid_m) {
                            l1_output_smem[r_0_1 * 128 + out_col] = _fp8_0[0] & 255;
                            l1_output_smem[r_0_1 * 128 + out_col + 1] = _fp8_0[0] >> 8 & 255;
                        }
                        if (r_1_1 < math_task_valid_m) {
                            l1_output_smem[r_1_1 * 128 + out_col] = _fp8_0[0] >> 16 & 255;
                            l1_output_smem[r_1_1 * 128 + out_col + 1] = _fp8_0[0] >> 24 & 255;
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
                            asm volatile("red.release.gpu.global.or.b64 [%0], %1;" :: "l"(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 210304)[(int)math_task_words[4]]), "l"((unsigned long long)(ready_bit)) : "memory");
                        }
                    }
                    __syncwarp();
                } else {
                    int math_task_valid_m_1 = (int)math_task_words[5];
                    int math_task_shape_k_1 = (int)math_task_words[7];
                    float final_accum_1[64];
                    final_accum_1[0] = 0.0f;
                    final_accum_1[1] = 0.0f;
                    final_accum_1[2] = 0.0f;
                    final_accum_1[3] = 0.0f;
                    final_accum_1[4] = 0.0f;
                    final_accum_1[5] = 0.0f;
                    final_accum_1[6] = 0.0f;
                    final_accum_1[7] = 0.0f;
                    final_accum_1[8] = 0.0f;
                    final_accum_1[9] = 0.0f;
                    final_accum_1[10] = 0.0f;
                    final_accum_1[11] = 0.0f;
                    final_accum_1[12] = 0.0f;
                    final_accum_1[13] = 0.0f;
                    final_accum_1[14] = 0.0f;
                    final_accum_1[15] = 0.0f;
                    final_accum_1[16] = 0.0f;
                    final_accum_1[17] = 0.0f;
                    final_accum_1[18] = 0.0f;
                    final_accum_1[19] = 0.0f;
                    final_accum_1[20] = 0.0f;
                    final_accum_1[21] = 0.0f;
                    final_accum_1[22] = 0.0f;
                    final_accum_1[23] = 0.0f;
                    final_accum_1[24] = 0.0f;
                    final_accum_1[25] = 0.0f;
                    final_accum_1[26] = 0.0f;
                    final_accum_1[27] = 0.0f;
                    final_accum_1[28] = 0.0f;
                    final_accum_1[29] = 0.0f;
                    final_accum_1[30] = 0.0f;
                    final_accum_1[31] = 0.0f;
                    final_accum_1[32] = 0.0f;
                    final_accum_1[33] = 0.0f;
                    final_accum_1[34] = 0.0f;
                    final_accum_1[35] = 0.0f;
                    final_accum_1[36] = 0.0f;
                    final_accum_1[37] = 0.0f;
                    final_accum_1[38] = 0.0f;
                    final_accum_1[39] = 0.0f;
                    final_accum_1[40] = 0.0f;
                    final_accum_1[41] = 0.0f;
                    final_accum_1[42] = 0.0f;
                    final_accum_1[43] = 0.0f;
                    final_accum_1[44] = 0.0f;
                    final_accum_1[45] = 0.0f;
                    final_accum_1[46] = 0.0f;
                    final_accum_1[47] = 0.0f;
                    final_accum_1[48] = 0.0f;
                    final_accum_1[49] = 0.0f;
                    final_accum_1[50] = 0.0f;
                    final_accum_1[51] = 0.0f;
                    final_accum_1[52] = 0.0f;
                    final_accum_1[53] = 0.0f;
                    final_accum_1[54] = 0.0f;
                    final_accum_1[55] = 0.0f;
                    final_accum_1[56] = 0.0f;
                    final_accum_1[57] = 0.0f;
                    final_accum_1[58] = 0.0f;
                    final_accum_1[59] = 0.0f;
                    final_accum_1[60] = 0.0f;
                    final_accum_1[61] = 0.0f;
                    final_accum_1[62] = 0.0f;
                    final_accum_1[63] = 0.0f;
                    #pragma unroll 1
                    for (int math_k_block_idx_1 = 0; math_k_block_idx_1 < math_task_shape_k_1 / 128; math_k_block_idx_1++) {
                        mbarrier_wait(l1_stage_full_addr + (l1_math_stage) * 8, _phase_l1_stage_full);
                        if (math_k_block_idx_1 == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(task_info_empty_addr + (consumed_task_stage) * 8);
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        int epilogue_warp_idx_2 = warp - 4;
                        int epilogue_thread_idx_1 = (unsigned int)(epilogue_warp_idx_2 * 32) + lane;
                        int warp_idx_in_wg_2 = epilogue_warp_idx_2 % 4;
                        int row_idx_2 = lane / 4;
                        int r_0_2 = warp_idx_in_wg_2 * 16 + row_idx_2;
                        int r_1_2 = r_0_2 + 8;
                        int packed_row_addr_1 = l1_packed_smem_addr + l1_math_stage * 20480 + (unsigned int)(epilogue_thread_idx_1 * 80);
                        unsigned int scale_words_1[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scale_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words_1[(0) + 1]))
                            : "r"(packed_row_addr_1 + 64));
                        float scale_0_2[1];
                        float scale_1_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_0_2[0])) : "r"(l1_sfa_smem_addr + l1_math_stage * 256 + (unsigned int)(r_0_2 * 4)));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_1_2[0])) : "r"(l1_sfa_smem_addr + l1_math_stage * 256 + (unsigned int)(r_1_2 * 4)));
                        float accum_1[64];
                        accum_1[0] = 0.0f;
                        accum_1[1] = 0.0f;
                        accum_1[2] = 0.0f;
                        accum_1[3] = 0.0f;
                        accum_1[4] = 0.0f;
                        accum_1[5] = 0.0f;
                        accum_1[6] = 0.0f;
                        accum_1[7] = 0.0f;
                        accum_1[8] = 0.0f;
                        accum_1[9] = 0.0f;
                        accum_1[10] = 0.0f;
                        accum_1[11] = 0.0f;
                        accum_1[12] = 0.0f;
                        accum_1[13] = 0.0f;
                        accum_1[14] = 0.0f;
                        accum_1[15] = 0.0f;
                        accum_1[16] = 0.0f;
                        accum_1[17] = 0.0f;
                        accum_1[18] = 0.0f;
                        accum_1[19] = 0.0f;
                        accum_1[20] = 0.0f;
                        accum_1[21] = 0.0f;
                        accum_1[22] = 0.0f;
                        accum_1[23] = 0.0f;
                        accum_1[24] = 0.0f;
                        accum_1[25] = 0.0f;
                        accum_1[26] = 0.0f;
                        accum_1[27] = 0.0f;
                        accum_1[28] = 0.0f;
                        accum_1[29] = 0.0f;
                        accum_1[30] = 0.0f;
                        accum_1[31] = 0.0f;
                        accum_1[32] = 0.0f;
                        accum_1[33] = 0.0f;
                        accum_1[34] = 0.0f;
                        accum_1[35] = 0.0f;
                        accum_1[36] = 0.0f;
                        accum_1[37] = 0.0f;
                        accum_1[38] = 0.0f;
                        accum_1[39] = 0.0f;
                        accum_1[40] = 0.0f;
                        accum_1[41] = 0.0f;
                        accum_1[42] = 0.0f;
                        accum_1[43] = 0.0f;
                        accum_1[44] = 0.0f;
                        accum_1[45] = 0.0f;
                        accum_1[46] = 0.0f;
                        accum_1[47] = 0.0f;
                        accum_1[48] = 0.0f;
                        accum_1[49] = 0.0f;
                        accum_1[50] = 0.0f;
                        accum_1[51] = 0.0f;
                        accum_1[52] = 0.0f;
                        accum_1[53] = 0.0f;
                        accum_1[54] = 0.0f;
                        accum_1[55] = 0.0f;
                        accum_1[56] = 0.0f;
                        accum_1[57] = 0.0f;
                        accum_1[58] = 0.0f;
                        accum_1[59] = 0.0f;
                        accum_1[60] = 0.0f;
                        accum_1[61] = 0.0f;
                        accum_1[62] = 0.0f;
                        accum_1[63] = 0.0f;
                        #pragma unroll
                        for (int accum_idx_2 = 0; accum_idx_2 < 64; accum_idx_2++) {
                            asm volatile("" : "+f"(accum_1[accum_idx_2]) :: "memory");
                        }
                        #pragma unroll
                        for (int quad_in_half_2 = 0; quad_in_half_2 < 2; quad_in_half_2++) {
                            unsigned int packed_quad_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_2[(0) + 3]))
                                : "r"(packed_row_addr_1 + quad_in_half_2 * 16));
                            unsigned int scale_word_2 = scale_words_1[0];
                            if (quad_in_half_2 >= 2) {
                                scale_word_2 = scale_words_1[1];
                            }
                            unsigned int scale0_2 = scale_word_2 >> (unsigned int)((quad_in_half_2 * 2 & 3) * 8) & 127;
                            unsigned int scale1_2 = scale_word_2 >> (unsigned int)((quad_in_half_2 * 2 + 1 & 3) * 8) & 127;
                            unsigned int lut0_2[2];
                            unsigned int lut1_2[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut0_2[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale0_2 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut1_2[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale1_2 * 8)));
                            unsigned int q0_hi_2 = 0;
                            unsigned int q0_lo_2 = 0;
                            unsigned int q1_hi_2 = 0;
                            unsigned int q1_lo_2 = 0;
                            unsigned int q2_hi_2 = 0;
                            unsigned int q2_lo_2 = 0;
                            unsigned int q3_hi_2 = 0;
                            unsigned int q3_lo_2 = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_2[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_2[0])), "r"((uint32_t)(lut0_2[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_2[0])), "r"((uint32_t)(lut0_2[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0_hi_2 = _nv_hi;
                                q0_lo_2 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_2[1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_2[0])), "r"((uint32_t)(lut0_2[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_2[0])), "r"((uint32_t)(lut0_2[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1_hi_2 = _nv_hi;
                                q1_lo_2 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_2[2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_2[0])), "r"((uint32_t)(lut1_2[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_2[0])), "r"((uint32_t)(lut1_2[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q2_hi_2 = _nv_hi;
                                q2_lo_2 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_2[3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_2[0])), "r"((uint32_t)(lut1_2[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_2[0])), "r"((uint32_t)(lut1_2[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q3_hi_2 = _nv_hi;
                                q3_lo_2 = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx_1 * 128 + quad_in_half_2 * 2 * 16 ^ (epilogue_thread_idx_1 * 128 + quad_in_half_2 * 2 * 16 >> 7 & 7) << 4))), "r"(q0_hi_2), "r"(q0_lo_2), "r"(q1_hi_2), "r"(q1_lo_2) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx_1 * 128 + (quad_in_half_2 * 2 + 1) * 16 ^ (epilogue_thread_idx_1 * 128 + (quad_in_half_2 * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q2_hi_2), "r"(q2_lo_2), "r"(q3_hi_2), "r"(q3_lo_2) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (warp < 8) {
                            asm volatile("bar.sync 3, 128;" ::: "memory");
                        } else {
                            asm volatile("bar.sync 4, 128;" ::: "memory");
                        }
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        uint64_t _wgmma_desc_3 = (((uint64_t)(((l1_act_smem_addr + l1_math_stage * 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                        uint64_t _wgmma_desc_4 = (((uint64_t)(((l1_decoded_smem_addr + l1_math_stage * 32768 + (warp - 4) / 4 * 128 * 128)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                        uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1;\n}\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3]), "+f"(accum_1[4]), "+f"(accum_1[5]), "+f"(accum_1[6]), "+f"(accum_1[7]), "+f"(accum_1[8]), "+f"(accum_1[9]), "+f"(accum_1[10]), "+f"(accum_1[11]), "+f"(accum_1[12]), "+f"(accum_1[13]), "+f"(accum_1[14]), "+f"(accum_1[15]), "+f"(accum_1[16]), "+f"(accum_1[17]), "+f"(accum_1[18]), "+f"(accum_1[19]), "+f"(accum_1[20]), "+f"(accum_1[21]), "+f"(accum_1[22]), "+f"(accum_1[23]), "+f"(accum_1[24]), "+f"(accum_1[25]), "+f"(accum_1[26]), "+f"(accum_1[27]), "+f"(accum_1[28]), "+f"(accum_1[29]), "+f"(accum_1[30]), "+f"(accum_1[31]), "+f"(accum_1[32]), "+f"(accum_1[33]), "+f"(accum_1[34]), "+f"(accum_1[35]), "+f"(accum_1[36]), "+f"(accum_1[37]), "+f"(accum_1[38]), "+f"(accum_1[39]), "+f"(accum_1[40]), "+f"(accum_1[41]), "+f"(accum_1[42]), "+f"(accum_1[43]), "+f"(accum_1[44]), "+f"(accum_1[45]), "+f"(accum_1[46]), "+f"(accum_1[47]), "+f"(accum_1[48]), "+f"(accum_1[49]), "+f"(accum_1[50]), "+f"(accum_1[51]), "+f"(accum_1[52]), "+f"(accum_1[53]), "+f"(accum_1[54]), "+f"(accum_1[55]), "+f"(accum_1[56]), "+f"(accum_1[57]), "+f"(accum_1[58]), "+f"(accum_1[59]), "+f"(accum_1[60]), "+f"(accum_1[61]), "+f"(accum_1[62]), "+f"(accum_1[63])
                            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_3)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3]), "+f"(accum_1[4]), "+f"(accum_1[5]), "+f"(accum_1[6]), "+f"(accum_1[7]), "+f"(accum_1[8]), "+f"(accum_1[9]), "+f"(accum_1[10]), "+f"(accum_1[11]), "+f"(accum_1[12]), "+f"(accum_1[13]), "+f"(accum_1[14]), "+f"(accum_1[15]), "+f"(accum_1[16]), "+f"(accum_1[17]), "+f"(accum_1[18]), "+f"(accum_1[19]), "+f"(accum_1[20]), "+f"(accum_1[21]), "+f"(accum_1[22]), "+f"(accum_1[23]), "+f"(accum_1[24]), "+f"(accum_1[25]), "+f"(accum_1[26]), "+f"(accum_1[27]), "+f"(accum_1[28]), "+f"(accum_1[29]), "+f"(accum_1[30]), "+f"(accum_1[31]), "+f"(accum_1[32]), "+f"(accum_1[33]), "+f"(accum_1[34]), "+f"(accum_1[35]), "+f"(accum_1[36]), "+f"(accum_1[37]), "+f"(accum_1[38]), "+f"(accum_1[39]), "+f"(accum_1[40]), "+f"(accum_1[41]), "+f"(accum_1[42]), "+f"(accum_1[43]), "+f"(accum_1[44]), "+f"(accum_1[45]), "+f"(accum_1[46]), "+f"(accum_1[47]), "+f"(accum_1[48]), "+f"(accum_1[49]), "+f"(accum_1[50]), "+f"(accum_1[51]), "+f"(accum_1[52]), "+f"(accum_1[53]), "+f"(accum_1[54]), "+f"(accum_1[55]), "+f"(accum_1[56]), "+f"(accum_1[57]), "+f"(accum_1[58]), "+f"(accum_1[59]), "+f"(accum_1[60]), "+f"(accum_1[61]), "+f"(accum_1[62]), "+f"(accum_1[63])
                            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_3 + 2)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int quad_in_half_3 = 0; quad_in_half_3 < 2; quad_in_half_3++) {
                            unsigned int packed_quad_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_quad_3[(0) + 3]))
                                : "r"(packed_row_addr_1 + (2 + quad_in_half_3) * 16));
                            unsigned int scale_word_3 = scale_words_1[0];
                            if (2 + quad_in_half_3 >= 2) {
                                scale_word_3 = scale_words_1[1];
                            }
                            unsigned int scale0_3 = scale_word_3 >> (unsigned int)(((2 + quad_in_half_3) * 2 & 3) * 8) & 127;
                            unsigned int scale1_3 = scale_word_3 >> (unsigned int)(((2 + quad_in_half_3) * 2 + 1 & 3) * 8) & 127;
                            unsigned int lut0_3[2];
                            unsigned int lut1_3[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut0_3[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale0_3 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut1_3[(0) + 1]))
                                : "r"(l1_lut_smem_addr + (unsigned int)((int)scale1_3 * 8)));
                            unsigned int q0_hi_3 = 0;
                            unsigned int q0_lo_3 = 0;
                            unsigned int q1_hi_3 = 0;
                            unsigned int q1_lo_3 = 0;
                            unsigned int q2_hi_3 = 0;
                            unsigned int q2_lo_3 = 0;
                            unsigned int q3_hi_3 = 0;
                            unsigned int q3_lo_3 = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_3[0]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_3[0])), "r"((uint32_t)(lut0_3[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_3[0])), "r"((uint32_t)(lut0_3[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0_hi_3 = _nv_hi;
                                q0_lo_3 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_3[1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut0_3[0])), "r"((uint32_t)(lut0_3[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut0_3[0])), "r"((uint32_t)(lut0_3[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1_hi_3 = _nv_hi;
                                q1_lo_3 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_3[2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_3[0])), "r"((uint32_t)(lut1_3[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_3[0])), "r"((uint32_t)(lut1_3[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q2_hi_3 = _nv_hi;
                                q2_lo_3 = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(packed_quad_3[3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut1_3[0])), "r"((uint32_t)(lut1_3[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut1_3[0])), "r"((uint32_t)(lut1_3[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q3_hi_3 = _nv_hi;
                                q3_lo_3 = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx_1 * 128 + (2 + quad_in_half_3) * 2 * 16 ^ (epilogue_thread_idx_1 * 128 + (2 + quad_in_half_3) * 2 * 16 >> 7 & 7) << 4))), "r"(q0_hi_3), "r"(q0_lo_3), "r"(q1_hi_3), "r"(q1_lo_3) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((l1_decoded_smem_addr + l1_math_stage * 32768 + (unsigned int)(epilogue_thread_idx_1 * 128 + ((2 + quad_in_half_3) * 2 + 1) * 16 ^ (epilogue_thread_idx_1 * 128 + ((2 + quad_in_half_3) * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q2_hi_3), "r"(q2_lo_3), "r"(q3_hi_3), "r"(q3_lo_3) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (warp < 8) {
                            asm volatile("bar.sync 3, 128;" ::: "memory");
                        } else {
                            asm volatile("bar.sync 4, 128;" ::: "memory");
                        }
                        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3]), "+f"(accum_1[4]), "+f"(accum_1[5]), "+f"(accum_1[6]), "+f"(accum_1[7]), "+f"(accum_1[8]), "+f"(accum_1[9]), "+f"(accum_1[10]), "+f"(accum_1[11]), "+f"(accum_1[12]), "+f"(accum_1[13]), "+f"(accum_1[14]), "+f"(accum_1[15]), "+f"(accum_1[16]), "+f"(accum_1[17]), "+f"(accum_1[18]), "+f"(accum_1[19]), "+f"(accum_1[20]), "+f"(accum_1[21]), "+f"(accum_1[22]), "+f"(accum_1[23]), "+f"(accum_1[24]), "+f"(accum_1[25]), "+f"(accum_1[26]), "+f"(accum_1[27]), "+f"(accum_1[28]), "+f"(accum_1[29]), "+f"(accum_1[30]), "+f"(accum_1[31]), "+f"(accum_1[32]), "+f"(accum_1[33]), "+f"(accum_1[34]), "+f"(accum_1[35]), "+f"(accum_1[36]), "+f"(accum_1[37]), "+f"(accum_1[38]), "+f"(accum_1[39]), "+f"(accum_1[40]), "+f"(accum_1[41]), "+f"(accum_1[42]), "+f"(accum_1[43]), "+f"(accum_1[44]), "+f"(accum_1[45]), "+f"(accum_1[46]), "+f"(accum_1[47]), "+f"(accum_1[48]), "+f"(accum_1[49]), "+f"(accum_1[50]), "+f"(accum_1[51]), "+f"(accum_1[52]), "+f"(accum_1[53]), "+f"(accum_1[54]), "+f"(accum_1[55]), "+f"(accum_1[56]), "+f"(accum_1[57]), "+f"(accum_1[58]), "+f"(accum_1[59]), "+f"(accum_1[60]), "+f"(accum_1[61]), "+f"(accum_1[62]), "+f"(accum_1[63])
                            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_3 + 4)
                            : "memory");
                        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                            : "+f"(accum_1[0]), "+f"(accum_1[1]), "+f"(accum_1[2]), "+f"(accum_1[3]), "+f"(accum_1[4]), "+f"(accum_1[5]), "+f"(accum_1[6]), "+f"(accum_1[7]), "+f"(accum_1[8]), "+f"(accum_1[9]), "+f"(accum_1[10]), "+f"(accum_1[11]), "+f"(accum_1[12]), "+f"(accum_1[13]), "+f"(accum_1[14]), "+f"(accum_1[15]), "+f"(accum_1[16]), "+f"(accum_1[17]), "+f"(accum_1[18]), "+f"(accum_1[19]), "+f"(accum_1[20]), "+f"(accum_1[21]), "+f"(accum_1[22]), "+f"(accum_1[23]), "+f"(accum_1[24]), "+f"(accum_1[25]), "+f"(accum_1[26]), "+f"(accum_1[27]), "+f"(accum_1[28]), "+f"(accum_1[29]), "+f"(accum_1[30]), "+f"(accum_1[31]), "+f"(accum_1[32]), "+f"(accum_1[33]), "+f"(accum_1[34]), "+f"(accum_1[35]), "+f"(accum_1[36]), "+f"(accum_1[37]), "+f"(accum_1[38]), "+f"(accum_1[39]), "+f"(accum_1[40]), "+f"(accum_1[41]), "+f"(accum_1[42]), "+f"(accum_1[43]), "+f"(accum_1[44]), "+f"(accum_1[45]), "+f"(accum_1[46]), "+f"(accum_1[47]), "+f"(accum_1[48]), "+f"(accum_1[49]), "+f"(accum_1[50]), "+f"(accum_1[51]), "+f"(accum_1[52]), "+f"(accum_1[53]), "+f"(accum_1[54]), "+f"(accum_1[55]), "+f"(accum_1[56]), "+f"(accum_1[57]), "+f"(accum_1[58]), "+f"(accum_1[59]), "+f"(accum_1[60]), "+f"(accum_1[61]), "+f"(accum_1[62]), "+f"(accum_1[63])
                            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_3 + 6)
                            : "memory");
                        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int accum_idx_3 = 0; accum_idx_3 < 64; accum_idx_3++) {
                            asm volatile("" : "+f"(accum_1[accum_idx_3]) :: "memory");
                        }
                        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(l1_stage_empty_addr + (l1_math_stage) * 8);
                        }
                        #pragma unroll
                        for (int accum_chunk_1 = 0; accum_chunk_1 < 16; accum_chunk_1++) {
                            final_accum_1[accum_chunk_1 * 4] = final_accum_1[accum_chunk_1 * 4] + scale_0_2[0] * accum_1[accum_chunk_1 * 4];
                            final_accum_1[accum_chunk_1 * 4 + 1] = final_accum_1[accum_chunk_1 * 4 + 1] + scale_0_2[0] * accum_1[accum_chunk_1 * 4 + 1];
                            final_accum_1[accum_chunk_1 * 4 + 2] = final_accum_1[accum_chunk_1 * 4 + 2] + scale_1_2[0] * accum_1[accum_chunk_1 * 4 + 2];
                            final_accum_1[accum_chunk_1 * 4 + 3] = final_accum_1[accum_chunk_1 * 4 + 3] + scale_1_2[0] * accum_1[accum_chunk_1 * 4 + 3];
                        }
                        l1_math_stage += 1;
                        if (l1_math_stage == 3) { l1_math_stage = 0; _phase_l1_stage_full ^= 1; }
                    }
                    float _vec_load_12[1];
                    {
                        uint32_t _scalar_bits_5;
                        asm volatile("ld.global.nc.b32 %0, [%1];"
                            : "=r"(_scalar_bits_5) : "l"((const void*)(l2_global_scales + ((int)math_task_words[1]))) : "memory");
                        _vec_load_12[0] = __uint_as_float(_scalar_bits_5);
                    }
                    float l2_global_scale = _vec_load_12[0];
                    int epilogue_warp_idx_3 = warp - 4;
                    int epilogue_wg_idx_1 = epilogue_warp_idx_3 / 4;
                    int warp_idx_in_wg_3 = epilogue_warp_idx_3 % 4;
                    int row_idx_3 = lane / 4;
                    int col_idx_1 = lane % 4;
                    int row_offset_r0 = warp_idx_in_wg_3 * 16 + row_idx_3;
                    int row_offset_r1 = row_offset_r0 + 8;
                    int n_idx = (int)math_task_words[3] * 256 + epilogue_wg_idx_1 * 128;
                    if (row_offset_r0 < math_task_valid_m_1) {
                        int metadata_base_r0 = ((int)math_task_words[4] * 64 + row_offset_r0) * 3;
                        int _vec_load_13[1];
                        {
                            _vec_load_13[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r0);
                        }
                        int _vec_load_14[1];
                        {
                            _vec_load_14[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r0 + 1);
                        }
                        int _vec_load_15[1];
                        {
                            _vec_load_15[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r0 + 2);
                        }
                        unsigned long long dst_word_base_r0 = (((unsigned long long)_vec_load_15[0] * 8448 + (unsigned long long)_vec_load_14[0]) * 4096 + (unsigned long long)n_idx) / 2;
                        #pragma unroll
                        for (int pair_idx_2 = 0; pair_idx_2 < 8; pair_idx_2++) {
                            float pair_values_r0_lo[2];
                            pair_values_r0_lo[0] = final_accum_1[pair_idx_2 * 2 * 4] * l2_global_scale;
                            pair_values_r0_lo[1] = final_accum_1[pair_idx_2 * 2 * 4 + 1] * l2_global_scale;
                            float pair_values_r0_hi[2];
                            pair_values_r0_hi[0] = final_accum_1[(pair_idx_2 * 2 + 1) * 4] * l2_global_scale;
                            pair_values_r0_hi[1] = final_accum_1[(pair_idx_2 * 2 + 1) * 4 + 1] * l2_global_scale;
                            unsigned int packed_r0_lo[1];
                            unsigned int packed_r0_hi[1];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values_r0_lo[_lp*2 + 0], pair_values_r0_lo[_lp*2+1 + 0]));
                                packed_r0_lo[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values_r0_hi[_lp*2 + 0], pair_values_r0_hi[_lp*2+1 + 0]));
                                packed_r0_hi[_lp] = *(uint32_t*)&_bf2;
                            }
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[_vec_load_13[0]]) + 4327982464)) + (dst_word_base_r0 + (unsigned long long)((pair_idx_2 * 2 * 8 + col_idx_1 * 2) / 2))) = (int)packed_r0_lo[0];
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[_vec_load_13[0]]) + 4327982464)) + (dst_word_base_r0 + (unsigned long long)(((pair_idx_2 * 2 + 1) * 8 + col_idx_1 * 2) / 2))) = (int)packed_r0_hi[0];
                        }
                    }
                    if (row_offset_r1 < math_task_valid_m_1) {
                        int metadata_base_r1 = ((int)math_task_words[4] * 64 + row_offset_r1) * 3;
                        int _vec_load_16[1];
                        {
                            _vec_load_16[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r1);
                        }
                        int _vec_load_17[1];
                        {
                            _vec_load_17[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r1 + 1);
                        }
                        int _vec_load_18[1];
                        {
                            _vec_load_18[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 69827968) + metadata_base_r1 + 2);
                        }
                        unsigned long long dst_word_base_r1 = (((unsigned long long)_vec_load_18[0] * 8448 + (unsigned long long)_vec_load_17[0]) * 4096 + (unsigned long long)n_idx) / 2;
                        #pragma unroll
                        for (int pair_idx_3 = 0; pair_idx_3 < 8; pair_idx_3++) {
                            float pair_values_r1_lo[2];
                            pair_values_r1_lo[0] = final_accum_1[pair_idx_3 * 2 * 4 + 2] * l2_global_scale;
                            pair_values_r1_lo[1] = final_accum_1[pair_idx_3 * 2 * 4 + 3] * l2_global_scale;
                            float pair_values_r1_hi[2];
                            pair_values_r1_hi[0] = final_accum_1[(pair_idx_3 * 2 + 1) * 4 + 2] * l2_global_scale;
                            pair_values_r1_hi[1] = final_accum_1[(pair_idx_3 * 2 + 1) * 4 + 3] * l2_global_scale;
                            unsigned int packed_r1_lo[1];
                            unsigned int packed_r1_hi[1];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values_r1_lo[_lp*2 + 0], pair_values_r1_lo[_lp*2+1 + 0]));
                                packed_r1_lo[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_values_r1_hi[_lp*2 + 0], pair_values_r1_hi[_lp*2+1 + 0]));
                                packed_r1_hi[_lp] = *(uint32_t*)&_bf2;
                            }
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[_vec_load_16[0]]) + 4327982464)) + (dst_word_base_r1 + (unsigned long long)((pair_idx_3 * 2 * 8 + col_idx_1 * 2) / 2))) = (int)packed_r1_lo[0];
                            *(reinterpret_cast<int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[_vec_load_16[0]]) + 4327982464)) + (dst_word_base_r1 + (unsigned long long)(((pair_idx_3 * 2 + 1) * 8 + col_idx_1 * 2) / 2))) = (int)packed_r1_hi[0];
                        }
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_6 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
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
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (bid == 0 && warp == 4) {
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
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_8 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
                        const unsigned int _gs_add_8 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_8;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_8) : "l"(_gs_ptr_8), "r"(_gs_add_8) : "memory");
                        unsigned int _gs_new_8;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_8) : "l"(_gs_ptr_8) : "memory");
                        } while (((_gs_new_8 ^ _gs_old_8) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 320;" ::: "memory");
            int epilogue_warp_idx_4 = warp - 4;
            unsigned int combine_phase = 0;
            unsigned int combine_load_stage = 0;
            #pragma unroll 1
            for (unsigned int combine_token_idx = bid * 8 + epilogue_warp_idx_4; combine_token_idx < num_tokens; combine_token_idx += 624) {
                int stored_topk_slot_idx = -1;
                if (lane < 6) {
                    long long _vec_load_19[1];
                    {
                        uint64_t _scalar_bits_9;
                        asm volatile("ld.global.nc.b64 %0, [%1];"
                            : "=l"(_scalar_bits_9) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 110452096) + (combine_token_idx * 6 + lane))) : "memory");
                        _vec_load_19[0] = (long long)_scalar_bits_9;
                    }
                    stored_topk_slot_idx = (int)_vec_load_19[0];
                }
                unsigned int _vote_2 = __ballot_sync(4294967295, stored_topk_slot_idx >= 0);
                unsigned int total_mask = _vote_2;
                #pragma unroll
                for (int combine_chunk = 0; combine_chunk < 1; combine_chunk++) {
                    unsigned int chunk_byte_offset = combine_chunk * 8192;
                    unsigned int combine_mask = total_mask;
                    int do_reduce = 0;
                    if (combine_mask != 0) {
                        int _ffs_1 = __ffs(combine_mask);
                        int slot_idx_1 = _ffs_1 - 1;
                        combine_mask = combine_mask ^ (unsigned int)(1 << slot_idx_1);
                        unsigned int load_barrier_stage = (unsigned int)(epilogue_warp_idx_4 * 2) + combine_load_stage;
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(combine_full_addr + (load_barrier_stage) * 8, 8192);
                            cp_async_bulk_gmem2smem(combine_load_smem_addr + load_barrier_stage * 8192, reinterpret_cast<uint8_t*>(sym_buffer) + (4327982464 + ((unsigned long long)slot_idx_1 * 8448 + (unsigned long long)combine_token_idx) * 4096 * 2 + chunk_byte_offset), 8192, combine_full_addr + (load_barrier_stage) * 8);
                        }
                        __syncwarp();
                        do_reduce = 1;
                    }
                    float reduced[128];
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
                    reduced[112] = 0.0f;
                    reduced[113] = 0.0f;
                    reduced[114] = 0.0f;
                    reduced[115] = 0.0f;
                    reduced[116] = 0.0f;
                    reduced[117] = 0.0f;
                    reduced[118] = 0.0f;
                    reduced[119] = 0.0f;
                    reduced[120] = 0.0f;
                    reduced[121] = 0.0f;
                    reduced[122] = 0.0f;
                    reduced[123] = 0.0f;
                    reduced[124] = 0.0f;
                    reduced[125] = 0.0f;
                    reduced[126] = 0.0f;
                    reduced[127] = 0.0f;
                    while (do_reduce != 0) {
                        do_reduce = 0;
                        if (combine_mask != 0) {
                            int _ffs_2 = __ffs(combine_mask);
                            int next_slot_idx = _ffs_2 - 1;
                            combine_mask = combine_mask ^ (unsigned int)(1 << next_slot_idx);
                            unsigned int next_load_stage = combine_load_stage ^ 1;
                            unsigned int next_barrier_stage = (unsigned int)(epilogue_warp_idx_4 * 2) + next_load_stage;
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(combine_full_addr + (next_barrier_stage) * 8, 8192);
                                cp_async_bulk_gmem2smem(combine_load_smem_addr + next_barrier_stage * 8192, reinterpret_cast<uint8_t*>(sym_buffer) + (4327982464 + ((unsigned long long)next_slot_idx * 8448 + (unsigned long long)combine_token_idx) * 4096 * 2 + chunk_byte_offset), 8192, combine_full_addr + (next_barrier_stage) * 8);
                            }
                            __syncwarp();
                            do_reduce = 1;
                        }
                        unsigned int current_barrier_stage = (unsigned int)(epilogue_warp_idx_4 * 2) + combine_load_stage;
                        mbarrier_wait(combine_full_addr + (current_barrier_stage) * 8, combine_phase);
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        #pragma unroll
                        for (int combine_vec = 0; combine_vec < 16; combine_vec++) {
                            unsigned int combine_packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&combine_packed[(0) + 3]))
                                : "r"(combine_load_smem_addr + current_barrier_stage * 8192 + ((unsigned int)(combine_vec * 32) + lane) * 16));
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
                    for (int combine_vec_1 = 0; combine_vec_1 < 16; combine_vec_1++) {
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
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx_4 * 8192) + ((unsigned int)(combine_vec_1 * 32) + lane) * 16 + (unsigned int)(cast_word * 4)), "r"((casted[cast_word])));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            void* _cpbulk_dst_10 = reinterpret_cast<void*>(output_bf16 + ((unsigned long long)combine_token_idx * 4096 + (unsigned long long)(chunk_byte_offset / 2)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_10), "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx_4 * 8192)), "r"((uint32_t)(8192))
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

