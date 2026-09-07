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
#define NUM_PIPE_STAGES 6
#define NUM_COMBINE_PIPE_STAGES 16
#define SMEM_SMEM_EXPERT_COUNT_OFF 1024
#define SMEM_SMEM_EXPERT_COUNT_STAGE_BYTES 1536
#define SMEM_SMEM_EXPERT_COUNT_STRIDE 1536
#define SMEM_LUT_SMEM_OFF 3072
#define SMEM_LUT_SMEM_STAGE_BYTES 1024
#define SMEM_LUT_SMEM_STRIDE 1024
#define SMEM_CD_SMEM_OFF 4096
#define SMEM_CD_SMEM_STAGE_BYTES 17408
#define SMEM_CD_SMEM_STRIDE 17408
#define SMEM_A_SMEM_OFF 21504
#define SMEM_A_SMEM_STAGE_BYTES 16384
#define SMEM_A_SMEM_STRIDE 16384
#define SMEM_B_PACKED_SMEM_OFF 119808
#define SMEM_B_PACKED_SMEM_STAGE_BYTES 10240
#define SMEM_B_PACKED_SMEM_STRIDE 16384
#define SMEM_B_DECODED_SMEM_OFF 119808
#define SMEM_B_DECODED_SMEM_STAGE_BYTES 16384
#define SMEM_B_DECODED_SMEM_STRIDE 16384
#define SMEM_SFA_SMEM_OFF 218112
#define SMEM_SFA_SMEM_STAGE_BYTES 512
#define SMEM_SFA_SMEM_STRIDE 512
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
kernel_split_l2_fragment(int num_tokens, LoomTensorMap const* l2_activation, LoomTensorMap const* l2_activation_scale, LoomTensorMap const* l2_packed_weights, float* __restrict__ l2_global_scales, __nv_bfloat16* __restrict__ output_bf16, int32_t pg_world, int32_t pg_rank, unsigned* const* __restrict__ pg_flags, uint8_t* __restrict__ sym_buffer, uint8_t* const* __restrict__ sym_buffer_peers)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = threadIdx.x % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define stage_full_addr (mbar_base + 0)
    #define stage_empty_addr (mbar_base + 48)
    #define stage_dequant_addr (mbar_base + 96)
    #define combine_full_addr (mbar_base + 144)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_activation)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_activation_scale)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(l2_packed_weights)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    unsigned int* smem_expert_count = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_expert_count_addr = smem + 1024;
    unsigned int* lut_smem = reinterpret_cast<unsigned int*>(smem_raw + 3072);
    const int lut_smem_addr = smem + 3072;
    uint8_t* cd_smem = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int cd_smem_addr = smem + 4096;
    uint8_t* a_smem = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int a_smem_addr = smem + 21504;
    uint8_t* b_packed_smem = reinterpret_cast<uint8_t*>(smem_raw + 119808);
    const int b_packed_smem_addr = smem + 119808;
    uint8_t* b_decoded_smem = reinterpret_cast<uint8_t*>(smem_raw + 119808);
    const int b_decoded_smem_addr = smem + 119808;
    float* sfa_smem = reinterpret_cast<float*>(smem_raw + 218112);
    const int sfa_smem_addr = smem + 218112;
    uint8_t* combine_load_smem = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int combine_load_smem_addr = smem + 1024;
    uint8_t* combine_store_smem = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int combine_store_smem_addr = smem + 115712;

    // Mbarrier init (4 groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // stage_full: 6 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // stage_empty: 6 barriers, init_count=8
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            // stage_dequant: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
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

    // ---- Role: loader ----
    if (warp <= 1) {
        { // loader_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
            {
                const uint4 __cchunk = reinterpret_cast<const uint4*>(kE2M1AndUe4m3ToFp8Lut)[tid];
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(lut_smem_addr + (unsigned int)(tid * 16)), "r"(__cchunk.x), "r"(__cchunk.y), "r"(__cchunk.z), "r"(__cchunk.w));
            }
            if (warp == 0) {
                #pragma unroll 1
                for (int expert = lane; expert < 384; expert += 32) {
                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_expert_count_addr + (unsigned int)(expert * 4)), "r"((0)));
                }
            }
            asm volatile("barrier.sync 0, 384;" ::: "memory");
            unsigned int stored[2];
            #pragma unroll
            for (int expert_group = 0; expert_group < 2; expert_group++) {
                int local_expert = (unsigned int)(expert_group * 32) + lane;
                unsigned long long ready_status = 0;
                if (local_expert < 48) {
                    unsigned long long _sys_volatile_0;
                    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_0) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert])) : "memory");
                    ready_status = _sys_volatile_0;
                    while ((unsigned int)(ready_status >> 32) != 624) {
                        unsigned long long _sys_volatile_1;
                        asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_1) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert])) : "memory");
                        ready_status = _sys_volatile_1;
                    }
                }
                stored[expert_group] = (unsigned int)ready_status;
            }
            __syncwarp();
            unsigned int state[8];
            state[0] = (unsigned int)bid;
            state[1] = 0;
            state[3] = 0;
            unsigned int first_tokens[1];
            unsigned int lane_value = 0;
            #pragma unroll
            for (int expert_group_1 = 0; expert_group_1 < 2; expert_group_1++) {
                if (state[1] == (unsigned int)(expert_group_1 * 32) + lane) {
                    lane_value = stored[expert_group_1];
                }
            }
            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_value, (int)(state[1] % 32));
            first_tokens[0] = _shfl_0;
            state[2] = first_tokens[0];
            unsigned int load_stage = 0;
            unsigned int _phase_stage_empty = 1;
            while (state[1] < 48) {
                state[4] = 0;
                int scanning = 1;
                while (scanning != 0) {
                    if (state[1] < 48) {
                        unsigned int num_m_blocks = (state[2] + 127) / 128;
                        if (state[0] < num_m_blocks * 56) {
                            state[5] = state[0] / 56;
                            state[6] = state[0] - state[5] * 56;
                            state[0] = state[0] + 78;
                            state[4] = 1;
                            scanning = 0;
                        } else {
                            state[0] = state[0] - num_m_blocks * 56;
                            state[3] = state[3] + num_m_blocks;
                            state[1] = state[1] + 1;
                            unsigned int next_tokens[1];
                            unsigned int lane_value_0 = 0;
                            #pragma unroll
                            for (int expert_group_2 = 0; expert_group_2 < 2; expert_group_2++) {
                                if (state[1] == (unsigned int)(expert_group_2 * 32) + lane) {
                                    lane_value_0 = stored[expert_group_2];
                                }
                            }
                            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_value_0, (int)(state[1] % 32));
                            next_tokens[0] = _shfl_1;
                            state[2] = next_tokens[0];
                        }
                    } else {
                        scanning = 0;
                    }
                }
                if (state[4] != 0) {
                    int pool_token_idx = (int)((state[3] + state[5]) * 128);
                    int _min_0 = (((int)(state[2] - state[5] * 128)) < (128) ? ((int)(state[2] - state[5] * 128)) : (128));
                    int valid_m = _min_0;
                    int weight_row = (int)(state[1] * 7168 + state[6] * 128);
                    #pragma unroll 1
                    for (int k_block_idx = 0; k_block_idx < 24; k_block_idx++) {
                        mbarrier_wait(stage_empty_addr + (load_stage) * 8, _phase_stage_empty);
                        if (warp == 0) {
                            if (valid_m > 0) {
                                if (elect_sync()) {
                                    tma_2d_gmem2smem(a_smem_addr + load_stage * 16384, l2_activation, k_block_idx * 128, pool_token_idx, stage_full_addr + (load_stage) * 8);
                                    tma_2d_gmem2smem(sfa_smem_addr + load_stage * 512, l2_activation_scale, pool_token_idx, k_block_idx, stage_full_addr + (load_stage) * 8);
                                    mbarrier_arrive_expect_tx(stage_full_addr + (load_stage) * 8, 16896);
                                }
                            } else {
                                if (elect_sync()) {
                                    mbarrier_arrive(stage_full_addr + (load_stage) * 8);
                                }
                            }
                        } else {
                            if (elect_sync()) {
                                tma_2d_gmem2smem(b_packed_smem_addr + load_stage * 16384, l2_packed_weights, k_block_idx * 80, weight_row, stage_full_addr + (load_stage) * 8);
                                mbarrier_arrive_expect_tx(stage_full_addr + (load_stage) * 8, 10240);
                            }
                        }
                        __syncwarp();
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_stage_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: decode ----
    if (warp >= 2 && warp <= 3) {
        { // decode_main
            asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
            asm volatile("barrier.sync 0, 384;" ::: "memory");
            int dequant_tid = tid - 64;
            unsigned int stored_d[2];
            #pragma unroll
            for (int expert_group_3 = 0; expert_group_3 < 2; expert_group_3++) {
                int local_expert_1 = (unsigned int)(expert_group_3 * 32) + lane;
                unsigned long long ready_status_1 = 0;
                if (local_expert_1 < 48) {
                    unsigned long long _sys_volatile_2;
                    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_2) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert_1])) : "memory");
                    ready_status_1 = _sys_volatile_2;
                    while ((unsigned int)(ready_status_1 >> 32) != 624) {
                        unsigned long long _sys_volatile_3;
                        asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_3) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert_1])) : "memory");
                        ready_status_1 = _sys_volatile_3;
                    }
                }
                stored_d[expert_group_3] = (unsigned int)ready_status_1;
            }
            __syncwarp();
            unsigned int state_d[8];
            state_d[0] = (unsigned int)bid;
            state_d[1] = 0;
            state_d[3] = 0;
            unsigned int first_tokens_1[1];
            unsigned int lane_value_1 = 0;
            #pragma unroll
            for (int expert_group_4 = 0; expert_group_4 < 2; expert_group_4++) {
                if (state_d[1] == (unsigned int)(expert_group_4 * 32) + lane) {
                    lane_value_1 = stored_d[expert_group_4];
                }
            }
            unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, lane_value_1, (int)(state_d[1] % 32));
            first_tokens_1[0] = _shfl_2;
            state_d[2] = first_tokens_1[0];
            unsigned int decode_stage = 0;
            unsigned int _phase_stage_full = 0;
            while (state_d[1] < 48) {
                state_d[4] = 0;
                int scanning_1 = 1;
                while (scanning_1 != 0) {
                    if (state_d[1] < 48) {
                        unsigned int num_m_blocks_1 = (state_d[2] + 127) / 128;
                        if (state_d[0] < num_m_blocks_1 * 56) {
                            state_d[5] = state_d[0] / 56;
                            state_d[6] = state_d[0] - state_d[5] * 56;
                            state_d[0] = state_d[0] + 78;
                            state_d[4] = 1;
                            scanning_1 = 0;
                        } else {
                            state_d[0] = state_d[0] - num_m_blocks_1 * 56;
                            state_d[3] = state_d[3] + num_m_blocks_1;
                            state_d[1] = state_d[1] + 1;
                            unsigned int next_tokens_1[1];
                            unsigned int lane_value_0_1 = 0;
                            #pragma unroll
                            for (int expert_group_5 = 0; expert_group_5 < 2; expert_group_5++) {
                                if (state_d[1] == (unsigned int)(expert_group_5 * 32) + lane) {
                                    lane_value_0_1 = stored_d[expert_group_5];
                                }
                            }
                            unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, lane_value_0_1, (int)(state_d[1] % 32));
                            next_tokens_1[0] = _shfl_3;
                            state_d[2] = next_tokens_1[0];
                        }
                    } else {
                        scanning_1 = 0;
                    }
                }
                if (state_d[4] != 0) {
                    #pragma unroll 1
                    for (int _k_block_idx = 0; _k_block_idx < 24; _k_block_idx++) {
                        mbarrier_wait(stage_full_addr + (decode_stage) * 8, _phase_stage_full);
                        int row0 = dequant_tid;
                        int row1 = dequant_tid + 64;
                        int row_addr0 = b_packed_smem_addr + decode_stage * 16384 + (unsigned int)(row0 * 80);
                        int row_addr1 = b_packed_smem_addr + decode_stage * 16384 + (unsigned int)(row1 * 80);
                        unsigned int scale_words0[2];
                        unsigned int scale_words1[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scale_words0[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words0[(0) + 1]))
                            : "r"(row_addr0 + 64));
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&scale_words1[0])), "=r"(*reinterpret_cast<uint32_t*>(&scale_words1[(0) + 1]))
                            : "r"(row_addr1 + 64));
                        unsigned int quads0[16];
                        unsigned int quads1[16];
                        #pragma unroll
                        for (int quad = 0; quad < 4; quad++) {
                            unsigned int quad_words0[4];
                            unsigned int quad_words1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&quad_words0[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words0[(0) + 3]))
                                : "r"(row_addr0 + quad * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&quad_words1[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad_words1[(0) + 3]))
                                : "r"(row_addr1 + quad * 16));
                            #pragma unroll
                            for (int word = 0; word < 4; word++) {
                                quads0[quad * 4 + word] = quad_words0[word];
                                quads1[quad * 4 + word] = quad_words1[word];
                            }
                        }
                        asm volatile("bar.sync 8, 64;" ::: "memory");
                        #pragma unroll
                        for (int quad_1 = 0; quad_1 < 4; quad_1++) {
                            unsigned int word0 = scale_words0[0];
                            unsigned int word1 = scale_words1[0];
                            if (quad_1 >= 2) {
                                word0 = scale_words0[1];
                                word1 = scale_words1[1];
                            }
                            unsigned int s00 = word0 >> (unsigned int)((quad_1 * 2 & 3) * 8) & 127;
                            unsigned int s10 = word1 >> (unsigned int)((quad_1 * 2 & 3) * 8) & 127;
                            unsigned int s01 = word0 >> (unsigned int)((quad_1 * 2 + 1 & 3) * 8) & 127;
                            unsigned int s11 = word1 >> (unsigned int)((quad_1 * 2 + 1 & 3) * 8) & 127;
                            unsigned int lut00[2];
                            unsigned int lut10[2];
                            unsigned int lut01[2];
                            unsigned int lut11[2];
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut00[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut00[(0) + 1]))
                                : "r"(lut_smem_addr + (unsigned int)((int)s00 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut10[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut10[(0) + 1]))
                                : "r"(lut_smem_addr + (unsigned int)((int)s10 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut01[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut01[(0) + 1]))
                                : "r"(lut_smem_addr + (unsigned int)((int)s01 * 8)));
                            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&lut11[0])), "=r"(*reinterpret_cast<uint32_t*>(&lut11[(0) + 1]))
                                : "r"(lut_smem_addr + (unsigned int)((int)s11 * 8)));
                            unsigned int q0x_hi = 0;
                            unsigned int q0x_lo = 0;
                            unsigned int q0y_hi = 0;
                            unsigned int q0y_lo = 0;
                            unsigned int q1x_hi = 0;
                            unsigned int q1x_lo = 0;
                            unsigned int q1y_hi = 0;
                            unsigned int q1y_lo = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads0[quad_1 * 4]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut00[0])), "r"((uint32_t)(lut00[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut00[0])), "r"((uint32_t)(lut00[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0x_hi = _nv_hi;
                                q0x_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads0[quad_1 * 4 + 1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut00[0])), "r"((uint32_t)(lut00[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut00[0])), "r"((uint32_t)(lut00[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0y_hi = _nv_hi;
                                q0y_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads1[quad_1 * 4]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut10[0])), "r"((uint32_t)(lut10[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut10[0])), "r"((uint32_t)(lut10[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1x_hi = _nv_hi;
                                q1x_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads1[quad_1 * 4 + 1]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut10[0])), "r"((uint32_t)(lut10[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut10[0])), "r"((uint32_t)(lut10[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1y_hi = _nv_hi;
                                q1y_lo = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((b_decoded_smem_addr + decode_stage * 16384 + (unsigned int)(row0 * 128 + quad_1 * 2 * 16 ^ (row0 * 128 + quad_1 * 2 * 16 >> 7 & 7) << 4))), "r"(q0x_hi), "r"(q0x_lo), "r"(q0y_hi), "r"(q0y_lo) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((b_decoded_smem_addr + decode_stage * 16384 + (unsigned int)(row1 * 128 + quad_1 * 2 * 16 ^ (row1 * 128 + quad_1 * 2 * 16 >> 7 & 7) << 4))), "r"(q1x_hi), "r"(q1x_lo), "r"(q1y_hi), "r"(q1y_lo) : "memory");
                            unsigned int q0z_hi = 0;
                            unsigned int q0z_lo = 0;
                            unsigned int q0w_hi = 0;
                            unsigned int q0w_lo = 0;
                            unsigned int q1z_hi = 0;
                            unsigned int q1z_lo = 0;
                            unsigned int q1w_hi = 0;
                            unsigned int q1w_lo = 0;
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads0[quad_1 * 4 + 2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut01[0])), "r"((uint32_t)(lut01[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut01[0])), "r"((uint32_t)(lut01[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0z_hi = _nv_hi;
                                q0z_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads0[quad_1 * 4 + 3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut01[0])), "r"((uint32_t)(lut01[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut01[0])), "r"((uint32_t)(lut01[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q0w_hi = _nv_hi;
                                q0w_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads1[quad_1 * 4 + 2]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut11[0])), "r"((uint32_t)(lut11[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut11[0])), "r"((uint32_t)(lut11[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1z_hi = _nv_hi;
                                q1z_lo = _nv_lo;
                            }
                            {
                                const uint32_t _nv_packed = (uint32_t)(quads1[quad_1 * 4 + 3]);
                                const uint32_t _nv_selectors = _nv_packed & 0x77777777u;
                                uint32_t _nv_hi, _nv_lo;
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_hi) : "r"((uint32_t)(lut11[0])), "r"((uint32_t)(lut11[1])), "r"(_nv_selectors));
                                asm volatile("prmt.b32 %0, %1, %2, %3;" : "=r"(_nv_lo) : "r"((uint32_t)(lut11[0])), "r"((uint32_t)(lut11[1])), "r"(_nv_selectors >> 16));
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_hi) : "r"(_nv_packed));
                                const uint32_t _nv_shifted = _nv_packed << 4;
                                asm volatile("lop3.b32 %0, %0, %1, 0x80808080, 0xf8;" : "+r"(_nv_lo) : "r"(_nv_shifted));
                                q1w_hi = _nv_hi;
                                q1w_lo = _nv_lo;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((b_decoded_smem_addr + decode_stage * 16384 + (unsigned int)(row0 * 128 + (quad_1 * 2 + 1) * 16 ^ (row0 * 128 + (quad_1 * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q0z_hi), "r"(q0z_lo), "r"(q0w_hi), "r"(q0w_lo) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((b_decoded_smem_addr + decode_stage * 16384 + (unsigned int)(row1 * 128 + (quad_1 * 2 + 1) * 16 ^ (row1 * 128 + (quad_1 * 2 + 1) * 16 >> 7 & 7) << 4))), "r"(q1z_hi), "r"(q1z_lo), "r"(q1w_hi), "r"(q1w_lo) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("bar.sync 8, 64;" ::: "memory");
                        if (dequant_tid == 0) {
                            mbarrier_arrive(stage_dequant_addr + (decode_stage) * 8);
                        }
                        __syncwarp();
                        decode_stage += 1;
                        if (decode_stage == 6) { decode_stage = 0; _phase_stage_full ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: math ----
    if (warp >= 4 && warp <= 11) {
        { // math_main
            asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
            asm volatile("barrier.sync 0, 384;" ::: "memory");
            int epilogue_warp_idx = warp - 4;
            int epilogue_wg_idx = epilogue_warp_idx / 4;
            int epilogue_thread_idx = tid - 128;
            int warp_idx_in_wg = epilogue_warp_idx % 4;
            int row_block_offset = epilogue_wg_idx * 64;
            int tile_addr = cd_smem_addr + (unsigned int)(epilogue_warp_idx * 2176);
            unsigned int stored_m[2];
            #pragma unroll
            for (int expert_group_6 = 0; expert_group_6 < 2; expert_group_6++) {
                int local_expert_2 = (unsigned int)(expert_group_6 * 32) + lane;
                unsigned long long ready_status_2 = 0;
                if (local_expert_2 < 48) {
                    unsigned long long _sys_volatile_4;
                    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_4) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert_2])) : "memory");
                    ready_status_2 = _sys_volatile_4;
                    while ((unsigned int)(ready_status_2 >> 32) != 624) {
                        unsigned long long _sys_volatile_5;
                        asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(_sys_volatile_5) : "l"(reinterpret_cast<const unsigned long long*>(&reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)[local_expert_2])) : "memory");
                        ready_status_2 = _sys_volatile_5;
                    }
                }
                stored_m[expert_group_6] = (unsigned int)ready_status_2;
            }
            __syncwarp();
            unsigned int state_m[8];
            state_m[0] = (unsigned int)bid;
            state_m[1] = 0;
            state_m[3] = 0;
            unsigned int first_tokens_2[1];
            unsigned int lane_value_2 = 0;
            #pragma unroll
            for (int expert_group_7 = 0; expert_group_7 < 2; expert_group_7++) {
                if (state_m[1] == (unsigned int)(expert_group_7 * 32) + lane) {
                    lane_value_2 = stored_m[expert_group_7];
                }
            }
            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, lane_value_2, (int)(state_m[1] % 32));
            first_tokens_2[0] = _shfl_4;
            state_m[2] = first_tokens_2[0];
            unsigned int math_stage = 0;
            unsigned int ts_block = 0;
            unsigned int ts_kblock = 0;
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            unsigned int _phase_stage_dequant = 0;
            while (state_m[1] < 48) {
                state_m[4] = 0;
                int scanning_2 = 1;
                while (scanning_2 != 0) {
                    if (state_m[1] < 48) {
                        unsigned int num_m_blocks_2 = (state_m[2] + 127) / 128;
                        if (state_m[0] < num_m_blocks_2 * 56) {
                            state_m[5] = state_m[0] / 56;
                            state_m[6] = state_m[0] - state_m[5] * 56;
                            state_m[0] = state_m[0] + 78;
                            state_m[4] = 1;
                            scanning_2 = 0;
                        } else {
                            state_m[0] = state_m[0] - num_m_blocks_2 * 56;
                            state_m[3] = state_m[3] + num_m_blocks_2;
                            state_m[1] = state_m[1] + 1;
                            unsigned int next_tokens_2[1];
                            unsigned int lane_value_0_2 = 0;
                            #pragma unroll
                            for (int expert_group_8 = 0; expert_group_8 < 2; expert_group_8++) {
                                if (state_m[1] == (unsigned int)(expert_group_8 * 32) + lane) {
                                    lane_value_0_2 = stored_m[expert_group_8];
                                }
                            }
                            unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, lane_value_0_2, (int)(state_m[1] % 32));
                            next_tokens_2[0] = _shfl_5;
                            state_m[2] = next_tokens_2[0];
                        }
                    } else {
                        scanning_2 = 0;
                    }
                }
                if (state_m[4] != 0) {
                    int local_expert_idx = (int)state_m[1];
                    int m_idx = (int)((state_m[3] + state_m[5]) * 128);
                    int _min_1 = (((int)(state_m[2] - state_m[5] * 128)) < (128) ? ((int)(state_m[2] - state_m[5] * 128)) : (128));
                    int valid_m_m = _min_1;
                    int n_idx = (int)(state_m[6] * 128);
                    if (row_block_offset >= valid_m_m) {
                        #pragma unroll 1
                        for (int _k_skip = 0; _k_skip < 24; _k_skip++) {
                            mbarrier_wait(stage_dequant_addr + (math_stage) * 8, _phase_stage_dequant);
                            if (elect_sync()) {
                                mbarrier_arrive(stage_empty_addr + (math_stage) * 8);
                            }
                            __syncwarp();
                            math_stage += 1;
                            if (math_stage == 6) { math_stage = 0; _phase_stage_dequant ^= 1; }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                    } else {
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
                        for (int _k_block_idx_1 = 0; _k_block_idx_1 < 24; _k_block_idx_1++) {
                            mbarrier_wait(stage_dequant_addr + (math_stage) * 8, _phase_stage_dequant);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            int epilogue_warp_idx_0 = warp - 4;
                            int warp_idx_in_wg_1 = epilogue_warp_idx_0 % 4;
                            int row_idx = lane / 4;
                            int r_0 = warp_idx_in_wg_1 * 16 + row_idx;
                            int r_1 = r_0 + 8;
                            float scale_0[1];
                            float scale_1[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_0[0])) : "r"(sfa_smem_addr + math_stage * 512 + (unsigned int)((row_block_offset + r_0) * 4)));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&scale_1[0])) : "r"(sfa_smem_addr + math_stage * 512 + (unsigned int)((row_block_offset + r_1) * 4)));
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
                            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                            uint64_t _wgmma_desc_0 = (((uint64_t)(((a_smem_addr + math_stage * 16384 + (unsigned int)(epilogue_wg_idx * 64 * 128))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                            uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
                            uint64_t _wgmma_desc_1 = (((uint64_t)(((b_decoded_smem_addr + math_stage * 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                            uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 0, 1, 1;\n}\n"
                                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                                : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                                : "memory");
                            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %65, 1, 1, 1;\n}\n"
                                : "+f"(accum[0]), "+f"(accum[1]), "+f"(accum[2]), "+f"(accum[3]), "+f"(accum[4]), "+f"(accum[5]), "+f"(accum[6]), "+f"(accum[7]), "+f"(accum[8]), "+f"(accum[9]), "+f"(accum[10]), "+f"(accum[11]), "+f"(accum[12]), "+f"(accum[13]), "+f"(accum[14]), "+f"(accum[15]), "+f"(accum[16]), "+f"(accum[17]), "+f"(accum[18]), "+f"(accum[19]), "+f"(accum[20]), "+f"(accum[21]), "+f"(accum[22]), "+f"(accum[23]), "+f"(accum[24]), "+f"(accum[25]), "+f"(accum[26]), "+f"(accum[27]), "+f"(accum[28]), "+f"(accum[29]), "+f"(accum[30]), "+f"(accum[31]), "+f"(accum[32]), "+f"(accum[33]), "+f"(accum[34]), "+f"(accum[35]), "+f"(accum[36]), "+f"(accum[37]), "+f"(accum[38]), "+f"(accum[39]), "+f"(accum[40]), "+f"(accum[41]), "+f"(accum[42]), "+f"(accum[43]), "+f"(accum[44]), "+f"(accum[45]), "+f"(accum[46]), "+f"(accum[47]), "+f"(accum[48]), "+f"(accum[49]), "+f"(accum[50]), "+f"(accum[51]), "+f"(accum[52]), "+f"(accum[53]), "+f"(accum[54]), "+f"(accum[55]), "+f"(accum[56]), "+f"(accum[57]), "+f"(accum[58]), "+f"(accum[59]), "+f"(accum[60]), "+f"(accum[61]), "+f"(accum[62]), "+f"(accum[63])
                                : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1 + 2)
                                : "memory");
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
                                mbarrier_arrive(stage_empty_addr + (math_stage) * 8);
                            }
                            #pragma unroll
                            for (int accum_chunk = 0; accum_chunk < 16; accum_chunk++) {
                                final_accum[accum_chunk * 4] = final_accum[accum_chunk * 4] + scale_0[0] * accum[accum_chunk * 4];
                                final_accum[accum_chunk * 4 + 1] = final_accum[accum_chunk * 4 + 1] + scale_0[0] * accum[accum_chunk * 4 + 1];
                                final_accum[accum_chunk * 4 + 2] = final_accum[accum_chunk * 4 + 2] + scale_1[0] * accum[accum_chunk * 4 + 2];
                                final_accum[accum_chunk * 4 + 3] = final_accum[accum_chunk * 4 + 3] + scale_1[0] * accum[accum_chunk * 4 + 3];
                            }
                            ts_kblock = ts_kblock + 1;
                            math_stage += 1;
                            if (math_stage == 6) { math_stage = 0; _phase_stage_dequant ^= 1; }
                        }
                        float _vec_load_0[1];
                        {
                            uint32_t _scalar_bits_2;
                            asm volatile("ld.global.nc.b32 %0, [%1];"
                                : "=r"(_scalar_bits_2) : "l"((const void*)(l2_global_scales + (local_expert_idx))) : "memory");
                            _vec_load_0[0] = __uint_as_float(_scalar_bits_2);
                        }
                        float l2_global_scale = _vec_load_0[0];
                        int warp_row_base = row_block_offset + warp_idx_in_wg * 16;
                        int row_idx_m = lane / 4;
                        if (valid_m_m > warp_row_base + row_idx_m) {
                            int row_idx_1 = lane / 4;
                            int col_idx = lane % 4;
                            #pragma unroll
                            for (int pair_idx = 0; pair_idx < 8; pair_idx++) {
                                float values_lo[2];
                                float values_hi[2];
                                values_lo[0] = final_accum[pair_idx * 2 * 4] * l2_global_scale;
                                values_lo[1] = final_accum[pair_idx * 2 * 4 + 1] * l2_global_scale;
                                values_hi[0] = final_accum[(pair_idx * 2 + 1) * 4] * l2_global_scale;
                                values_hi[1] = final_accum[(pair_idx * 2 + 1) * 4 + 1] * l2_global_scale;
                                unsigned int packed_lo[1];
                                unsigned int packed_hi[1];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_lo[_lp*2 + 0], values_lo[_lp*2+1 + 0]));
                                    packed_lo[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_hi[_lp*2 + 0], values_hi[_lp*2+1 + 0]));
                                    packed_hi[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(tile_addr + (row_idx_1 * 136 + (pair_idx * 2 * 8 + col_idx * 2)) * 2), "r"((packed_lo[0])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(tile_addr + (row_idx_1 * 136 + ((pair_idx * 2 + 1) * 8 + col_idx * 2)) * 2), "r"((packed_hi[0])));
                            }
                        }
                        __syncwarp();
                        int scatter_row_in_pair = lane / 16;
                        int lane_in_row = lane % 16;
                        int group_leader = lane - (unsigned int)lane_in_row;
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            int stage_row = j * 2 + scatter_row_in_pair;
                            int row_offset = warp_row_base + stage_row;
                            int dst_rank = 0;
                            int dst_token = 0;
                            int dst_topk = 0;
                            if (lane_in_row == 0) {
                                int metadata_base = (m_idx + row_offset) * 3;
                                int _vec_load_1[1];
                                {
                                    _vec_load_1[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base);
                                }
                                int _vec_load_2[1];
                                {
                                    _vec_load_2[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base + 1);
                                }
                                int _vec_load_3[1];
                                {
                                    _vec_load_3[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base + 2);
                                }
                                dst_rank = _vec_load_1[0];
                                dst_token = _vec_load_2[0];
                                dst_topk = _vec_load_3[0];
                            }
                            int _shfl_6 = __shfl_sync(0xFFFFFFFF, dst_rank, group_leader);
                            dst_rank = _shfl_6;
                            int _shfl_7 = __shfl_sync(0xFFFFFFFF, dst_token, group_leader);
                            dst_token = _shfl_7;
                            int _shfl_8 = __shfl_sync(0xFFFFFFFF, dst_topk, group_leader);
                            dst_topk = _shfl_8;
                            unsigned int packed[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                                : "r"(tile_addr + stage_row * 272 + lane_in_row * 16));
                            if (row_offset < valid_m_m) {
                                unsigned long long dst_word_offset = (((unsigned long long)dst_topk * 8448 + (unsigned long long)dst_token) * 7168 + (unsigned long long)n_idx + (unsigned long long)lane_in_row * 8) / 2;
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank]) + 7181238272) + dst_word_offset)[0] = reinterpret_cast<int4*>(packed)[0];
                            }
                        }
                        __syncwarp();
                        if (valid_m_m > warp_row_base + row_idx_m + 8) {
                            int row_idx_2 = lane / 4;
                            int col_idx_1 = lane % 4;
                            #pragma unroll
                            for (int pair_idx_1 = 0; pair_idx_1 < 8; pair_idx_1++) {
                                float values_lo_1[2];
                                float values_hi_1[2];
                                values_lo_1[0] = final_accum[pair_idx_1 * 2 * 4 + 2] * l2_global_scale;
                                values_lo_1[1] = final_accum[pair_idx_1 * 2 * 4 + 2 + 1] * l2_global_scale;
                                values_hi_1[0] = final_accum[(pair_idx_1 * 2 + 1) * 4 + 2] * l2_global_scale;
                                values_hi_1[1] = final_accum[(pair_idx_1 * 2 + 1) * 4 + 2 + 1] * l2_global_scale;
                                unsigned int packed_lo_1[1];
                                unsigned int packed_hi_1[1];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_lo_1[_lp*2 + 0], values_lo_1[_lp*2+1 + 0]));
                                    packed_lo_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values_hi_1[_lp*2 + 0], values_hi_1[_lp*2+1 + 0]));
                                    packed_hi_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(tile_addr + (row_idx_2 * 136 + (pair_idx_1 * 2 * 8 + col_idx_1 * 2)) * 2), "r"((packed_lo_1[0])));
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"(tile_addr + (row_idx_2 * 136 + ((pair_idx_1 * 2 + 1) * 8 + col_idx_1 * 2)) * 2), "r"((packed_hi_1[0])));
                            }
                        }
                        __syncwarp();
                        int scatter_row_in_pair_0 = lane / 16;
                        int lane_in_row_1 = lane % 16;
                        int group_leader_2 = lane - (unsigned int)lane_in_row_1;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            int stage_row_1 = j_1 * 2 + scatter_row_in_pair_0;
                            int row_offset_1 = warp_row_base + 8 + stage_row_1;
                            int dst_rank_1 = 0;
                            int dst_token_1 = 0;
                            int dst_topk_1 = 0;
                            if (lane_in_row_1 == 0) {
                                int metadata_base_1 = (m_idx + row_offset_1) * 3;
                                int _vec_load_4[1];
                                {
                                    _vec_load_4[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base_1);
                                }
                                int _vec_load_5[1];
                                {
                                    _vec_load_5[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base_1 + 1);
                                }
                                int _vec_load_6[1];
                                {
                                    _vec_load_6[0] = *reinterpret_cast<const int*>(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 104437760) + metadata_base_1 + 2);
                                }
                                dst_rank_1 = _vec_load_4[0];
                                dst_token_1 = _vec_load_5[0];
                                dst_topk_1 = _vec_load_6[0];
                            }
                            int _shfl_9 = __shfl_sync(0xFFFFFFFF, dst_rank_1, group_leader_2);
                            dst_rank_1 = _shfl_9;
                            int _shfl_10 = __shfl_sync(0xFFFFFFFF, dst_token_1, group_leader_2);
                            dst_token_1 = _shfl_10;
                            int _shfl_11 = __shfl_sync(0xFFFFFFFF, dst_topk_1, group_leader_2);
                            dst_topk_1 = _shfl_11;
                            unsigned int packed_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                                : "r"(tile_addr + stage_row_1 * 272 + lane_in_row_1 * 16));
                            if (row_offset_1 < valid_m_m) {
                                unsigned long long dst_word_offset_1 = (((unsigned long long)dst_topk_1 * 8448 + (unsigned long long)dst_token_1) * 7168 + (unsigned long long)n_idx + (unsigned long long)lane_in_row_1 * 8) / 2;
                                reinterpret_cast<int4*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer_peers[dst_rank_1]) + 7181238272) + dst_word_offset_1)[0] = reinterpret_cast<int4*>(packed_1)[0];
                            }
                        }
                        asm volatile("bar.sync 2, 256;" ::: "memory");
                    }
                    ts_block = ts_block + 1;
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_3 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
                        const unsigned int _gs_add_3 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_3;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_3) : "l"(_gs_ptr_3), "r"(_gs_add_3) : "memory");
                        unsigned int _gs_new_3;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_3) : "l"(_gs_ptr_3) : "memory");
                        } while (((_gs_new_3 ^ _gs_old_3) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (bid == 0 && warp == 4) {
                {
                    const unsigned int _rz_status_4 = (*reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)) & 3u;
                    const unsigned int _rz_phase_4 = _rz_status_4 & 1u;
                    const int _rz_sign_4 = (int)(_rz_status_4 >> 1);
                    if ((int)(lane) < (int)(pg_world)) {
                        int* _rz_peer_4 = reinterpret_cast<int*>(reinterpret_cast<char*>((sym_buffer_peers)[lane]) + (unsigned long long)(20)) + _rz_phase_4;
                        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(_rz_peer_4), "r"(_rz_sign_4 ? -1 : 1) : "memory");
                    }
                    __syncwarp();
                    if ((int)(lane) == 0) {
                        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)), "r"(1u) : "memory");
                        const int _rz_target_4 = _rz_sign_4 ? 0 : (int)(pg_world);
                        int* _rz_local_4 = reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4) + (unsigned int)(1)) + _rz_phase_4;
                        while (true) {
                            int _rz_seen_4;
                            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_rz_seen_4) : "l"(_rz_local_4) : "memory");
                            if (_rz_seen_4 == _rz_target_4) break;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_5 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 1);
                        const unsigned int _gs_add_5 = (bid == 0) ? (0x80000000u - ((unsigned int)(78) - 1u)) : 1u;
                        unsigned int _gs_old_5;
                        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(_gs_old_5) : "l"(_gs_ptr_5), "r"(_gs_add_5) : "memory");
                        unsigned int _gs_new_5;
                        do {
                            asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(_gs_new_5) : "l"(_gs_ptr_5) : "memory");
                        } while (((_gs_new_5 ^ _gs_old_5) & 0x80000000u) == 0u);
                    }
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            unsigned int combine_phase = 0;
            unsigned int combine_load_stage = 0;
            #pragma unroll 1
            for (unsigned int combine_token_idx = bid * 8 + epilogue_warp_idx; combine_token_idx < num_tokens; combine_token_idx += 624) {
                int stored_topk_slot_idx = -1;
                if (lane < 6) {
                    long long _vec_load_7[1];
                    {
                        uint64_t _scalar_bits_6;
                        asm volatile("ld.global.nc.b64 %0, [%1];"
                            : "=l"(_scalar_bits_6) : "l"((const void*)(reinterpret_cast<long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 171862016) + (combine_token_idx * 6 + lane))) : "memory");
                        _vec_load_7[0] = (long long)_scalar_bits_6;
                    }
                    stored_topk_slot_idx = (int)_vec_load_7[0];
                }
                unsigned int _vote_0 = __ballot_sync(4294967295, stored_topk_slot_idx >= 0);
                unsigned int total_mask = _vote_0;
                #pragma unroll
                for (int combine_chunk = 0; combine_chunk < 2; combine_chunk++) {
                    unsigned int chunk_byte_offset = combine_chunk * 7168;
                    unsigned int combine_mask = total_mask;
                    int do_reduce = 0;
                    if (combine_mask != 0) {
                        int _ffs_0 = __ffs(combine_mask);
                        int slot_idx = _ffs_0 - 1;
                        combine_mask = combine_mask ^ (unsigned int)(1 << slot_idx);
                        unsigned int load_barrier_stage = (unsigned int)(epilogue_warp_idx * 2) + combine_load_stage;
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(combine_full_addr + (load_barrier_stage) * 8, 7168);
                            cp_async_bulk_gmem2smem(combine_load_smem_addr + load_barrier_stage * 7168, reinterpret_cast<uint8_t*>(sym_buffer) + (7181238272 + ((unsigned long long)slot_idx * 8448 + (unsigned long long)combine_token_idx) * 7168 * 2 + chunk_byte_offset), 7168, combine_full_addr + (load_barrier_stage) * 8);
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
                            int _ffs_1 = __ffs(combine_mask);
                            int next_slot_idx = _ffs_1 - 1;
                            combine_mask = combine_mask ^ (unsigned int)(1 << next_slot_idx);
                            unsigned int next_load_stage = combine_load_stage ^ 1;
                            unsigned int next_barrier_stage = (unsigned int)(epilogue_warp_idx * 2) + next_load_stage;
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(combine_full_addr + (next_barrier_stage) * 8, 7168);
                                cp_async_bulk_gmem2smem(combine_load_smem_addr + next_barrier_stage * 7168, reinterpret_cast<uint8_t*>(sym_buffer) + (7181238272 + ((unsigned long long)next_slot_idx * 8448 + (unsigned long long)combine_token_idx) * 7168 * 2 + chunk_byte_offset), 7168, combine_full_addr + (next_barrier_stage) * 8);
                            }
                            __syncwarp();
                            do_reduce = 1;
                        }
                        unsigned int current_barrier_stage = (unsigned int)(epilogue_warp_idx * 2) + combine_load_stage;
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
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx * 7168) + ((unsigned int)(combine_vec_1 * 32) + lane) * 16 + (unsigned int)(cast_word * 4)), "r"((casted[cast_word])));
                        }
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        {
                            void* _cpbulk_dst_7 = reinterpret_cast<void*>(output_bf16 + ((unsigned long long)combine_token_idx * 7168 + (unsigned long long)(chunk_byte_offset / 2)));
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                :: "l"(_cpbulk_dst_7), "r"(combine_store_smem_addr + (unsigned int)(epilogue_warp_idx * 7168)), "r"((uint32_t)(7168))
                                : "memory");
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    __syncwarp();
                }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (bid == 0) {
                #pragma unroll 1
                for (int expert_1 = epilogue_thread_idx; expert_1 < 384; expert_1 += 256) {
                    *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 128)) + (expert_1)) = 0;
                }
            } else {
                #pragma unroll 1
                for (int cleanup_expert = bid - 1; cleanup_expert < 48; cleanup_expert += 77) {
                    unsigned long long _vec_load_8[1];
                    {
                        _vec_load_8[0] = *reinterpret_cast<const unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272) + cleanup_expert);
                    }
                    unsigned int cleanup_num_tokens = (unsigned int)_vec_load_8[0];
                    unsigned int cleanup_num_m_blocks = (cleanup_num_tokens + 127) / 128;
                    unsigned int cleanup_pool_block_offset[1];
                    unsigned int lane_blocks = 0;
                    #pragma unroll
                    for (int expert_group_9 = 0; expert_group_9 < 2; expert_group_9++) {
                        if ((unsigned int)(expert_group_9 * 32) + lane < (unsigned int)cleanup_expert) {
                            lane_blocks = lane_blocks + (stored_m[expert_group_9] + 127) / 128;
                        }
                    }
                    unsigned int _warp_redux_u32_0;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(lane_blocks));
                    cleanup_pool_block_offset[0] = _warp_redux_u32_0;
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                    if (epilogue_thread_idx == 0) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6272)) + (cleanup_expert)) = 0;
                    }
                    if (epilogue_thread_idx < 8) {
                        *(reinterpret_cast<unsigned long long*>(reinterpret_cast<unsigned long long*>(reinterpret_cast<uint8_t*>(sym_buffer) + 3200)) + (epilogue_thread_idx * 48 + cleanup_expert)) = 0;
                    }
                    #pragma unroll 1
                    for (unsigned int cleanup_block = (unsigned int)epilogue_thread_idx; cleanup_block < cleanup_num_m_blocks; cleanup_block += 256) {
                        *(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer) + 6656)) + (cleanup_pool_block_offset[0] + cleanup_block)) = 0;
                    }
                    asm volatile("bar.sync 2, 256;" ::: "memory");
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    {
                        unsigned int* _gs_ptr_8 = reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 0);
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
            if (bid == 0 && warp == 4) {
                {
                    const unsigned int _rz_status_9 = (*reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)) & 3u;
                    const unsigned int _rz_phase_9 = _rz_status_9 & 1u;
                    const int _rz_sign_9 = (int)(_rz_status_9 >> 1);
                    if ((int)(lane) < (int)(pg_world)) {
                        int* _rz_peer_9 = reinterpret_cast<int*>(reinterpret_cast<char*>((sym_buffer_peers)[lane]) + (unsigned long long)(20)) + _rz_phase_9;
                        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(_rz_peer_9), "r"(_rz_sign_9 ? -1 : 1) : "memory");
                    }
                    __syncwarp();
                    if ((int)(lane) == 0) {
                        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4)), "r"(1u) : "memory");
                        const int _rz_target_9 = _rz_sign_9 ? 0 : (int)(pg_world);
                        int* _rz_local_9 = reinterpret_cast<int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<unsigned int*>(reinterpret_cast<uint8_t*>(sym_buffer)) + 4) + (unsigned int)(1)) + _rz_phase_9;
                        while (true) {
                            int _rz_seen_9;
                            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(_rz_seen_9) : "l"(_rz_local_9) : "memory");
                            if (_rz_seen_9 == _rz_target_9) break;
                        }
                    }
                    __syncwarp();
                }
            }
            asm volatile("bar.sync 2, 256;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"

