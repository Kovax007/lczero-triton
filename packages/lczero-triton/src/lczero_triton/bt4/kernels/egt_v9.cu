// EGT2 attention, v9 (backend lane 09-28/29; `attention_egt_v9.py`, the H3 shape): per-warp head pipelines over a half-position.
// A CTA owns a half-position (32 query rows): the FP16 cell-major edge state is loaded once into shared memory and read by every
// head. It runs 4 heads at a time -- 16 warps = 4 heads x 2 row tiles x 2 key halves -- and each warp runs its head's pipeline in
// registers: q.k, the logit assembly (E, S * cbias, HFMA2 read / door / gate), a local softmax, P as the next MMA's operand, P.V,
// then one merge with its key-half partner. Q, K and V load straight into MMA fragments (V transposed with movmatrix). E is the
// served kernel's prefix-sum form per half-position: the half-position's bits are listed once, each head's 128 threads compute the
// entries' values (balanced whatever the clustering), scan them in FP32 into the head's table region, and a cell reads
// P[end] - P[start]. Only named barriers per head / pair; CTA barriers only when the work list moves to another half-position.
// Measured (4090 microbench, b64, K1's 1,024 real positions): 76.0 us per block against the served kernel's 111.9 (B9 FP16).
// Arithmetic class = B9 (FP16 E tables, HFMA2 state math, FP16 or FP32 MMA accumulate, FP32 softmax / prefix sums).
// Inputs are the builder's own plan tables (FP32, `BLOCK_TABLES` order) and the cell-major FP16 state copy
// (`egt_state_tiles.cast_state_cell_major`). Compile-time: HEADS, HD, HALVES (2 x batch), GATE_SCALE, OGATE, CAP, EXPORT_H,
// OUT_F16 (FP16 output instead of int8 codes), ACC16, GATE_TANH, SKEW, SMEM_BYTES (checked against the layout), H_F16 (the H export
// in FP16, each lane storing 8 consecutive keys: the edge site reads it with `logits_f16`).
#include <cuda_fp16.h>
#include <cstdint>

#ifndef HEADS
#define HEADS 32
#endif
#ifndef HD
#define HD 32
#endif
#ifndef OGATE
#define OGATE 1
#endif
#ifndef CAP
#define CAP 0
#endif
#ifndef EXPORT_H
#define EXPORT_H 0
#endif
#ifndef H_F16
#define H_F16 0
#endif
#ifndef OUT_F16
#define OUT_F16 0
#endif
#ifndef GATE_SCALE
#define GATE_SCALE 2.0f
#endif
#ifdef CONTROL_NO_SIGMA  // the fidelity gate's CONTROL (must fail): E without its scaled-coefficient term
#define SIGMA_TERM(norm_value, coefficient) 0.f
#else
#define SIGMA_TERM(norm_value, coefficient) ((norm_value) * (coefficient))
#endif
#ifndef HALVES
#error "HALVES (2 x batch) is a compile-time constant"
#endif

namespace {

constexpr int kHeads = HEADS, kHd = HD, kWidth = kHeads * kHd, kStride = (OGATE ? 4 : 3) * kWidth;
#ifndef GROUP
#define GROUP 4
#endif
constexpr int kGroup = GROUP, kGroups = kHeads / kGroup;       // heads in flight per CTA; groups per half-position
constexpr int kThreads = 128 * kGroup, kRows = 32;             // 4 warps per head in flight (2 row tiles x 2 key halves)
constexpr int kKs = kHd / 16, kNd = kHd / 8, kHalfNd = kNd / 2;  // k-steps over head_dim; P.V n-tiles; the half each partner finalizes
constexpr int kTab = 40, kPar = 128;                           // E channels padded 34 -> 40; FP32 scalars per head
constexpr int kSPitch = 64 * 16 + 16;                          // bytes per row of one state plane (8 channels x FP16 per cell, +16)
constexpr int kPlane = kRows * kSPitch;
constexpr int kState = 2 * kPlane;
constexpr int kTabHead = (kRows + 64) * kTab * 2 + kTab * 4;   // FP16 row table [32][40] + column table [64][40] + FP32 scaled [40]
constexpr int kExWarp = 32 + 16 * (kHd / 2);                   // floats: (max, sum) per row + the partner's half of O
constexpr int kHeadBytes = kTabHead > 4 * kExWarp * 4 ? kTabHead : 4 * kExWarp * 4;
constexpr int kHeadsEnd = kState + kGroup * kHeadBytes;
constexpr int kListCta = 512;                                  // entries per half-position (real positions: at most 321)
constexpr int kOffTot = kHeadsEnd;                             // 16 ints: per-warp bit totals; then 16 floats: per-warp E totals
constexpr int kOffList = kOffTot + 128;                        // uint32 (row << 12 | key << 6 | channel) << 15 | S (FP16, sign dropped)
constexpr int kSmem = kOffList + 4 * kListCta;
static_assert(kListCta * 4 <= kTabHead, "the prefix lives in the head's table region");
static_assert(4 * kListCta / 128 == 16, "4 entries per head thread");
static_assert(kHeads % kGroup == 0 && kHd % 16 == 0 && kHd <= 64, "shape");
static_assert(kSmem <= 101376, "shared memory over the sm_89 / sm_120 opt-in ceiling");
static_assert(kHeadBytes % 16 == 0, "head region alignment");
#ifdef SMEM_BYTES
static_assert(kSmem == SMEM_BYTES, "the builder's shared-memory size disagrees with the kernel's layout");
#endif

#ifdef ACC16
using Acc = uint32_t[2];  // half2 x 2
#else
using Acc = float[4];
#endif

__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* d, const void* s) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr(d)), "l"(s));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
__device__ __forceinline__ void named_bar(int id, int count) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(count) : "memory"); }
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void zero(float (&c)[4]) { c[0] = c[1] = c[2] = c[3] = 0.f; }
__device__ __forceinline__ void zero(uint32_t (&c)[2]) { c[0] = c[1] = 0u; }
// element i of an accumulator tile: (row g, col 2t), (g, 2t + 1), (g + 8, 2t), (g + 8, 2t + 1)
__device__ __forceinline__ float elem(const float (&c)[4], int i) { return c[i]; }
__device__ __forceinline__ float elem(const uint32_t (&c)[2], int i) {
  const __half2 h = *reinterpret_cast<const __half2*>(&c[i >> 1]);
  return (i & 1) ? __high2float(h) : __low2float(h);
}
__device__ __forceinline__ uint32_t transpose8x8(uint32_t x) {
  uint32_t y;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ uint32_t ld32(const __half* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float hsum(__half2 v) {
  const float2 f = __half22float2(v);
  return f.x + f.y;
}
__device__ __forceinline__ float two_sigmoid(float x) {
#ifdef GATE_TANH
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(0.5f * x));
  return 1.f + y;
#else
  return __fdividef(2.f, 1.f + __expf(-x));
#endif
}
__device__ __forceinline__ signed char code_of(float x) { return static_cast<signed char>(fminf(fmaxf(floorf(x + 0.5f), -127.f), 127.f)); }

#ifdef PACKED
// PACKED: the query / key tables as FP16 [heads][40][hd] (channels 34-39 zero), one 32-bit load per pair
__device__ __forceinline__ uint32_t table_pair(const float* __restrict__ table, int head, int channel, int dim) {
  return ld32(reinterpret_cast<const __half*>(table) + ((size_t)head * kTab + channel) * kHd + dim);
}
#else
// FP32 [heads][34][hd] pair of a table row's dims d, d + 1 as half2; channels past 34 read as zero (the tables' padding)
__device__ __forceinline__ uint32_t table_pair(const float* __restrict__ table, int head, int channel, int dim) {
  if (channel >= 34) return 0u;
  const float2 v = __ldg(reinterpret_cast<const float2*>(table + ((size_t)head * 34 + channel) * kHd + dim));
  return pack(v.x, v.y);
}
#endif

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1) egt_v9_attention(
    void* __restrict__ out_ptr, void* __restrict__ hexp_ptr, const __half* __restrict__ qkv,
    const unsigned long long* __restrict__ edges, const __half* __restrict__ norm, const __half* __restrict__ state,
    const float* __restrict__ qk_scale_t, const float* __restrict__ attack_t, const float* __restrict__ key_t,
    const float* __restrict__ query_t, const float* __restrict__ scaled_t, const float* __restrict__ cbias,
    const float* __restrict__ read_t, const float* __restrict__ door_t, const float* __restrict__ gate_t,
    const float* __restrict__ gate_b_t, const float* __restrict__ prescale) {
  constexpr int halves = HALVES;
#if OUT_F16
  __half* out = static_cast<__half*>(out_ptr);
  (void)prescale;
#else
  int8_t* out = static_cast<int8_t*>(out_ptr);
#endif
#if H_F16
  __half* hexp = static_cast<__half*>(hexp_ptr);
#else
  float* hexp = static_cast<float*>(hexp_ptr);
#endif
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int hs = warp >> 2, rt = (warp >> 1) & 1, kh = warp & 1;  // head slot, row tile, key half
  const unsigned char* sS = smem;
  unsigned char* head_region = smem + kState + hs * kHeadBytes;
  __half* sR = reinterpret_cast<__half*>(head_region);
  __half* sC = sR + kRows * kTab;
  float* sSig = reinterpret_cast<float*>(sC + 64 * kTab);
  float* sEx = reinterpret_cast<float*>(head_region);
  float* sP = reinterpret_cast<float*>(head_region);  // the head's E prefix over the list, once its tables are dead
  int* sWarpTot = reinterpret_cast<int*>(smem + kOffTot);
  float* sWarpSum = reinterpret_cast<float*>(smem + kOffTot + 64) + hs * 4;  // this head's 4 warps' E totals
  const uint32_t* sList = reinterpret_cast<const uint32_t*>(smem + kOffList);
  uint32_t* sListW = reinterpret_cast<uint32_t*>(smem + kOffList);
  const int bar_head = 1 + hs, bar_pair = 5 + 2 * hs + rt;

  // The work list over (half-position, head group) units, v8's L2-aware order: full rounds of whole half-positions (sister halves on
  // neighbouring CTAs), then the remainder's units in equal contiguous slices.
  const int S = gridDim.x, full = halves / S, rest = halves - full * S;
  const int t0 = static_cast<int>((long long)blockIdx.x * rest * kGroups / S);
  const int t1 = static_cast<int>((long long)(blockIdx.x + 1) * rest * kGroups / S);
  const int nq = full * kGroups + (t1 - t0);
  if (nq <= 0) return;
#define TICK(slot) do { } while (0)

  int current = -1;
  __half2 nrm[2][4];              // this lane's cells' S: rows g, g + 8 of the row tile; keys kh * 32 + 8n + 2t, + 1
  uint32_t nz = 0;                // which of its 16 cells (k = 8a + 2n + b) have edge bits
  int estart = 0;                 // this owner lane's first entry (owner = row tile, key half, lane; both head slots share it)
  unsigned long long counts = 0;  // its bits per cell k in 4-bit fields (a cell has at most 8 of 34 on real positions)
  bool eover = false;             // the half-position's bits exceed kListCta: every lane walks its cells' masks (exact, slow)
  int etotal = 0;                 // entries in the half-position's list
  for (int q = 0; q < nq; ++q) {
    const int u = q < full * kGroups ? ((q / kGroups) * S + blockIdx.x) * kGroups + q % kGroups
                                     : full * S * kGroups + t0 + (q - full * kGroups);
    const int hp = u / kGroups, grp = u % kGroups, sample = hp >> 1, row0 = (hp & 1) * kRows;
    const bool fresh = hp != current;
    if (fresh) {
      __syncthreads();  // every warp is past its last read of the previous state
      const __half* src = state + ((size_t)sample * 64 + row0) * 64 * 16;
      for (int i = tid; i < kRows * 64 * 2; i += kThreads) {
        const int chunk = i & 1, cell = (i >> 1) & 63, r = i >> 7;
        cp16(smem + chunk * kPlane + r * kSPitch + cell * 16, src + ((size_t)r * 64 + cell) * 16 + chunk * 8);
      }
      cp_commit();
      current = hp;
    }
    TICK(0);
    const int h = grp * kGroup + hs;
    const __half* base = qkv + (size_t)sample * 64 * kStride + h * kHd;

    // ---- loads for q.k and the tables, all issued ahead of the MMAs
    uint32_t qa[kKs][4];
    const __half* qrow = base + (size_t)(row0 + rt * 16 + g) * kStride + 2 * t;
#pragma unroll
    for (int ks = 0; ks < kKs; ++ks) {
      qa[ks][0] = ld32(qrow + 16 * ks);
      qa[ks][1] = ld32(qrow + 8 * kStride + 16 * ks);
      qa[ks][2] = ld32(qrow + 16 * ks + 8);
      qa[ks][3] = ld32(qrow + 8 * kStride + 16 * ks + 8);
    }
    uint32_t kb[4][kKs][2];  // keys kh * 32 + 8n + g
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const __half* krow = base + kWidth + (size_t)(kh * 32 + 8 * n + g) * kStride + 2 * t;
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) {
        kb[n][ks][0] = ld32(krow + 16 * ks);
        kb[n][ks][1] = ld32(krow + 16 * ks + 8);
      }
    }
    uint32_t tqf[3][kKs][2], tkf[3][kKs][2];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = kh ? (i < 2 ? 3 + i : 4) : i;  // kh = 1 has two tiles; its third slot repeats tile 4 and is not stored
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) {
        tqf[i][ks][0] = table_pair(query_t, h, 8 * nn + g, 16 * ks + 2 * t);
        tqf[i][ks][1] = table_pair(query_t, h, 8 * nn + g, 16 * ks + 2 * t + 8);
      }
    }
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      const int nn = rt ? (i < 2 ? 3 + i : 4) : i;  // the row tile picks the column table's channel tiles: 0-2 or 3-4
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) {
        tkf[i][ks][0] = table_pair(key_t, h, 8 * nn + g, 16 * ks + 2 * t);
        tkf[i][ks][1] = table_pair(key_t, h, 8 * nn + g, 16 * ks + 2 * t + 8);
      }
    }

    // ---- q.k and this warp's share of the head's E tables: row table (its 16 rows; channel tiles 0-2 or 3-4 by key half) and
    // column table (its 32 keys, whose A fragments are its K fragments; channel tiles 0-2 or 3-4 by row tile)
    Acc s[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      zero(s[n]);
#pragma unroll
      for (int ks = 0; ks < kKs; ++ks) mma(s[n], qa[ks], kb[n][ks][0], kb[n][ks][1]);
    }
    Acc racc[3], cacc[2][3];
#pragma unroll
    for (int i = 0; i < 3; ++i) {
      zero(racc[i]);
      if (!kh || i < 2) {
#pragma unroll
        for (int ks = 0; ks < kKs; ++ks) mma(racc[i], qa[ks], tqf[i][ks][0], tqf[i][ks][1]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        zero(cacc[mt][i]);
        if (!rt || i < 2) {
#pragma unroll
          for (int ks = 0; ks < kKs; ++ks) {
            const uint32_t a[4] = {kb[2 * mt][ks][0], kb[2 * mt + 1][ks][0], kb[2 * mt][ks][1], kb[2 * mt + 1][ks][1]};
            mma(cacc[mt][i], a, tkf[i][ks][0], tkf[i][ks][1]);
          }
        }
      }
    TICK(1);
    if (fresh) {  // uniform: every warp of the CTA takes this branch in the same unit
      nz = 0;
      counts = 0;
      int count = 0;
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const size_t row = (size_t)sample * 64 + row0 + rt * 16 + g + 8 * a;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          const int key = kh * 32 + 8 * n + 2 * t;
          nrm[a][n] = *reinterpret_cast<const __half2*>(norm + row * 64 + key);
          const ulonglong2 e = *reinterpret_cast<const ulonglong2*>(edges + row * 64 + key);
#pragma unroll
          for (int b = 0; b < 2; ++b) {
            const int k = a * 8 + n * 2 + b, bits = __popcll(b ? e.y : e.x);
            nz |= (bits ? 1u : 0u) << k;
            counts |= static_cast<unsigned long long>(bits < 15 ? bits : 15) << (4 * k);
            count += bits;
          }
        }
      }
      int inclusive = count;  // warp scan of the owner lanes' bit counts
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += v;
      }
      if (hs == 0 && lane == 31) sWarpTot[warp] = inclusive;
      bool cell_over = false;
#pragma unroll
      for (int k = 0; k < 16; ++k) cell_over |= ((counts >> (4 * k)) & 15u) == 15u;
      cp_wait_all();
      __syncthreads();  // the state is in; head slot 0's warp totals are visible
      int base_count = 0, total = 0;
#pragma unroll
      for (int w = 0; w < 4; ++w) {
        const int v = sWarpTot[w];
        base_count += w < (warp & 3) ? v : 0;
        total += v;
      }
      eover = __syncthreads_or(total > kListCta || cell_over) != 0;
      estart = base_count + inclusive - count;
      etotal = total;
      if (hs == 0 && !eover) {  // the owner's entries, in cell order, with the cell's S
        int at = estart;
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          const int rl = rt * 16 + g + 8 * a;
          const size_t row = (size_t)sample * 64 + row0 + rl;
#pragma unroll
          for (int n = 0; n < 4; ++n) {
            const ulonglong2 e = *reinterpret_cast<const ulonglong2*>(edges + row * 64 + kh * 32 + 8 * n + 2 * t);
#pragma unroll
            for (int b = 0; b < 2; ++b) {
              const int key = kh * 32 + 8 * n + 2 * t + b;
              const __half nb = b ? __high2half(nrm[a][n]) : __low2half(nrm[a][n]);
              const uint32_t sbits = static_cast<uint32_t>(__half_as_ushort(nb)) & 0x7FFFu;
              unsigned long long m = b ? e.y : e.x;
              while (m) {
                const uint32_t c = static_cast<uint32_t>(__ffsll(static_cast<long long>(m)) - 1);
                sListW[at++] = ((static_cast<uint32_t>(rl) << 12 | static_cast<uint32_t>(key) << 6 | c) << 15) | sbits;
                m &= m - 1;
              }
            }
          }
        }
      }
      __syncthreads();  // the list is visible to every head slot
#ifdef SKEW
      // probe (09-28): delay the odd head slots by SKEW cycles so that, on every SMSP, two warps run the FMA-heavy assembly while
      // the other two run their MMAs (the four head slots are otherwise in lockstep and the pipes take turns)
#ifdef SKEW_GRADED  // slot hs waits hs * SKEW / 4
      const long long wait = (long long)hs * SKEW / 4;
#else
      const long long wait = (hs & 1) ? SKEW : 0;
#endif
      if (wait) {
        const long long start = clock64();
        while (clock64() - start < wait) {
        }
      }
#endif
    }
    TICK(0);
    // the head's scalars, state weights and cbias: issued before the barrier so their latency hides under it
    const float qk_scale = __ldg(qk_scale_t + h), gate_b = __ldg(gate_b_t + h);
    __half2 wr[8], wd[8], wg[8];
#ifdef PACKED  // read_t = 24 packed half2 per head: read, door, gate x 8 channel pairs
    {
      const uint4* wv = reinterpret_cast<const uint4*>(read_t) + h * 6;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint4 x = __ldg(wv + i), y = __ldg(wv + 2 + i), z = __ldg(wv + 4 + i);
        const __half2* hx = reinterpret_cast<const __half2*>(&x);
        const __half2* hy = reinterpret_cast<const __half2*>(&y);
        const __half2* hz = reinterpret_cast<const __half2*>(&z);
#pragma unroll
        for (int m = 0; m < 4; ++m) { wr[4 * i + m] = hx[m]; wd[4 * i + m] = hy[m]; wg[4 * i + m] = hz[m]; }
      }
      (void)door_t;
      (void)gate_t;
    }
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float2 r = __ldg(reinterpret_cast<const float2*>(read_t + h * 16) + i);
      const float2 d = __ldg(reinterpret_cast<const float2*>(door_t + h * 16) + i);
      const float2 x = __ldg(reinterpret_cast<const float2*>(gate_t + h * 16) + i);
      wr[i] = __floats2half2_rn(r.x, r.y);
      wd[i] = __floats2half2_rn(d.x, d.y);
      wg[i] = __floats2half2_rn(x.x, x.y);
    }
#endif
    const float* cbh = cbias + (size_t)h * 4096;
    __half2 cb[2][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int n = 0; n < 4; ++n)
#ifdef PACKED  // cbias as FP16 [heads][64][64]
        cb[a][n] = __ldg(reinterpret_cast<const __half2*>(cbias) + (h * 4096 + (row0 + rt * 16 + g + 8 * a) * 64 + kh * 32 + 8 * n + 2 * t) / 2);
#else
        cb[a][n] = __float22half2_rn(
            __ldg(reinterpret_cast<const float2*>(cbh + (row0 + rt * 16 + g + 8 * a) * 64 + kh * 32 + 8 * n + 2 * t)));
#endif
    named_bar(bar_head, 128);  // B0: the head slot's previous exchange has been read
    {
      const int r = rt * 16 + g;
#pragma unroll
      for (int i = 0; i < 3; ++i) {
        if (!kh || i < 2) {
          const int c = 8 * (kh ? 3 + i : i) + 2 * t;
          const float a0 = c < 34 ? __ldg(attack_t + h * 34 + c) : 0.f, a1 = c + 1 < 34 ? __ldg(attack_t + h * 34 + c + 1) : 0.f;
          *reinterpret_cast<uint32_t*>(sR + r * kTab + c) = pack(elem(racc[i], 0) + a0, elem(racc[i], 1) + a1);
          *reinterpret_cast<uint32_t*>(sR + (r + 8) * kTab + c) = pack(elem(racc[i], 2) + a0, elem(racc[i], 3) + a1);
        }
      }
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        const int kr = kh * 32 + 16 * mt + g;
#pragma unroll
        for (int i = 0; i < 3; ++i) {
          if (!rt || i < 2) {
            const int c = 8 * (rt ? 3 + i : i) + 2 * t;
            *reinterpret_cast<uint32_t*>(sC + kr * kTab + c) = pack(elem(cacc[mt][i], 0), elem(cacc[mt][i], 1));
            *reinterpret_cast<uint32_t*>(sC + (kr + 8) * kTab + c) = pack(elem(cacc[mt][i], 2), elem(cacc[mt][i], 3));
          }
        }
      }
      if (rt == 0 && kh == 0) {
        sSig[lane] = __ldg(scaled_t + h * 34 + lane);                                         // channels 0-31
        if (lane < kTab - 32) sSig[32 + lane] = lane < 2 ? __ldg(scaled_t + h * 34 + 32 + lane) : 0.f;  // 32, 33, padding
      }
    }
    named_bar(bar_head, 128);  // B1: the tables are visible
    TICK(2);

    // ---- the logit assembly per cell: E entries, norm * cbias, read / door / gate from the shared state (HFMA2),
    // H = (F + read)(1 + door)
    // V for P.V, loaded now so its latency hides under the assembly (transposed at use)
    const __half* vrow = base + 2 * kWidth + (size_t)(kh * 32 + g) * kStride + 2 * t;
    uint32_t vraw[kNd][2][2];
#pragma unroll
    for (int nd = 0; nd < kNd; ++nd)
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        vraw[nd][kk][0] = ld32(vrow + (size_t)(16 * kk) * kStride + 8 * nd);
        vraw[nd][kk][1] = ld32(vrow + (size_t)(16 * kk + 8) * kStride + 8 * nd);
      }
    const bool walk = eover;  // the half-position's bits did not fit the list: every lane reads its cells' masks
#ifndef DBG_NO_E
    if (!walk) {  // this head's E values: 4 consecutive entries per head thread, then an FP32 prefix over the list
      const int ht = (warp & 3) * 32 + lane;  // thread within the head
      float v[4], run = 0.f;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        // slots past the list decode as entry 0 (cell 0, channel 0): every read stays inside this head's region (a stale word could
        // name channel 63, past the head's table; racecheck 09-29) and their value is selected out below
        const bool live = 4 * ht + r < etotal;
        const uint32_t entry = live ? sList[4 * ht + r] : 0u;
        const int c = static_cast<int>((entry >> 15) & 63u), key = static_cast<int>((entry >> 21) & 63u);
        const int rl = static_cast<int>(entry >> 27);
        const float nb = __half2float(__ushort_as_half(static_cast<unsigned short>(entry & 0x7FFFu)));
        const float x = __half2float(sR[rl * kTab + c]) + __half2float(sC[key * kTab + c]) + SIGMA_TERM(nb, sSig[c]);
        run += live ? x : 0.f;
        v[r] = run;
      }
      float incl = run;  // warp scan of the threads' sums
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float x = __shfl_up_sync(0xffffffffu, incl, offset);
        if (lane >= offset) incl += x;
      }
      if (lane == 31) sWarpSum[warp & 3] = incl;
      named_bar(bar_head, 128);  // the head's values are computed (its tables are dead) and its warp totals visible
      float offset_sum = incl - run;
#pragma unroll
      for (int w = 0; w < 4; ++w) offset_sum += w < (warp & 3) ? sWarpSum[w] : 0.f;
#pragma unroll
      for (int r = 0; r < 4; ++r) sP[4 * ht + r] = offset_sum + v[r];
      named_bar(bar_head, 128);  // the prefix is visible
    }
#endif
    int cell_start = estart;  // running start of cell k's entries
    float hv[4][4];
    __half2 dg[4][4];  // (door, 2 sigmoid(gate)) per cell
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int rl = rt * 16 + g + 8 * a;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        const float2 nr = __half22float2(nrm[a][n]), cbf = __half22float2(cb[a][n]);
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int k = a * 8 + n * 2 + b, key = kh * 32 + 8 * n + 2 * t + b;
#ifdef DBG_NO_STATE  // pricing switches (results wrong): drop one term
          const uint4 c0 = make_uint4(0, 0, 0, 0), c1 = c0;
#else
          const uint4 c0 = *reinterpret_cast<const uint4*>(sS + rl * kSPitch + key * 16);
          const uint4 c1 = *reinterpret_cast<const uint4*>(sS + kPlane + rl * kSPitch + key * 16);
#endif
          const __half2* x0 = reinterpret_cast<const __half2*>(&c0);
          const __half2* x1 = reinterpret_cast<const __half2*>(&c1);
          __half2 ar = __float2half2_rn(0.f), ad = ar, ag = ar;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            ar = __hfma2(x0[i], wr[i], ar);
            ad = __hfma2(x0[i], wd[i], ad);
            ag = __hfma2(x0[i], wg[i], ag);
            ar = __hfma2(x1[i], wr[4 + i], ar);
            ad = __hfma2(x1[i], wd[4 + i], ad);
            ag = __hfma2(x1[i], wg[4 + i], ag);
          }
          const float nb = b ? nr.y : nr.x;
          if (walk && ((nz >> k) & 1u)) {  // the rare lane whose bits did not fit the list
            unsigned long long m = __ldg(edges + ((size_t)sample * 64 + row0 + rl) * 64 + key);
            float e = 0.f;
            while (m) {
              const int c = __ffsll(static_cast<long long>(m)) - 1;
              m &= m - 1;
              e += __half2float(sR[rl * kTab + c]) + __half2float(sC[key * kTab + c]) + SIGMA_TERM(nb, sSig[c]);
            }
            hv[n][2 * a + b] = e;
          } else {
            const int cnt = static_cast<int>((counts >> (4 * k)) & 15u);
#ifdef DBG_NO_E
            hv[n][2 * a + b] = 0.f;
#else
            hv[n][2 * a + b] = cnt ? sP[cell_start + cnt - 1] - (cell_start ? sP[cell_start - 1] : 0.f) : 0.f;
#endif
            cell_start += cnt;
          }
          hv[n][2 * a + b] += elem(s[n], 2 * a + b) * qk_scale + nb * (b ? cbf.y : cbf.x) + hsum(ar);
          const float x = hsum(ag) + gate_b;
#if CAP
          dg[n][2 * a + b] = __floats2half2_rn(hsum(ad), x >= 0.f ? 1.f : two_sigmoid(x));
#else
          dg[n][2 * a + b] = __floats2half2_rn(hsum(ad), two_sigmoid(x));
#endif
        }
      }
    }
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int rl = rt * 16 + g + 8 * a;
#pragma unroll
      for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int b = 0; b < 2; ++b) hv[n][2 * a + b] *= 1.f + __low2float(dg[n][2 * a + b]);
#if EXPORT_H && !H_F16
        *reinterpret_cast<float2*>(hexp + (((size_t)sample * kHeads + h) * 64 + row0 + rl) * 64 + kh * 32 + 8 * n + 2 * t) =
            make_float2(hv[n][2 * a], hv[n][2 * a + 1]);
#endif
      }
#if EXPORT_H && H_F16
      {  // FP16 H: a 4 x 4 transpose across the quad (lane t, register n) -> (lane n, register t) gives lane t the key half's
         // n-tile t, 8 consecutive keys stored as one 16-byte write (a row's 32 keys = 64 contiguous bytes per instruction)
        uint32_t p[4];
#pragma unroll
        for (int n = 0; n < 4; ++n) p[n] = pack(hv[n][2 * a], hv[n][2 * a + 1]);
        const bool b1 = t & 2, b0 = t & 1;
        uint32_t x0 = __shfl_xor_sync(0xffffffffu, b1 ? p[0] : p[2], 2);  // registers whose bit 1 differs from the lane's
        uint32_t x1 = __shfl_xor_sync(0xffffffffu, b1 ? p[1] : p[3], 2);
        if (b1) { p[0] = x0; p[1] = x1; } else { p[2] = x0; p[3] = x1; }
        x0 = __shfl_xor_sync(0xffffffffu, b0 ? p[0] : p[1], 1);  // then bit 0
        x1 = __shfl_xor_sync(0xffffffffu, b0 ? p[2] : p[3], 1);
        if (b0) { p[0] = x0; p[2] = x1; } else { p[1] = x0; p[3] = x1; }
        *reinterpret_cast<uint4*>(hexp + (((size_t)sample * kHeads + h) * 64 + row0 + rl) * 64 + kh * 32 + 8 * t) =
            make_uint4(p[0], p[1], p[2], p[3]);
      }
#endif
    }
    TICK(3);

    // ---- local softmax over this key half (quad shuffles only); the gate multiplies the unnormalised exponentials
    float mrow[2], zrow[2];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      float mx = fmaxf(fmaxf(hv[0][2 * a], hv[0][2 * a + 1]), fmaxf(hv[1][2 * a], hv[1][2 * a + 1]));
      mx = fmaxf(mx, fmaxf(fmaxf(hv[2][2 * a], hv[2][2 * a + 1]), fmaxf(hv[3][2 * a], hv[3][2 * a + 1])));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
      float z = 0.f;
#pragma unroll
      for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          const float p = __expf(hv[n][2 * a + b] - mx);
          z += p;
          hv[n][2 * a + b] = p * __high2float(dg[n][2 * a + b]);
        }
      z += __shfl_xor_sync(0xffffffffu, z, 1);
      z += __shfl_xor_sync(0xffffffffu, z, 2);
      mrow[a] = mx;
      zrow[a] = z;
    }

    // the output-gate lane and the prescale of the n-tiles this warp finalizes, ahead of the exchange
    const int nd0 = kh * kHalfNd;
    float2 gl[kHalfNd][2], ps[kHalfNd];
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {
#if OUT_F16
      ps[i] = make_float2(1.f, 1.f);
#else
      ps[i] = __ldg(reinterpret_cast<const float2*>(prescale + h * kHd + 8 * (nd0 + i) + 2 * t));
#endif
#pragma unroll
      for (int a = 0; a < 2; ++a) {
#if OGATE
        gl[i][a] = __half22float2(__ldg(reinterpret_cast<const __half2*>(
            base + 3 * kWidth + (size_t)(row0 + rt * 16 + g + 8 * a) * kStride + 8 * (nd0 + i) + 2 * t)));
#else
        gl[i][a] = make_float2(0.f, 0.f);
#endif
      }
    }
    // ---- P.V: P as A fragments from the accumulator layout; V fragments transposed in registers
    uint32_t pa[2][4];
#pragma unroll
    for (int kk = 0; kk < 2; ++kk) {
      pa[kk][0] = pack(hv[2 * kk][0], hv[2 * kk][1]);
      pa[kk][1] = pack(hv[2 * kk][2], hv[2 * kk][3]);
      pa[kk][2] = pack(hv[2 * kk + 1][0], hv[2 * kk + 1][1]);
      pa[kk][3] = pack(hv[2 * kk + 1][2], hv[2 * kk + 1][3]);
    }
    Acc o[kNd];
#pragma unroll
    for (int nd = 0; nd < kNd; ++nd) {
      zero(o[nd]);
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) mma(o[nd], pa[kk], transpose8x8(vraw[nd][kk][0]), transpose8x8(vraw[nd][kk][1]));
    }
    TICK(4);

    // ---- merge with the key-half partner through the head's region (its tables are dead once all 4 warps pass B2)
    named_bar(bar_head, 128);  // B2
    float* mine = sEx + (rt * 2 + kh) * kExWarp;
    const float* theirs = sEx + (rt * 2 + (1 - kh)) * kExWarp;
    if (t == 0) {
      mine[g] = mrow[0];
      mine[g + 8] = mrow[1];
      mine[16 + g] = zrow[0];
      mine[16 + g + 8] = zrow[1];
    }
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {  // the partner finalizes n-tiles (1 - kh) * kHalfNd + i
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = kh ? elem(o[i], j) : elem(o[kHalfNd + i], j);
      float* dst = mine + 32 + g * (kHd / 2) + 8 * i + 2 * t;
      *reinterpret_cast<float2*>(dst) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(dst + 8 * (kHd / 2)) = make_float2(v[2], v[3]);
    }
    named_bar(bar_pair, 64);  // B3
    TICK(5);
    float fm[2], fp[2], inv[2];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
      const float mp = theirs[g + 8 * a], zp = theirs[16 + g + 8 * a];
      const float mm = fmaxf(mrow[a], mp);
      fm[a] = __expf(mrow[a] - mm);
      fp[a] = __expf(mp - mm);
      inv[a] = __frcp_rn(zrow[a] * fm[a] + zp * fp[a]);
    }
    constexpr float gate_scale = GATE_SCALE;
#pragma unroll
    for (int i = 0; i < kHalfNd; ++i) {
      float v[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) v[j] = kh ? elem(o[kHalfNd + i], j) : elem(o[i], j);
      const float* src = theirs + 32 + g * (kHd / 2) + 8 * i + 2 * t;
      const float2 p0 = *reinterpret_cast<const float2*>(src), p1 = *reinterpret_cast<const float2*>(src + 8 * (kHd / 2));
      const int d = h * kHd + 8 * (nd0 + i) + 2 * t;
#pragma unroll
      for (int a = 0; a < 2; ++a) {
        const int row = row0 + rt * 16 + g + 8 * a;
        float x0 = (v[2 * a] * fm[a] + (a ? p1.x : p0.x) * fp[a]) * inv[a];
        float x1 = (v[2 * a + 1] * fm[a] + (a ? p1.y : p0.y) * fp[a]) * inv[a];
#if OGATE
        x0 *= 0.5f * gate_scale * two_sigmoid(gl[i][a].x);  // gate_scale * sigmoid
        x1 *= 0.5f * gate_scale * two_sigmoid(gl[i][a].y);
#endif
#if OUT_F16
        *reinterpret_cast<__half2*>(out + ((size_t)sample * 64 + row) * kWidth + d) = __floats2half2_rn(x0, x1);
#else
        char2 code;
        code.x = code_of(x0 * ps[i].x);
        code.y = code_of(x1 * ps[i].y);
        *reinterpret_cast<char2*>(out + ((size_t)sample * 64 + row) * kWidth + d) = code;
#endif
      }
    }
    TICK(6);
  }
}
