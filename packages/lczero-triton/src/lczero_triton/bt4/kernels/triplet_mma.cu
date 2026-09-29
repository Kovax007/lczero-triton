// EGT2 triplet operator on the tensor cores (backend lane 09-29, `triplet_mma.py`): form "fused"'s math (`triplet_site.py`), one CTA per
// (sample, direction, triplet head), reading the sample's e_hat ONCE from its FP16 cell-major copy (a cell's 16 channels, 32 bytes).
//   phase 1: the head's six maps (logit, door, v0..v3) are linear in the state -- one m16n8k16 MMA per 16 cells, K = the 16 channels,
//            N = the maps (weights split hi + lo FP16, so only the state's FP16 rounding remains); the rms from the same registers
//            (W rms(x) = inv * (W x)). Logit and sigmoid(door) go to shared memory in FP32, oriented so that the softmax runs over the
//            last axis in both directions (the outward branch transposed); the values in FP16, oriented as the contraction's B operand.
//   phase 2: 16 warps = 4 row tiles x 4 dot channels: a row softmax x sigmoid(door) in FP32 (quad shuffles), A kept as FP16 MMA
//            fragments, va[d] = A V[d] on the tensor cores with FP32 accumulation, stored FP32 into va's layout ([B, 32, 64, 64],
//            channel direction * 16 + head * 4 + d), which `triplet_site`'s `out` stage reads unchanged.
// Arithmetic class: the served "fused" form with `state_f16` (e_hat rounded to FP16 once) and `dot="fp16"` (A and V rounded to FP16).
// Compile-time: SAMPLES (the batch), CONTRACTION (0 path, 1 ag), SMEM_BYTES (checked against the layout), VA_F16 (va stored FP16:
// half the bytes of the kernel's write and of `out`'s read).
#include <cuda_fp16.h>
#include <cstdint>

#ifndef SAMPLES
#error "SAMPLES (the batch) is a compile-time constant"
#endif
#ifndef CONTRACTION
#define CONTRACTION 0
#endif
#ifndef VA_F16
#define VA_F16 0
#endif

namespace {

constexpr int kThreads = 512;
constexpr int kHeads = 4, kDots = 4, kStates = 16;
constexpr int kLPitch = 65;                        // FP32 rows of 64 (+1 word: conflict-free transposed stores)
constexpr int kVPitch = 72;                        // FP16 rows of 64 (+8 halves: conflict-free B-fragment loads)
constexpr int kLBytes = 64 * kLPitch * 4;
constexpr int kVBytes = kDots * 64 * kVPitch * 2;
constexpr int kAPitch = 72;                        // FP16 rows of A (+8 halves)
constexpr int kABytes = 64 * kAPitch * 2;
constexpr int kSmem = 2 * kLBytes + kVBytes + kABytes;  // logit, sigmoid(door), values, A
constexpr float kEpsilon = 1e-6f;
#ifdef SMEM_BYTES
static_assert(kSmem == SMEM_BYTES, "SMEM_BYTES disagrees with the kernel's layout");
#endif

__device__ __forceinline__ uint32_t ld32(const __half* p) { return __ldg(reinterpret_cast<const unsigned int*>(p)); }
__device__ __forceinline__ uint32_t pack(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 unpack(uint32_t x) { return __half22float2(*reinterpret_cast<const __half2*>(&x)); }
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float sigmoid(float x) {  // 1 / (1 + e^-x) = (1 + tanh(x / 2)) / 2 (B9's gated form)
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(0.5f * x));
  return fmaf(0.5f, y, 0.5f);
}
__device__ __forceinline__ float squares(uint32_t x) {
  const float2 v = unpack(x);
  return v.x * v.x + v.y * v.y;
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1) triplet_mma(
    void* __restrict__ va_ptr, const __half* __restrict__ copy, const float* __restrict__ value_weight,
    const float* __restrict__ gate_weight, const float* __restrict__ gate_bias) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* sL = reinterpret_cast<float*>(smem);                         // logit, [i][k]: softmax over k
  float* sG = reinterpret_cast<float*>(smem + kLBytes);               // sigmoid(door), [i][k]
  __half* sV = reinterpret_cast<__half*>(smem + 2 * kLBytes);         // values as B: [d][j][k]
  __half* sA = reinterpret_cast<__half*>(smem + 2 * kLBytes + kVBytes); // A = softmax x sigmoid, FP16 [i][k]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int unit = blockIdx.x, head = unit & 3, direction = (unit >> 2) & 1, sample = unit >> 3;
#ifdef CONTROL_NO_SWAP  // the fidelity gate's CONTROL (must fail): V never transposed (the contraction order ignored)
  const bool swap = false;
#else
  const bool swap = ((direction + CONTRACTION) & 1) != 0;
#endif

  // ---- the head's weights as the maps' B fragments (n = map g: logit, door, v0..v3, two zero columns), hi + lo FP16
  const int gate_row = direction * 2 * kHeads + head, door_row = gate_row + kHeads;
  const int value_row = (direction * kHeads + head) * kDots;
  const float* wrow = g == 0 ? gate_weight + gate_row * kStates
                    : g == 1 ? gate_weight + door_row * kStates
                    : g < 6  ? value_weight + (value_row + g - 2) * kStates : nullptr;
  uint32_t bh[2], bl[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const float w0 = wrow ? __ldg(wrow + 2 * t + 8 * r) : 0.f, w1 = wrow ? __ldg(wrow + 2 * t + 8 * r + 1) : 0.f;
    bh[r] = pack(w0, w1);
    const float2 h = unpack(bh[r]);
    bl[r] = pack(w0 - h.x, w1 - h.y);
  }
#ifdef CONTROL_NO_DOOR_BIAS  // the fidelity gate's CONTROL (must fail): the door without its bias
  const float logit_bias = __ldg(gate_bias + gate_row), door_bias = 0.f;
#else
  const float logit_bias = __ldg(gate_bias + gate_row), door_bias = __ldg(gate_bias + door_row);
#endif

  // ---- phase 1: 256 tiles of 16 cells; warp w takes tiles w, w + 16, ... (P1_UNROLL in flight)
#ifndef P1_UNROLL
#define P1_UNROLL 4
#endif
  const __half* src = copy + (size_t)sample * 4096 * kStates;
#pragma unroll 1
  for (int base = warp; base < 256; base += 16 * P1_UNROLL) {
    uint32_t a[P1_UNROLL][4];
#pragma unroll
    for (int u = 0; u < P1_UNROLL; ++u) {
      const int c0 = (base + 16 * u) * 16 + g;
#ifdef DBG_NO_LOAD  // pricing switches (results wrong): drop one part
      a[u][0] = a[u][1] = a[u][2] = a[u][3] = pack(0.5f + 1e-3f * c0, 0.25f);
#else
      a[u][0] = ld32(src + c0 * kStates + 2 * t);
      a[u][1] = ld32(src + (c0 + 8) * kStates + 2 * t);
      a[u][2] = ld32(src + c0 * kStates + 2 * t + 8);
      a[u][3] = ld32(src + (c0 + 8) * kStates + 2 * t + 8);
#endif
    }
#pragma unroll
    for (int u = 0; u < P1_UNROLL; ++u) {
      float s0 = squares(a[u][0]) + squares(a[u][2]), s1 = squares(a[u][1]) + squares(a[u][3]);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 1);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 1);
      s0 += __shfl_xor_sync(0xffffffffu, s0, 2);
      s1 += __shfl_xor_sync(0xffffffffu, s1, 2);
      const float inv0 = rsqrtf(fmaf(s0, 1.f / kStates, kEpsilon)), inv1 = rsqrtf(fmaf(s1, 1.f / kStates, kEpsilon));
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      mma(c, a[u], bh[0], bh[1]);
      mma(c, a[u], bl[0], bl[1]);
      // c[0], c[1]: cell c0, maps 2t, 2t + 1; c[2], c[3]: cell c0 + 8
      const int tile = base + 16 * u, row = tile >> 2;  // a tile is 16 cells of one grid row (i, j) = (row, col)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int col = ((tile & 3) << 4) + g + 8 * half;
        const float inv = half ? inv1 : inv0;
        const float x0 = c[2 * half] * inv, x1 = c[2 * half + 1] * inv;
        if (t == 0) {
          // A[i][k]: inward A = softmax_k(e_in[i, k]); outward A[i][k] = a_out[k, i], its tile transposed
          const int at = direction ? col * kLPitch + row : row * kLPitch + col;
          sL[at] = x0 + logit_bias;
          sG[at] = sigmoid(x1 + door_bias);
        } else if (t < 3) {
          // V[d] as B = [j][k]: Vs[k][j] = V[k][j] (no swap) -> stored at [col][row]; V[j][k] (swap) -> at [row][col]
          const int at = swap ? row * kVPitch + col : col * kVPitch + row;
          const int d = 2 * (t - 1);
          sV[d * 64 * kVPitch + at] = __float2half_rn(x0);
          sV[(d + 1) * 64 * kVPitch + at] = __float2half_rn(x1);
        }
      }
    }
  }
  __syncthreads();
#ifdef DBG_NO_P2
  if (sL[tid] != 12345.f) return;
#endif

  // ---- phase 2a: A = softmax_k(L) x sigmoid(door), once per row: warp w takes rows 4w .. 4w + 3, lane l keys 2l, 2l + 1
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    const int row = 4 * warp + rr;
    const float2 x = make_float2(sL[row * kLPitch + 2 * lane], sL[row * kLPitch + 2 * lane + 1]);
    float mx = fmaxf(x.x, x.y);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    const float p0 = __expf(x.x - mx), p1 = __expf(x.y - mx);
    float z = p0 + p1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
    const float rz = 1.f / z;
    *reinterpret_cast<uint32_t*>(sA + row * kAPitch + 2 * lane) =
        pack(p0 * rz * sG[row * kLPitch + 2 * lane], p1 * rz * sG[row * kLPitch + 2 * lane + 1]);
  }
  __syncthreads();

  // ---- phase 2b: warp = (row tile, dot channel): A fragments from sA (4 k-steps), va[d] = A V[d] over 8 n-tiles
  const int mt = warp & 3, d = warp >> 2;
  const int r0 = 16 * mt + g, r1 = r0 + 8;
  uint32_t af[4][4];
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
    af[ks][0] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t);
    af[ks][1] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t);
    af[ks][2] = *reinterpret_cast<const uint32_t*>(sA + r0 * kAPitch + 16 * ks + 2 * t + 8);
    af[ks][3] = *reinterpret_cast<const uint32_t*>(sA + r1 * kAPitch + 16 * ks + 2 * t + 8);
  }
  const __half* vd = sV + d * 64 * kVPitch;
  const size_t plane = ((size_t)sample * 2 * kStates + direction * kStates + head * kDots + d) * 4096;
#if VA_F16
  // FP16 va: per group of 4 n-tiles, a 4 x 4 transpose across the quad gives lane t n-tile 4q + t's 8 keys, one 16-byte store
  // per row (a row's 32 keys = 64 contiguous bytes per store instruction)
  __half* out = static_cast<__half*>(va_ptr) + plane;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    uint32_t p[2][4];  // [row r0 / r1][n-tile 4q + n]
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      float c[4] = {0.f, 0.f, 0.f, 0.f};
      const __half* brow = vd + (8 * (4 * q + n) + g) * kVPitch + 2 * t;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks + 8);
        mma(c, af[ks], b0, b1);
      }
      p[0][n] = pack(c[0], c[1]);
      p[1][n] = pack(c[2], c[3]);
    }
    const bool b1 = t & 2, b0 = t & 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      uint32_t x0 = __shfl_xor_sync(0xffffffffu, b1 ? p[r][0] : p[r][2], 2);
      uint32_t x1 = __shfl_xor_sync(0xffffffffu, b1 ? p[r][1] : p[r][3], 2);
      if (b1) { p[r][0] = x0; p[r][1] = x1; } else { p[r][2] = x0; p[r][3] = x1; }
      x0 = __shfl_xor_sync(0xffffffffu, b0 ? p[r][0] : p[r][1], 1);
      x1 = __shfl_xor_sync(0xffffffffu, b0 ? p[r][2] : p[r][3], 1);
      if (b0) { p[r][0] = x0; p[r][2] = x1; } else { p[r][1] = x0; p[r][3] = x1; }
#ifndef DBG_NO_STORE
      *reinterpret_cast<uint4*>(out + (r ? r1 : r0) * 64 + 32 * q + 8 * t) = make_uint4(p[r][0], p[r][1], p[r][2], p[r][3]);
#endif
    }
  }
#else
  float* out = static_cast<float*>(va_ptr) + plane;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    const __half* brow = vd + (8 * nt + g) * kVPitch + 2 * t;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      const uint32_t b0 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks);
      const uint32_t b1 = *reinterpret_cast<const uint32_t*>(brow + 16 * ks + 8);
      mma(c, af[ks], b0, b1);
    }
#ifdef DBG_NO_STORE
    if (c[0] == 12345.f) *reinterpret_cast<float2*>(smem) = make_float2(c[1], c[2] + c[3]);
    continue;
#endif
    *reinterpret_cast<float2*>(out + r0 * 64 + 8 * nt + 2 * t) = make_float2(c[0], c[1]);
    *reinterpret_cast<float2*>(out + r1 * 64 + 8 * nt + 2 * t) = make_float2(c[2], c[3]);
  }
#endif
}
