#include "core.h"

#include <algorithm>
#include <assert.h>
#include <cuda_runtime_api.h>
#include <device_atomic_functions.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define W 32
#define G 1024
#define B 256

#define IDX3(n, t, u, D1, D2) (n) * (D1) * (D2) + (t) * (D2) + (u)

#define LOG_PROB_BLANK(n, t, u)                                                \
  (f[IDX3(n, t, 0, T, U)] + g[IDX3(n, u, 0, U, 2)] - den[IDX3(n, t, u, T, U)])

#define LOG_PROB_Y(n, t, u)                                                    \
  (f[IDX3(n, t, (u) + 1, T, U)] + g[IDX3(n, u, 1, U, 2)] -                     \
   den[IDX3(n, t, u, T, U)])

__forceinline__ __device__ static float logaddexpf(float a, float b) {
  float const tmp = a - b;

  if (a == b)
    return (float)(a + M_LN2);

  if (tmp > 0)
    return a + log1pf(expf(-tmp));
  else if (tmp <= 0)
    return b + log1pf(expf(tmp));
  // in case of overflow
  return tmp;
}

__device__ void kernel_warp_alphas(unsigned int *counts, volatile float *alphas,
                                   const float *f, const float *g,
                                   const float *den, const int *lf,
                                   const int *ly, unsigned int T,
                                   unsigned int U) {

  unsigned int d = threadIdx.x;
  unsigned int gorin = blockIdx.x;
  unsigned int u = blockIdx.y + 1;
  unsigned int n = blockIdx.z;
  unsigned int p = gorin * W;
  unsigned int t = p + d + 1;

  if (t > lf[n] || u > ly[n] + 1)
    return;

  unsigned int actual_t = lf[n];
  unsigned int actual_u = ly[n] + 1;

  unsigned int *lock = counts + n * U * 2 + blockIdx.y;

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    alphas[IDX3(n, 0, 0, T, U)] = 0.0f;
  }

  if (blockIdx.x > 0) {
    // Wait previous row
    do {
    } while (atomicAdd(lock, 0) < gorin);
  }

  if (blockIdx.y > 0) {
    // Wait previous column
    do {
    } while (atomicAdd(lock - 1, 0) <= gorin);
  }

  if (blockIdx.x == 0 && u < actual_u) {
    // Compute initial row value (t=0, :)
    alphas[IDX3(n, 0, u, T, U)] =
        alphas[IDX3(n, 0, u - 1, T, U)] + LOG_PROB_Y(n, 0, u - 1);
  }

  if (blockIdx.y == 0 && t < actual_t) {
    // Compute initial column with local scan algorithm (:, u=0)
    float a;
    float b = LOG_PROB_BLANK(n, t - 1, 0);

#pragma unroll
    for (unsigned int i = 1; i < W; i *= 2) {
      a = __shfl_up_sync(0xffffffff, b, i);
      if (i <= d) {
        b += a;
      }
    }

    alphas[IDX3(n, t, 0, T, U)] = alphas[IDX3(n, p, 0, T, U)] + b;
  }

  if (t < actual_t && u < actual_u) {
    // Ready to compute alphas(t, u)
    float bias = LOG_PROB_BLANK(n, t - 1, u);
    float skip = alphas[IDX3(n, p, u, T, U)] + bias;
    float emit = alphas[IDX3(n, t, u - 1, T, U)] + LOG_PROB_Y(n, t, u - 1);

    float r = logaddexpf(skip, emit);
    float output = r;

    for (unsigned int i = 1; i < W; i++) {
      r = __shfl_up_sync(0xffffffff, r, 1);
      if (i == d) {
        r = logaddexpf(r + bias, emit);
        output = r;
      }
    }

    alphas[IDX3(n, t, u, T, U)] = output;
  }

  if (d == 0) {
    // https://stackoverflow.com/a/5233737
    __threadfence();
    atomicAdd(lock, 1);
  }
}

__device__ void kernel_warp_betas(unsigned int *counts, volatile float *betas,
                                  const float *f, const float *g,
                                  const float *den, const int *lf,
                                  const int *ly, unsigned int T,
                                  unsigned int U) {
  unsigned int d = threadIdx.x;
  unsigned int gorin = blockIdx.x;
  unsigned int u = blockIdx.y + 1;
  unsigned int n = blockIdx.z;
  unsigned int p = gorin * W;
  unsigned int t = p + d + 1;

  if (t > lf[n] || u > ly[n] + 1)
    return;

  unsigned int actual_t = lf[n];
  unsigned int T1 = actual_t - 1;
  unsigned int U1 = ly[n];
  unsigned int actual_u = U1 + 1;
  unsigned int *lock = counts + n * U * 2 + U + blockIdx.y;

  if (blockIdx.x == 0 && blockIdx.y == 0) {
    betas[IDX3(n, T1, U1, T, U)] = LOG_PROB_BLANK(n, T1, U1);
  }

  if (blockIdx.x > 0) {
    // Wait previous row
    do {
    } while (atomicAdd(lock, 0) < gorin);
  }

  if (blockIdx.y > 0) {
    // Wait previous column
    do {
    } while (atomicAdd(lock - 1, 0) <= gorin);
  }

  if (blockIdx.x == 0 && u < actual_u) {
    // Compute last row value
    betas[IDX3(n, T1, U1 - u, T, U)] =
        betas[IDX3(n, T1, U1 - u + 1, T, U)] + LOG_PROB_Y(n, T1, U1 - u);
  }

  if (blockIdx.y == 0 && t < actual_t) {
    // Compute last column with local scan algorithm
    float a;
    float b = LOG_PROB_BLANK(n, T1 - t, U1);

#pragma unroll
    for (unsigned int i = 1; i < W; i *= 2) {
      a = __shfl_up_sync(0xffffffff, b, i);
      if (i <= d) {
        b += a;
      }
    }

    betas[IDX3(n, T1 - t, U1, T, U)] = betas[IDX3(n, T1 - p, U1, T, U)] + b;
  }

  if (t < actual_t && u < actual_u) {
    // Ready to compute betas(T1-t, U1-u)
    float bias = LOG_PROB_BLANK(n, T1 - t, U1 - u);
    float skip = betas[IDX3(n, T1 - p, U1 - u, T, U)] + bias;
    float emit = betas[IDX3(n, T1 - t, U1 - u + 1, T, U)] +
                 LOG_PROB_Y(n, T1 - t, U1 - u);

    float r = logaddexpf(skip, emit);
    float output = r;

    for (unsigned int i = 1; i < W; i++) {
      r = __shfl_up_sync(0xffffffff, r, 1);
      if (i == d) {
        r = logaddexpf(r + bias, emit);
        output = r;
      }
    }

    betas[IDX3(n, T1 - t, U1 - u, T, U)] = output;
  }

  if (d == 0) {
    // https://stackoverflow.com/a/5233737
    __threadfence();
    atomicAdd(lock, 1);
  }
}

__global__ void kernel_warp(unsigned int *counts, volatile float *alphas,
                            volatile float *betas, const float *f,
                            const float *g, const float *den, const int *lf,
                            const int *ly, unsigned int T, unsigned int U) {
  if (threadIdx.y == 0) {
    kernel_warp_alphas(counts, alphas, f, g, den, lf, ly, T, U);
  } else if (threadIdx.y == 1) {
    kernel_warp_betas(counts, betas, f, g, den, lf, ly, T, U);
  }
}

void run_warp_rnnt_simple(unsigned int *counts, volatile float *alphas,
                          volatile float *betas, const float *f, const float *g,
                          const float *den, const int *lf, const int *ly,
                          unsigned int N, unsigned int T, unsigned int U) {

  dim3 threads(W, 2);
  dim3 blocks1((T + W - 1) / W, U, N);
  kernel_warp<<<blocks1, threads>>>(counts, alphas, betas, f, g, den, lf, ly, T,
                                    U);
  CHECK_KERNEL_STAT("rnnt_loss_simple computing alpha/beta")
  return;
}

__global__ void kernel_warp_grad_f_label(volatile float *grad_f,
                                         const float *alphas,
                                         const float *betas, const float *f,
                                         const float *g, const float *den,
                                         const int *lf, const int *ly,
                                         unsigned int T, unsigned int U) {
  unsigned int n = blockIdx.z;
  unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int u = blockIdx.y * blockDim.y + threadIdx.y;

  if (t >= T || u >= U)
    return;

  if (t >= lf[n] || u > ly[n]) {
    // zero the paddings
    grad_f[IDX3(n, t, u, T, U)] = 0.0f;
    return;
  }

  grad_f += IDX3(n, t, u, T, U);
  if (u == 0) {
    // f[:, :, 0] is reserved for blank
    *grad_f = -1.0f;
  } else {
    // computing grad(n, t, u) for label
    u -= 1;
    *grad_f =
        -expf(LOG_PROB_Y(n, t, u) + alphas[IDX3(n, t, u, T, U)] +
              betas[IDX3(n, t, u + 1, T, U)] - betas[IDX3(n, 0, 0, T, U)]);
  }
}

void run_rnnt_simple_fill_grad_f(volatile float *grad_f, const float *alphas,
                                 const float *betas, const float *f,
                                 const float *g, const float *den,
                                 const int *lf, const int *ly, unsigned int N,
                                 unsigned int T, unsigned int U) {
  dim3 threads(W, W);
  dim3 blocks((T + W - 1) / W, (U + W - 1) / W, N);
  kernel_warp_grad_f_label<<<blocks, threads>>>(grad_f, alphas, betas, f, g,
                                                den, lf, ly, T, U);
  CHECK_KERNEL_STAT("rnnt simple loss computing gradients for f labels")
}

__global__ void kernel_warp_grad_g_blank(
    volatile float *grad_g, unsigned int *counts, const float *alphas,
    const float *betas, const float *f, const float *g, const float *den,
    const int *lf, const int *ly, unsigned int T, unsigned int U) {

  unsigned int n = blockIdx.z;
  unsigned int d = threadIdx.x;
  unsigned int gorin = blockIdx.x;
  unsigned int t = gorin * W + d;
  unsigned int u = blockIdx.y * blockDim.y + threadIdx.y;

  unsigned int accum_t = lf[n] - 1;
  if (t >= accum_t || u >= U)
    return;

  grad_g += IDX3(n, u, 0, U, 2);
  if (u > ly[n]) {
    if (t == 0) {
      *(grad_g) = 0.0f;
      *(grad_g + 1) = 0.0f;
    }
    return;
  }

  unsigned int *lock = counts + n * U + u;

  float tmp;
  float r = LOG_PROB_BLANK(n, t, u) + alphas[IDX3(n, t, u, T, U)] +
            betas[IDX3(n, t + 1, u, T, U)] - betas[IDX3(n, 0, 0, T, U)];
  bool non_last_thread = (gorin * W + W < lf[n]);

  // local scan, sum up results in current thread and store at d=0
#pragma unroll
  for (unsigned int i = 1; i < W; i *= 2) {
    tmp = __shfl_down_sync(0xffffffff, r, i);
    if ((d + i) < W && (non_last_thread || (t + i) < accum_t)) {
      r = logaddexpf(tmp, r);
    }
  }
  if (d > 0)
    return;

  if (gorin == 0) {
    *grad_g = r;
  } else {
    // Wait previous thread, accumulate all threads
    do {
    } while (atomicAdd(lock, 0) < gorin);
    *grad_g = logaddexpf(r, *grad_g);
  }
  // compute -exp(s/P) at last thread
  if (!non_last_thread) {
    *grad_g = -expf(*grad_g);

    // addup blank(n, T, U) (which directs to the final state)
    if (u == ly[n]) {
      *grad_g -= 1.0f;
    } else {
      *(grad_g + 1) = -1.0f;
    }
  }

  // https://stackoverflow.com/a/5233737
  __threadfence();
  atomicAdd(lock, 1);
}

void run_rnnt_simple_fill_grad_g(volatile float *grad_f, unsigned int *counts,
                                 const float *alphas, const float *betas,
                                 const float *f, const float *g,
                                 const float *den, const int *lf, const int *ly,
                                 unsigned int N, unsigned int T,
                                 unsigned int U) {
  dim3 threads(W, W);
  dim3 blocks((T + W - 1) / W, (U + W - 1) / W, N);
  kernel_warp_grad_g_blank<<<blocks, threads>>>(grad_f, counts, alphas, betas,
                                                f, g, den, lf, ly, T, U);
  CHECK_KERNEL_STAT("rnnt simple loss computing gradients for g blank")
}

__global__ void kernel_warp_grad_den(volatile float *grad_den,
                                     const float *alphas, const float *betas,
                                     const float *f, const float *g,
                                     const float *den, const int *lf,
                                     const int *ly, unsigned int N,
                                     unsigned int T, unsigned int U) {
  unsigned int n = blockIdx.z;
  unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int u = blockIdx.y * blockDim.y + threadIdx.y;

  if (t >= T || u >= U)
    return;

  grad_den += IDX3(n, t, u, T, U);
  if (t >= lf[n] || u > ly[n]) {
    // zero the paddings
    *grad_den = 0.0f;
    return;
  }

  float p_zero;
  float p_label;

  if (t == lf[n] - 1) {
    if (u == ly[n])
      p_zero = LOG_PROB_BLANK(n, t, u);
    else
      p_zero = -1.0f / 0.0f;
  } else {
    p_zero = LOG_PROB_BLANK(n, t, u) + betas[IDX3(n, t + 1, u, T, U)];
  }

  if (u == ly[n]) {
    p_label = -1.0f / 0.0f;
  } else {
    p_label = LOG_PROB_Y(n, t, u) + betas[IDX3(n, t, u + 1, T, U)];
  }

  *grad_den = expf(alphas[IDX3(n, t, u, T, U)] - betas[IDX3(n, 0, 0, T, U)] +
                   logaddexpf(p_zero, p_label));
}

void run_rnnt_simple_fill_grad_den(volatile float *grad_den,
                                   const float *alphas, const float *betas,
                                   const float *f, const float *g,
                                   const float *den, const int *lf,
                                   const int *ly, unsigned int N,
                                   unsigned int T, unsigned int U) {
  dim3 threads(W, W);
  dim3 blocks((T + W - 1) / W, (U + W - 1) / W, N);
  kernel_warp_grad_den<<<blocks, threads>>>(grad_den, alphas, betas, f, g, den,
                                            lf, ly, N, T, U);
  CHECK_KERNEL_STAT("rnnt simple loss computing gradients for denonimator.")
}