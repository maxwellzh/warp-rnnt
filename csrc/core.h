#ifndef RNNT_CORE_H
#define RNNT_CORE_H

#include <cuda_runtime.h>
#define CHECK_KERNEL_STAT(s)                                                   \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, #s " error: %s\n", cudaGetErrorString(err));             \
      exit(-1);                                                                \
    }                                                                          \
  }

void run_warp_rnnt_compact_gather(
    unsigned int *counts, float *alphas, float *betas, const float *log_probs,
    float *grads, float *costs, const unsigned int *xn, const unsigned int *yn,
    const unsigned int *memPref, const unsigned int *labelPref, unsigned int N,
    unsigned int T, unsigned int U, float fastemit_lambda);

void run_rnnt_cost_cal_compact(unsigned int *counts, float *alphas,
                               const unsigned int *labels,
                               const float *log_probs, float *costs,
                               const unsigned int *xn, const unsigned int *yn,
                               const unsigned int *memPref,
                               const unsigned int *labelPref, unsigned int N,
                               unsigned int Tm, unsigned int Um, unsigned int V,
                               unsigned int blank);

void run_warp_rnnt_compact(unsigned int *counts, float *alphas, float *betas,
                           const unsigned int *labels, const float *log_probs,
                           float *grads, float *costs, const unsigned int *xn,
                           const unsigned int *yn, const unsigned int *memPref,
                           const unsigned int *labelPref, unsigned int N,
                           unsigned int Tm, unsigned int Um, unsigned int V,
                           unsigned int blank, float fastemit_lambda);

void run_alphabeta_div_prob(float *alphabetas, const float *costs,
                            const unsigned int *xn, const unsigned int *yn,
                            const unsigned int *memPref, unsigned int N,
                            unsigned int Tm, unsigned int Um);

void run_warp_rnnt_gather(unsigned int *counts, float *alphas, float *betas,
                          const float *log_probs, float *grads, float *costs,
                          const int *xn, const int *yn, int N, int T, int U,
                          float fastemit_lambda);

void run_warp_rnnt(unsigned int *counts, float *alphas, float *betas,
                   const int *labels, const float *log_probs, float *grads,
                   float *costs, const int *xn, const int *yn, int N, int T,
                   int U, int V, int blank, float fastemit_lambda);

void run_gather(const float *xs, const int *ys, const unsigned int *xn,
                const unsigned int *yn, float *gather_xs, long *loc,
                const unsigned int *memPref, const unsigned int *labelPref,
                unsigned int N, unsigned int T, unsigned int U, unsigned int V,
                unsigned int blank);

void run_scatter_grad(const float *grad_cost, const float *gather_grad,
                      const long *loc, const unsigned int *cumSum,
                      float *scatter_grad, unsigned int STU, unsigned int N,
                      unsigned int V, unsigned int blank);

void run_backward_compact(const float *grad_cost, float *grad,
                          const unsigned int *cumSum, unsigned int STU,
                          unsigned N, unsigned int V);

void run_warp_rnnt_simple(unsigned int *counts, volatile float *alphas,
                          volatile float *betas, const float *f, const float *g,
                          const int *lf, const int *ly, unsigned int N,
                          unsigned int T, unsigned int U);

void run_rnnt_simple_fill_grad_f(volatile float *grad_f, const float *alphas,
                                 const float *betas, const float *f,
                                 const float *g, const int *lf, const int *ly,
                                 unsigned int N, unsigned int T,
                                 unsigned int U);

void run_rnnt_simple_fill_grad_g(volatile float *grad_f, unsigned int *counts,
                                 const float *alphas, const float *betas,
                                 const float *f, const float *g, const int *lf,
                                 const int *ly, unsigned int N, unsigned int T,
                                 unsigned int U);
#endif