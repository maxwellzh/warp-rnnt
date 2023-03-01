#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#define CHECK_KERNEL_STAT(s)                                                   \
    {                                                                          \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, #s " error: %s\n", cudaGetErrorString(err));       \
            exit(-1);                                                          \
        }                                                                      \
    }

void run_warp_rnnt(unsigned int *counts, float *alphas, float *betas,
                   const int *labels, const float *log_probs, float *grads,
                   float *costs, const int *xn, const int *yn, int N, int T,
                   int U, int V, int blank, float fastemit_lambda);
void run_warp_rnnt_gather(unsigned int *counts, float *alphas, float *betas,
                          const float *log_probs, float *grads, float *costs,
                          const int *xn, const int *yn, int N, int T, int U,
                          float fastemit_lambda);

void run_gather_for_compact(const float *xs, const int *ys,
                            const unsigned int *xn, const unsigned int *yn,
                            float *gather_xs, long *loc,
                            const unsigned int *memPref,
                            const unsigned int *labelPref, unsigned int N,
                            unsigned int T, unsigned int U, unsigned int V,
                            unsigned int blank);
void run_warp_rnnt_compact(unsigned int *counts, float *alphas, float *betas,
                           const float *log_probs, float *grads, float *costs,
                           const unsigned int *xn, const unsigned int *yn,
                           const unsigned int *memPref,
                           const unsigned int *labelPref, unsigned int N,
                           unsigned int T, unsigned int U,
                           float fastemit_lambda, bool requires_grad);
void run_scatter_grad_compact(const float *grad_cost, const float *gather_grad,
                      const long *loc, const unsigned int *cumSum,
                      float *scatter_grad, unsigned int STU, unsigned int N,
                      unsigned int V, unsigned int blank);

void run_warp_rnnt_simple(unsigned int *counts, volatile float *alphas,
                          volatile float *betas, const float *f, const float *g,
                          const int *lf, const int *ly, unsigned int N,
                          unsigned int T, unsigned int U);
void run_warp_rnnt_simple(unsigned int *counts, volatile float *alphas,
                          volatile float *betas, const float *f, const float *g,
                          const float *den, const int *lf, const int *ly,
                          unsigned int N, unsigned int T, unsigned int U);

void run_rnnt_simple_fill_grad_f(volatile float *grad_f, const float *alphas,
                                 const float *betas, const float *f,
                                 const float *g, const int *lf, const int *ly,
                                 unsigned int N, unsigned int T,
                                 unsigned int U);
void run_rnnt_simple_fill_grad_f(volatile float *grad_f, const float *alphas,
                                 const float *betas, const float *f,
                                 const float *g, const float *den,
                                 const int *lf, const int *ly, unsigned int N,
                                 unsigned int T, unsigned int U);

void run_rnnt_simple_fill_grad_g(volatile float *grad_g, unsigned int *counts,
                                 const float *alphas, const float *betas,
                                 const float *f, const float *g, const int *lf,
                                 const int *ly, unsigned int N, unsigned int T,
                                 unsigned int U);
void run_rnnt_simple_fill_grad_g(volatile float *grad_g, unsigned int *counts,
                                 const float *alphas, const float *betas,
                                 const float *f, const float *g,
                                 const float *den, const int *lf, const int *ly,
                                 unsigned int N, unsigned int T,
                                 unsigned int U);

void run_rnnt_simple_fill_grad_den(volatile float *grad_den,
                                   const float *alphas, const float *betas,
                                   const float *f, const float *g,
                                   const float *den, const int *lf,
                                   const int *ly, unsigned int N,
                                   unsigned int T, unsigned int U);

void log_matmul_cuda_impl(const at::Tensor &lhs_, const at::Tensor &rhs_,
                          const at::Tensor &out);
