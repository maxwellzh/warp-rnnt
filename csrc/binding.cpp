#include <string>
#include <tuple>

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "core.h"

#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK((x).device().is_cuda(), #x " must be located in the CUDA")

#define CHECK_FLOAT(x)                                                         \
  TORCH_CHECK((x).scalar_type() == at::ScalarType::Float,                      \
              #x " must be a Float tensor")

#define CHECK_INT(x)                                                           \
  TORCH_CHECK((x).scalar_type() == at::ScalarType::Int,                        \
              #x " must be a Int tensor")

#define None torch::indexing::None
#define Slice torch::indexing::Slice

std::tuple<at::Tensor, at::Tensor, at::Tensor>
rnnt_loss_simple_fwd(const at::Tensor &f, const at::Tensor &g,
                     const at::Tensor &lf, const at::Tensor &ly) {
  // check contiguous
  CHECK_CONTIGUOUS(f);
  CHECK_CONTIGUOUS(g);
  CHECK_CONTIGUOUS(lf);
  CHECK_CONTIGUOUS(ly);
  // check types
  CHECK_FLOAT(f);
  CHECK_FLOAT(g);
  CHECK_INT(lf);
  CHECK_INT(ly);
  // check device
  CHECK_CUDA(f);
  CHECK_CUDA(g);
  CHECK_CUDA(lf);
  CHECK_CUDA(ly);
  // check shape
  TORCH_CHECK(f.dim() == 3, "f must have 3 dims.")
  TORCH_CHECK(g.dim() == 3, "g must have 3 dims.")
  TORCH_CHECK(f.size(0) == g.size(0), "f and g must have the same dim 0 size.")
  TORCH_CHECK(f.size(2) == g.size(1),
              "f.size(2) != g.size(1), have you run the gather step?")
  TORCH_CHECK(g.size(2) == 2, "g.size(2) != 2, have you run the gather step?")

  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  auto alphas = f.new_empty({N, T, U});
  auto betas = f.new_empty({N, T, U});
  // for thread syncing
  auto counts = torch::zeros({N, U * 2}, lf.options());

  run_warp_rnnt_simple(
      (unsigned int *)counts.data_ptr<int>(), (float *)alphas.data_ptr<float>(),
      (float *)betas.data_ptr<float>(), (const float *)f.data_ptr<float>(),
      (const float *)g.data_ptr<float>(), (const int *)lf.data_ptr<int>(),
      (const int *)ly.data_ptr<int>(), N, T, U);

  // costs = betas[:, 0, 0]
  auto costs = -betas.index({Slice(), 0, 0});
  return std::make_tuple(costs, alphas, betas);
}

at::Tensor rnnt_loss_simple_bwd_f(const at::Tensor &f, const at::Tensor &g,
                                  const at::Tensor &alphas,
                                  const at::Tensor &betas, const at::Tensor &lf,
                                  const at::Tensor &ly) {
  // removed checking since this function won't be called from outside the
  // ... package
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  // zero init takes time.
  auto grads = torch::empty_like(f);

  run_rnnt_simple_fill_grad_f(
      (float *)grads.data_ptr<float>(), (const float *)alphas.data_ptr<float>(),
      (const float *)betas.data_ptr<float>(),
      (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
      (const int *)lf.data_ptr<int>(), (const int *)ly.data_ptr<int>(), N, T,
      U);

  return grads;
}

at::Tensor rnnt_loss_simple_bwd_g(const at::Tensor &f, const at::Tensor &g,
                                  const at::Tensor &alphas,
                                  const at::Tensor &betas, const at::Tensor &lf,
                                  const at::Tensor &ly) {
  // removed checking since this function won't be called from outside the
  // ... package
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  // zero init takes time.
  auto grads = torch::empty_like(g);
  auto counts = torch::zeros({N, T}, g.options());

  run_rnnt_simple_fill_grad_g((float *)grads.data_ptr<float>(),
                              (unsigned int *)counts.data_ptr<float>(),
                              (const float *)alphas.data_ptr<float>(),
                              (const float *)betas.data_ptr<float>(),
                              (const float *)f.data_ptr<float>(),
                              (const float *)g.data_ptr<float>(),
                              (const int *)lf.data_ptr<int>(),
                              (const int *)ly.data_ptr<int>(), N, T, U);

  return grads;
}

std::tuple<at::Tensor, at::Tensor>
rnnt_loss(const at::Tensor &xs, const at::Tensor &ys, const at::Tensor &xn,
          const at::Tensor &yn, const int blank, const float fastemit_lambda) {
  // Check contiguous
  CHECK_CONTIGUOUS(xs);
  CHECK_CONTIGUOUS(ys);
  CHECK_CONTIGUOUS(xn);
  CHECK_CONTIGUOUS(yn);
  // Check types
  CHECK_FLOAT(xs);
  CHECK_INT(ys);
  CHECK_INT(xn);
  CHECK_INT(yn);
  // Check device
  CHECK_CUDA(xs);
  CHECK_CUDA(ys);
  CHECK_CUDA(xn);
  CHECK_CUDA(yn);
  // Check number of dimensions and elements
  TORCH_CHECK(xs.dim() == 4, "xs must have 4 dimensions")
  TORCH_CHECK(xn.numel() == xs.size(0), "xn shape must be equal (N,)")
  TORCH_CHECK(yn.numel() == xs.size(0), "yn shape must be equal (N,)")
  TORCH_CHECK(xs.size(2) == ys.size(1) + 1,
              "ys shape (N, U-1) mismatched with xs (N, T, U, V)")

  const at::cuda::OptionalCUDAGuard device_guard(device_of(xs));

  const auto N = xs.size(0);
  const auto T = xs.size(1);
  const auto U = xs.size(2);
  const auto V = xs.size(3);

  at::Tensor grads = at::zeros_like(xs);

  at::TensorOptions buffer_opts(xs.device());
  at::TensorOptions counts_opts(xs.device());
  at::TensorOptions costs_opts(xs.device());

  counts_opts = counts_opts.dtype(at::ScalarType::Int);
  buffer_opts = buffer_opts.dtype(at::ScalarType::Float);
  costs_opts = costs_opts.dtype(at::ScalarType::Float);

  auto counts_shape = {N, U * 2};
  auto buffer_shape = {N, T, U};
  auto costs_shape = {N};

  torch::Tensor costs = torch::empty(costs_shape, costs_opts);
  at::Tensor counts = at::zeros(counts_shape, counts_opts);
  at::Tensor alphas = at::empty(buffer_shape, buffer_opts);
  at::Tensor betas = at::empty(buffer_shape, buffer_opts);

  if (blank == -1) {

    TORCH_CHECK(V == 2, "xs must have values only for blank and label")

    run_warp_rnnt_gather((unsigned int *)counts.data_ptr<int>(),
                         alphas.data_ptr<float>(), betas.data_ptr<float>(),
                         xs.data_ptr<float>(), grads.data_ptr<float>(),
                         costs.data_ptr<float>(), xn.data_ptr<int>(),
                         yn.data_ptr<int>(), N, T, U, fastemit_lambda);
  } else {
    run_warp_rnnt(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        betas.data_ptr<float>(), ys.data_ptr<int>(), xs.data_ptr<float>(),
        grads.data_ptr<float>(), costs.data_ptr<float>(), xn.data_ptr<int>(),
        yn.data_ptr<int>(), N, T, U, V, blank, fastemit_lambda);
  }

  return std::make_tuple(costs, grads);
}

// return (costs, grad, loc, blank)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rnnt_loss_compact_forward(const torch::Tensor &xs, const torch::Tensor &ys,
                          const torch::Tensor &xn, const torch::Tensor &yn,
                          const int blank, const float fastemit_lambda,
                          const bool require_grad) {
  // Check contiguous
  CHECK_CONTIGUOUS(xs);
  CHECK_CONTIGUOUS(ys);
  CHECK_CONTIGUOUS(xn);
  CHECK_CONTIGUOUS(yn);
  // Check types
  CHECK_FLOAT(xs);
  CHECK_INT(ys);
  CHECK_INT(xn);
  CHECK_INT(yn);
  // Check device
  CHECK_CUDA(xs);
  CHECK_CUDA(ys);
  CHECK_CUDA(xn);
  CHECK_CUDA(yn);
  // Check number of dimensions and elements
  TORCH_CHECK(xs.dim() == 2, "xs must have 2 dimensions")
  TORCH_CHECK(xn.size(0) == yn.size(0), "xn and yn shape must be equal (N,)")
  TORCH_CHECK(ys.numel() == yn.sum().item<int64_t>(),
              "ys shape must be equal to (sum(yn), )")
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xs));

  const auto N = xn.size(0);
  const auto Tm = xn.max().item<int64_t>();     // max of {T_i}
  const auto Um = yn.max().item<int64_t>() + 1; // max of {U_i}
  const auto V = xs.size(1);

  auto memPref =
      (xn * (yn + 1))
          .cumsum(0, at::ScalarType::Int); // count of frames by current batch
  auto labelPref = yn.cumsum(0, at::ScalarType::Int); // copy yn

  int64_t STU = memPref[-1].item<int64_t>();
  TORCH_CHECK(xs.size(0) == STU, "xs shape mismatch with (\\sum{xn*(yn+1)}, )")

  // set begin of memory location of each sequence
  {
    auto cumsumMemPref = memPref.index({Slice(0, -1, None)}).clone();
    auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}).clone();
    memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
  }
  memPref[0] = 0;
  labelPref[0] = 0;

  const auto device = xs.device();
  // the negtive log likelihood
  torch::Tensor costs =
      torch::empty({N}, torch::dtype(torch::kFloat32).device(device));
  //  for maintain the execute status of forward/backward calculation
  torch::Tensor counts = torch::zeros(
      {ys.numel() * 2 + 2 * N}, torch::dtype(torch::kInt32).device(device));
  // forward variable of RNN-T
  torch::Tensor alphas =
      torch::empty({STU}, torch::dtype(torch::kFloat32).device(device));
  // backward variable of RNN-T
  torch::Tensor betas = torch::empty_like(alphas);
  torch::Tensor grads;

  int modified_blank = blank;
  if (!require_grad && modified_blank < 0) {
    // For better efficiency, gather is not required, if we don't need to
    // compute gradients.
    modified_blank = (-1) - blank;
  }

  if (modified_blank < 0) {
    // gather mode
    int real_blank = (-1) - blank;

    torch::Tensor gather_xs =
        torch::empty({STU, 2L}, torch::dtype(torch::kFloat32).device(device));
    torch::Tensor loc =
        torch::zeros({STU}, torch::dtype(torch::kInt64).device(device));

    run_gather(xs.data_ptr<float>(), ys.data_ptr<int>(),
               (unsigned int *)xn.data_ptr<int>(),
               (unsigned int *)yn.data_ptr<int>(), gather_xs.data_ptr<float>(),
               loc.data_ptr<long>(), (unsigned int *)memPref.data_ptr<int>(),
               (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V,
               real_blank);

    grads = torch::zeros_like(gather_xs);

    run_warp_rnnt_compact_gather(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        betas.data_ptr<float>(), gather_xs.data_ptr<float>(),
        grads.data_ptr<float>(), costs.data_ptr<float>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        (unsigned int *)memPref.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, fastemit_lambda);

    return std::make_tuple(costs, grads, loc);
  } else {
    memPref *= V;
    if (require_grad) {

      grads = torch::zeros_like(xs);
      run_warp_rnnt_compact(
          (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
          betas.data_ptr<float>(), (unsigned int *)ys.data_ptr<int>(),
          xs.data_ptr<float>(), grads.data_ptr<float>(),
          costs.data_ptr<float>(), (unsigned int *)xn.data_ptr<int>(),
          (unsigned int *)yn.data_ptr<int>(),
          (unsigned int *)memPref.data_ptr<int>(),
          (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V, blank,
          fastemit_lambda);

      // non-gather mode, only (costs, grad) is useful.
      return std::make_tuple(costs, grads, grads);
    } else {
      run_rnnt_cost_cal_compact(
          (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
          (unsigned int *)ys.data_ptr<int>(), xs.data_ptr<float>(),
          costs.data_ptr<float>(), (unsigned int *)xn.data_ptr<int>(),
          (unsigned int *)yn.data_ptr<int>(),
          (unsigned int *)memPref.data_ptr<int>(),
          (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V,
          modified_blank);

      // non-gather mode, only (costs, ) is useful.
      return std::make_tuple(costs, costs, costs);
    }
  }
}

// return (costs, grad)
std::tuple<torch::Tensor, torch::Tensor>
rnnt_loss_fused_forward(torch::Tensor &xs, const torch::Tensor &ys,
                        const torch::Tensor &xn, const torch::Tensor &yn,
                        const int blank, const bool require_grad) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(xs));
  const auto N = xn.size(0);
  const auto Tm = xn.max().item<int64_t>();     // max of {T_i}
  const auto Um = yn.max().item<int64_t>() + 1; // max of {U_i}
  const auto V = xs.size(1);

  auto memPref =
      (xn * (yn + 1))
          .cumsum(0, at::ScalarType::Int); // count of frames by current batch
  auto labelPref = yn.cumsum(0, at::ScalarType::Int); // copy yn

  int64_t STU = memPref[-1].item<int64_t>();
  TORCH_CHECK(xs.size(0) == STU, "xs shape mismatch with (\\sum{xn*(yn+1)}, )")

  // set begin of memory location of each sequence
  {
    auto cumsumMemPref = memPref.index({Slice(0, -1, None)}).clone();
    auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}).clone();
    memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
    labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
  }
  memPref[0] = 0;
  const auto Offset = memPref * V;
  labelPref[0] = 0;

  // x -= c, c = max(x)
  xs -= xs.max();
  // log_softmax(x) = x - c - log(sum(exp(x-c)))
  const auto summed = torch::logsumexp(xs, -1, true); // shape (STU, )
  xs -= summed;

  const auto device = xs.device();
  // the negtive log likelihood
  torch::Tensor costs =
      torch::empty({N}, torch::dtype(torch::kFloat32).device(device));
  //  for maintain the execute status of forward/backward calculation
  torch::Tensor counts = torch::zeros(
      {ys.numel() * 2 + 2 * N}, torch::dtype(torch::kInt32).device(device));
  // forward variable of RNN-T
  torch::Tensor alphas =
      torch::empty({STU}, torch::dtype(torch::kFloat32).device(device));

  if (require_grad) {
    // backward variable of RNN-T
    torch::Tensor betas = torch::empty_like(alphas);
    torch::Tensor grads = torch::zeros_like(xs);

    run_warp_rnnt_compact(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        betas.data_ptr<float>(), (unsigned int *)ys.data_ptr<int>(),
        xs.data_ptr<float>(), grads.data_ptr<float>(), costs.data_ptr<float>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        (unsigned int *)Offset.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V, blank, 0.0f);

    alphas += betas;
    run_alphabeta_div_prob(alphas.data_ptr<float>(), costs.data_ptr<float>(),
                           (unsigned int *)xn.data_ptr<int>(),
                           (unsigned int *)yn.data_ptr<int>(),
                           (unsigned int *)memPref.data_ptr<int>(), N, Tm, Um);

    xs += alphas.view({STU, 1});
    xs = torch::exp_(xs);
    grads += xs;

    // non gather mode, only (costs, grad) is useful.
    return std::make_tuple(costs, grads);
  } else {
    run_rnnt_cost_cal_compact(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        (unsigned int *)ys.data_ptr<int>(), xs.data_ptr<float>(),
        costs.data_ptr<float>(), (unsigned int *)xn.data_ptr<int>(),
        (unsigned int *)yn.data_ptr<int>(),
        (unsigned int *)Offset.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V, blank);

    return std::make_tuple(costs, costs);
  }
}

torch::Tensor rnnt_loss_compact_backward(const torch::Tensor &grad_cost,
                                         torch::Tensor &grad,
                                         const torch::Tensor &cumSum,
                                         const torch::Tensor &loc, long V,
                                         int blank) {
  // Check contiguous
  CHECK_CONTIGUOUS(grad_cost);
  CHECK_CONTIGUOUS(grad);
  // Check types
  CHECK_FLOAT(grad_cost);
  CHECK_FLOAT(grad);
  // Check device
  CHECK_CUDA(grad_cost);
  CHECK_CUDA(grad);
  // Check number of dimensions and elements
  TORCH_CHECK(grad_cost.dim() == 1, "grad_cost must have 1 dimensions") // (N,)
  TORCH_CHECK(grad.dim() == 2, "grad must have 2 dimensions") // (STU, 2)
  const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_cost));

  const auto N = grad_cost.size(0);
  const auto STU = grad.size(0);

  const auto device = grad_cost.device();

  if (blank < 0) {
    CHECK_CONTIGUOUS(loc);
    TORCH_CHECK(loc.scalar_type() == at::ScalarType::Long,
                "loc must be a Long tensor");
    CHECK_CUDA(loc);
    TORCH_CHECK(grad.size(0) == loc.size(0),
                " grad and loc must be equal in dim=0")

    int real_blank = -1 - blank;

    torch::Tensor scatter_grad =
        torch::zeros({STU, V}, torch::dtype(torch::kFloat32).device(device));

    run_scatter_grad(grad_cost.data_ptr<float>(), grad.data_ptr<float>(),
                     loc.data_ptr<long>(),
                     (unsigned int *)cumSum.data_ptr<int>(),
                     scatter_grad.data_ptr<float>(), STU, N, V, real_blank);

    return scatter_grad;
  } else {

    run_backward_compact(grad_cost.data_ptr<float>(), grad.data_ptr<float>(),
                         (unsigned int *)cumSum.data_ptr<int>(), STU, N, V);

    return grad;
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rnnt_loss", &rnnt_loss,
        "CUDA-Warp RNN-Transducer loss (forward and backward).",
        pybind11::arg("xs"), pybind11::arg("ys"), pybind11::arg("xn"),
        pybind11::arg("yn"), pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0);

  m.def("rnnt_loss_compact_forward", &rnnt_loss_compact_forward,
        "CUDA-Warp RNN-Transducer loss with compact memory layout",
        pybind11::arg("xs"), pybind11::arg("ys"), pybind11::arg("xn"),
        pybind11::arg("yn"), pybind11::arg("blank") = 0,
        pybind11::arg("fastemit_lambda") = 0.0,
        pybind11::arg("require_grad") = true);

  m.def("rnnt_loss_compact_backward", &rnnt_loss_compact_backward,
        "Compact RNN-T loss backward", pybind11::arg("grad_cost"),
        pybind11::arg("grad"), pybind11::arg("cumSum"), pybind11::arg("loc"),
        pybind11::arg("V"), pybind11::arg("blank"));

  m.def("rnnt_loss_fused_forward", &rnnt_loss_fused_forward,
        "CUDA-Warp RNN-Transducer loss with fused operations",
        pybind11::arg("xs"), pybind11::arg("ys"), pybind11::arg("xn"),
        pybind11::arg("yn"), pybind11::arg("blank") = 0,
        pybind11::arg("require_grad") = true);

  m.def("rnnt_loss_simple_fwd", &rnnt_loss_simple_fwd,
        "Simple RNN-T loss foward computing", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("lf"), pybind11::arg("ly"));

  m.def("rnnt_loss_simple_bwd_f", &rnnt_loss_simple_bwd_f,
        "Simple RNN-T loss backward computing for f", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("alphas"), pybind11::arg("betas"),
        pybind11::arg("lf"), pybind11::arg("ly"));

  m.def("rnnt_loss_simple_bwd_g", &rnnt_loss_simple_bwd_g,
        "Simple RNN-T loss backward computing for g", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("alphas"), pybind11::arg("betas"),
        pybind11::arg("lf"), pybind11::arg("ly"));
}
