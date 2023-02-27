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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rnnt_loss_simple_fwd(const torch::Tensor &f, const torch::Tensor &g,
                     const torch::Tensor &den, const torch::Tensor &lf,
                     const torch::Tensor &ly) {
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

  if (den.dim() == 3) {
    // normalzied cost
    CHECK_CONTIGUOUS(den);
    CHECK_FLOAT(den);
    CHECK_CUDA(den);
  }
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  auto alphas = f.new_empty({N, T, U});
  auto betas = f.new_empty({N, T, U});
  // for thread syncing
  auto counts = torch::zeros({N, U * 2}, lf.options());

  if (den.dim() == 3) {
    run_warp_rnnt_simple(
        (unsigned int *)counts.data_ptr<int>(),
        (float *)alphas.data_ptr<float>(), (float *)betas.data_ptr<float>(),
        (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
        (const float *)den.data_ptr<float>(), (const int *)lf.data_ptr<int>(),
        (const int *)ly.data_ptr<int>(), N, T, U);
  } else {
    run_warp_rnnt_simple(
        (unsigned int *)counts.data_ptr<int>(),
        (float *)alphas.data_ptr<float>(), (float *)betas.data_ptr<float>(),
        (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
        (const int *)lf.data_ptr<int>(), (const int *)ly.data_ptr<int>(), N, T,
        U);
  }
  // costs = betas[:, 0, 0]
  auto costs = -betas.index({Slice(), 0, 0});
  return std::make_tuple(costs, alphas, betas);
}

torch::Tensor
rnnt_loss_simple_bwd_f(const torch::Tensor &f, const torch::Tensor &g,
                       const torch::Tensor &den, const torch::Tensor &alphas,
                       const torch::Tensor &betas, const torch::Tensor &lf,
                       const torch::Tensor &ly) {
  // removed checking since this function won't be called from outside the
  // ... package
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  // zero init takes time.
  auto grads = torch::empty_like(f);

  if (den.dim() == 3) {
    run_rnnt_simple_fill_grad_f(
        (float *)grads.data_ptr<float>(),
        (const float *)alphas.data_ptr<float>(),
        (const float *)betas.data_ptr<float>(),
        (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
        (const float *)den.data_ptr<float>(), (const int *)lf.data_ptr<int>(),
        (const int *)ly.data_ptr<int>(), N, T, U);
  } else {
    run_rnnt_simple_fill_grad_f((float *)grads.data_ptr<float>(),
                                (const float *)alphas.data_ptr<float>(),
                                (const float *)betas.data_ptr<float>(),
                                (const float *)f.data_ptr<float>(),
                                (const float *)g.data_ptr<float>(),
                                (const int *)lf.data_ptr<int>(),
                                (const int *)ly.data_ptr<int>(), N, T, U);
  }

  return grads;
}

torch::Tensor
rnnt_loss_simple_bwd_g(const torch::Tensor &f, const torch::Tensor &g,
                       const torch::Tensor &den, const torch::Tensor &alphas,
                       const torch::Tensor &betas, const torch::Tensor &lf,
                       const torch::Tensor &ly) {
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

  if (den.dim() == 3) {
    run_rnnt_simple_fill_grad_g(
        (float *)grads.data_ptr<float>(),
        (unsigned int *)counts.data_ptr<float>(),
        (const float *)alphas.data_ptr<float>(),
        (const float *)betas.data_ptr<float>(),
        (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
        (const float *)den.data_ptr<float>(), (const int *)lf.data_ptr<int>(),
        (const int *)ly.data_ptr<int>(), N, T, U);
  } else {
    run_rnnt_simple_fill_grad_g((float *)grads.data_ptr<float>(),
                                (unsigned int *)counts.data_ptr<float>(),
                                (const float *)alphas.data_ptr<float>(),
                                (const float *)betas.data_ptr<float>(),
                                (const float *)f.data_ptr<float>(),
                                (const float *)g.data_ptr<float>(),
                                (const int *)lf.data_ptr<int>(),
                                (const int *)ly.data_ptr<int>(), N, T, U);
  }

  return grads;
}

torch::Tensor
rnnt_loss_simple_bwd_den(const torch::Tensor &f, const torch::Tensor &g,
                         const torch::Tensor &den, const torch::Tensor &alphas,
                         const torch::Tensor &betas, const torch::Tensor &lf,
                         const torch::Tensor &ly) {
  // removed checking since this function won't be called from outside the
  // ... package
  const at::cuda::OptionalCUDAGuard device_guard(device_of(f));

  const auto N = f.size(0);
  const auto T = f.size(1);
  // U is indeed U+1
  const auto U = g.size(1);

  // zero init takes time.
  auto grads = torch::empty_like(den);

  run_rnnt_simple_fill_grad_den(
      (float *)grads.data_ptr<float>(), (const float *)alphas.data_ptr<float>(),
      (const float *)betas.data_ptr<float>(),
      (const float *)f.data_ptr<float>(), (const float *)g.data_ptr<float>(),
      (const float *)den.data_ptr<float>(), (const int *)lf.data_ptr<int>(),
      (const int *)ly.data_ptr<int>(), N, T, U);

  return grads;
}

std::tuple<torch::Tensor, torch::Tensor>
rnnt_loss(const torch::Tensor &xs, const torch::Tensor &ys,
          const torch::Tensor &xn, const torch::Tensor &yn, const int blank,
          const float fastemit_lambda) {
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

  torch::Tensor grads = at::zeros_like(xs);

  torch::TensorOptions buffer_opts(xs.device());
  torch::TensorOptions counts_opts(xs.device());
  torch::TensorOptions costs_opts(xs.device());

  counts_opts = counts_opts.dtype(at::ScalarType::Int);
  buffer_opts = buffer_opts.dtype(at::ScalarType::Float);
  costs_opts = costs_opts.dtype(at::ScalarType::Float);

  auto counts_shape = {N, U * 2};
  auto buffer_shape = {N, T, U};
  auto costs_shape = {N};

  torch::Tensor costs = torch::empty(costs_shape, costs_opts);
  torch::Tensor counts = at::zeros(counts_shape, counts_opts);
  torch::Tensor alphas = at::empty(buffer_shape, buffer_opts);
  torch::Tensor betas = at::empty(buffer_shape, buffer_opts);

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

// return (costs, grad, loc)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
rnnt_loss_compact_forward(const torch::Tensor &xs, const torch::Tensor &ys,
                          const torch::Tensor &xn, const torch::Tensor &yn,
                          const int blank, const float fastemit_lambda,
                          const bool requires_grad) {
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
            .cumsum(0, torch::kInt32); // count of frames by current batch
    auto labelPref = yn.cumsum(0, torch::kInt32); // copy yn

    int64_t STU = memPref[-1].item<int64_t>();
    TORCH_CHECK(xs.size(0) == STU,
                "xs shape mismatch with (\\sum{xn*(yn+1)}, )")

    // set begin of memory location of each sequence
    {
        auto cumsumMemPref = memPref.index({Slice(0, -1, None)}).clone();
        auto cumsumLablePref = labelPref.index({Slice(0, -1, None)}).clone();
        memPref.index_put_({Slice(1, None, None)}, cumsumMemPref);
        labelPref.index_put_({Slice(1, None, None)}, cumsumLablePref);
    }
    memPref[0] = 0;
    labelPref[0] = 0;

    torch::Tensor gather_xs = torch::empty({STU, 2L}, xs.options());
    torch::Tensor loc =
        torch::zeros({STU}, torch::dtype(torch::kInt64).device(xs.device()));

    // gather the labels & blank
    run_gather_for_compact(
        xs.data_ptr<float>(), ys.data_ptr<int>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        gather_xs.data_ptr<float>(), loc.data_ptr<long>(),
        (unsigned int *)memPref.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, V, blank);

    // the negtive log likelihood
    torch::Tensor costs = torch::empty({N}, xs.options());
    //  for maintain the execute status of forward/backward calculation
    torch::Tensor counts = torch::zeros({ys.numel() * 2 + 2 * N}, ys.options());

    // backward variable of RNN-T
    torch::Tensor betas = torch::empty({STU}, xs.options());

    torch::Tensor grads, alphas;
    if (requires_grad) {
        // forward variable of RNN-T
        alphas = torch::empty_like(betas);
        grads = torch::empty_like(gather_xs);
    } else {
        // refer unused alphas / grads to betas
        // otherwise following alphas.data_ptr<float>() would raise error
        alphas = betas;
        grads = betas;
    }

    run_warp_rnnt_compact(
        (unsigned int *)counts.data_ptr<int>(), alphas.data_ptr<float>(),
        betas.data_ptr<float>(), gather_xs.data_ptr<float>(),
        grads.data_ptr<float>(), costs.data_ptr<float>(),
        (unsigned int *)xn.data_ptr<int>(), (unsigned int *)yn.data_ptr<int>(),
        (unsigned int *)memPref.data_ptr<int>(),
        (unsigned int *)labelPref.data_ptr<int>(), N, Tm, Um, fastemit_lambda,
        requires_grad);

    return std::make_tuple(costs, grads, loc);
}


torch::Tensor rnnt_loss_compact_backward(const torch::Tensor &grad_cost,
                                         const torch::Tensor &grad_xs,
                                         const torch::Tensor &cum_lens,
                                         const torch::Tensor &loc, long V,
                                         int blank) {
    // Check contiguous
    CHECK_CONTIGUOUS(grad_cost);
    CHECK_CONTIGUOUS(grad_xs);
    CHECK_CONTIGUOUS(loc);
    // Check types
    CHECK_FLOAT(grad_cost);
    CHECK_FLOAT(grad_xs);
    TORCH_CHECK(loc.scalar_type() == at::ScalarType::Long,
                "loc must be a Long tensor");
    // Check device
    CHECK_CUDA(grad_cost);
    CHECK_CUDA(grad_xs);
    CHECK_CUDA(cum_lens);
    CHECK_CUDA(loc);
    // Check number of dimensions and elements
    TORCH_CHECK(grad_cost.dim() == 1,
                "grad_cost must have 1 dimensions")                // (N,)
    TORCH_CHECK(grad_xs.dim() == 2, "grad must have 2 dimensions") // (STU, 2)
    TORCH_CHECK(grad_xs.size(0) == loc.size(0),
                "grad and loc must be equal in dim=0")
    const at::cuda::OptionalCUDAGuard device_guard(device_of(grad_cost));

    const auto N = grad_cost.size(0);
    const auto STU = grad_xs.size(0);

    torch::Tensor scatter_grad = torch::zeros({STU, V}, grad_cost.options());

    run_scatter_grad_compact(
        grad_cost.data_ptr<float>(), grad_xs.data_ptr<float>(),
        loc.data_ptr<long>(), (unsigned int *)cum_lens.data_ptr<int>(),
        scatter_grad.data_ptr<float>(), STU, N, V, blank);

    return scatter_grad;
}

torch::Tensor log_matmul_cuda(const torch::Tensor &self,
                              const torch::Tensor &other) {

  CHECK_FLOAT(self);
  CHECK_FLOAT(other);

  TORCH_CHECK(self.is_cuda(), "need cuda tensors");
  TORCH_CHECK(self.device() == other.device(), "need tensors on same GPU");
  TORCH_CHECK(self.dim() == 3 && other.dim() == 3, "invalid sizes");
  TORCH_CHECK(self.size(2) == other.size(1), "invalid sizes");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(self));

  const int B = self.size(0);
  const int M = self.size(1);
  const int R = self.size(2);
  const int N = other.size(2);
  const auto out = torch::empty({B, M, N}, self.options());

  log_matmul_cuda_impl(self, other, out);
  return out;
}

std::tuple<torch::Tensor, torch::Tensor>
log_matmul_backward(const torch::Tensor &grad_out, const torch::Tensor &lhs,
                    const torch::Tensor &rhs, const torch::Tensor &result,
                    const std::array<bool, 2> grad_input_mask) {
  // The derivative of res = log(exp(A) @ exp(B)) w.r.t. A is defined by
  // (dres/dA @ grad_out) = ((1/exp(res) * grad_out) @ exp(B.t())) * exp(A)
  // and the analogous thing for B

  // Now just as we don't want to compute log matmul by the naive formula and
  // use logsumexp instead, we also want to be stable here. But this is
  // difficult because the gradient is not in log space and can be negative. But
  // as the operator is linear, we can split the gradient in positive and
  // negative parts, take logs separately and then combine them at the end.
  // (incidentally, this is similar to logcumsumexp_backward).

  // Batching means
  // - the transpose is in the final dimensions
  // - we have to add reductions (sum to size) for broadcasting

  torch::Tensor grad_lhs, grad_rhs;
  if (!grad_out.defined()) {
    return std::tuple<torch::Tensor, torch::Tensor>{grad_lhs, grad_rhs};
  }
  auto almost_neg_inf = AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, grad_out.scalar_type(), "log_matmul_backward",
      [&]() {
        return at::full({}, std::numeric_limits<scalar_t>::lowest(),
                        grad_out.options());
      });

  auto compute_grad_part =
      [result](const torch::Tensor &me, const torch::Tensor &other,
               const torch::Tensor &log_grad_out_result_part, bool me_is_lhs) {
        auto gr_part = (me_is_lhs ? log_matmul_cuda(log_grad_out_result_part,
                                                    other.transpose(-2, -1))
                                  : log_matmul_cuda(other.transpose(-2, -1),
                                                    log_grad_out_result_part));
        // undo broadcasting
        for (int d = result.dim() - 3; d >= 0; d--) {
          int d2 = d - (result.dim() - me.dim());
          if (d2 < 0 || (me.size(d2) == 1 && result.size(d) > 1)) {
            gr_part = gr_part.logsumexp(std::max(d2, 0), /*keepdim=*/(d2 >= 0));
          }
        }
        return gr_part + me;
      };

  auto combine_parts = [](const torch::Tensor &pos_part,
                          const torch::Tensor &neg_part) {
    auto max_part = at::maximum(pos_part, neg_part);
    max_part = at::where(max_part.abs() == INFINITY,
                         at::zeros({}, max_part.options()), max_part);
    return ((pos_part - max_part).exp() - (neg_part - max_part).exp()) *
           max_part.exp();
  };
  auto log_grad_out_pos_res =
      at::where(grad_out > 0, grad_out.log() - result, almost_neg_inf);
  auto log_grad_out_neg_res =
      at::where(grad_out < 0, grad_out.neg().log() - result, almost_neg_inf);
  if (grad_input_mask[0]) {
    grad_lhs = combine_parts(
        compute_grad_part(lhs, rhs, log_grad_out_pos_res, /*me_is_lhs=*/true),
        compute_grad_part(lhs, rhs, log_grad_out_neg_res, /*me_is_lhs=*/true));
  }
  if (grad_input_mask[1]) {
    grad_rhs = combine_parts(
        compute_grad_part(rhs, lhs, log_grad_out_pos_res, /*me_is_lhs=*/false),
        compute_grad_part(rhs, lhs, log_grad_out_neg_res, /*me_is_lhs=*/false));
  }
  return std::tuple<torch::Tensor, torch::Tensor>{grad_lhs, grad_rhs};
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
        pybind11::arg("requires_grad") = true);

  m.def("rnnt_loss_compact_backward", &rnnt_loss_compact_backward,
        "Compact RNN-T loss backward", pybind11::arg("grad_cost"),
        pybind11::arg("grad"), pybind11::arg("cumSum"), pybind11::arg("loc"),
        pybind11::arg("V"), pybind11::arg("blank"));

  m.def("rnnt_loss_simple_fwd", &rnnt_loss_simple_fwd,
        "Simple RNN-T loss foward computing", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("den"), pybind11::arg("lf"),
        pybind11::arg("ly"));

  m.def("rnnt_loss_simple_bwd_f", &rnnt_loss_simple_bwd_f,
        "Simple RNN-T loss backward computing for f", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("den"), pybind11::arg("alphas"),
        pybind11::arg("betas"), pybind11::arg("lf"), pybind11::arg("ly"));

  m.def("rnnt_loss_simple_bwd_g", &rnnt_loss_simple_bwd_g,
        "Simple RNN-T loss backward computing for g", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("den"), pybind11::arg("alphas"),
        pybind11::arg("betas"), pybind11::arg("lf"), pybind11::arg("ly"));

  m.def("rnnt_loss_simple_bwd_den", &rnnt_loss_simple_bwd_den,
        "Simple RNN-T loss backward computing for g", pybind11::arg("f"),
        pybind11::arg("g"), pybind11::arg("den"), pybind11::arg("alphas"),
        pybind11::arg("betas"), pybind11::arg("lf"), pybind11::arg("ly"));

  m.def("log_matmul", &log_matmul_cuda, "Log matrix multiplication",
        pybind11::arg("lhs"), pybind11::arg("rhs"));

  m.def("log_matmul_backward", &log_matmul_backward,
        "Log matrix multiplication backward", pybind11::arg("grad_out"),
        pybind11::arg("lhs"), pybind11::arg("rhs"), pybind11::arg("res"),
        pybind11::arg("grad_masks"));
}
