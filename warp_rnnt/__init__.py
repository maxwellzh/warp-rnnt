import torch
import torch.nn.functional as F
import warp_rnnt._C as core
from typing import *
from pkg_resources import get_distribution

__version__ = get_distribution("warp_rnnt").version


class _RNNTLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        labels,
        frames_lengths,
        labels_lengths,
        blank=0,
        fastemit_lambda=0.0,
    ):
        costs, ctx.grads = core.rnnt_loss(
            xs=log_probs,
            ys=labels,
            xn=frames_lengths,
            yn=labels_lengths,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
        )
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads_output = grads_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grads_output), None, None, None, None, None


class _RNNTLossCompact(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs,
        labels,
        frames_lengths,
        labels_lengths,
        blank=0,
        fastemit_lambda=0.0,
        enable_grad: bool = True,
    ):
        costs, grads, loc = core.rnnt_loss_compact_forward(
            xs=log_probs,
            ys=labels,
            xn=frames_lengths,
            yn=labels_lengths,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
            requires_grad=enable_grad,
        )
        if enable_grad:
            cumlen = torch.cumsum(
                frames_lengths * (labels_lengths + 1), dim=0, dtype=torch.int32
            )
            ctx.V = log_probs.size(-1)
            ctx.blank = blank
            ctx.save_for_backward(grads, loc, cumlen)
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads, loc, cumlen = ctx.saved_tensors
        grads_input = core.rnnt_loss_compact_backward(
            grads_output.contiguous(), grads, cumlen, loc, ctx.V, ctx.blank
        )

        return grads_input, None, None, None, None, None, None


def rnnt_loss(
    log_probs: torch.FloatTensor,
    labels: torch.IntTensor,
    frames_lengths: torch.IntTensor,
    labels_lengths: torch.IntTensor,
    average_frames: bool = False,
    reduction: Literal["sum", "mean", "none"] = "mean",
    blank: int = 0,
    gather: bool = True,
    fastemit_lambda: float = 0.0,
    compact: bool = False,
) -> torch.Tensor:
    """The CUDA-Warp RNN-Transducer loss.

    Args:
        log_probs (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        labels (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        frames_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        labels_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.
        average_frames (bool, optional): Specifies whether the loss of each
            sample should be divided by its number of frames.
            Default: False.
        reduction (string, optional): Specifies the type of reduction.
            Default: None.
        blank (int, optional): label used to represent the blank symbol.
            Default: 0.
        gather (bool, optional): Reduce memory consumption.
            Default: False.
        fastemit_lambda (float, optional): FastEmit regularization
            (https://arxiv.org/abs/2010.11148).
            Default: 0.0.
        compact (bool, optional): Use compact layout, if True, shapes of inputs should be:
            log_probs: (STU, 2)
            labels:    (SU, )
            where STU = sum(frames_lengths * (labels_lengths+1))
                  SU  = sum(labels_lengths)
    """

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)
    assert isinstance(gather, bool)

    assert not labels.requires_grad, "labels does not require gradients"
    assert not frames_lengths.requires_grad, "frames_lengths does not require gradients"
    assert not labels_lengths.requires_grad, "labels_lengths does not require gradients"

    if compact:
        costs = _RNNTLossCompact.apply(
            log_probs.float(),
            labels,
            frames_lengths,
            labels_lengths,
            blank,
            fastemit_lambda,
            (log_probs.requires_grad and torch.is_grad_enabled()),
        )
    else:
        if gather:
            N, T, U, V = log_probs.size()
            index = torch.full(
                [N, T, U, 2], blank, device=labels.device, dtype=torch.long
            )
            index[:, :, : U - 1, 1] = labels.unsqueeze(dim=1)
            log_probs = log_probs.gather(dim=3, index=index)
            blank = -1

        costs = _RNNTLoss.apply(
            log_probs.float(),
            labels,
            frames_lengths,
            labels_lengths,
            blank,
            fastemit_lambda,
        )

    if average_frames:
        costs = costs / frames_lengths.to(log_probs)

    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    else:
        raise ValueError(
            f"Unknown reduction method: {reduction}, expected to be one of ['mean', 'sum', 'none']"
        )


# NOTE (huahuan): to simplify the code, I hard-coded the blank=0 in rnnt_loss_simple()


class _RNNTLossSimple(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        f,
        g,
        lf,
        ll,
        track_grad_f: bool = True,
        track_grad_g: bool = True,
        den: torch.Tensor = None,
    ):
        if den is None:
            den = torch.empty(1)

        costs, alphas, betas = core.rnnt_loss_simple_fwd(f, g, den, lf, ll)

        grad_f = (
            core.rnnt_loss_simple_bwd_f(f, g, den, alphas, betas, lf, ll)
            if track_grad_f
            else None
        )
        grad_g = (
            core.rnnt_loss_simple_bwd_g(f, g, den, alphas, betas, lf, ll)
            if track_grad_g
            else None
        )

        if den.dim() == 3 and (track_grad_f or track_grad_g):
            grad_den = core.rnnt_loss_simple_bwd_den(f, g, den, alphas, betas, lf, ll)
        else:
            grad_den = None

        ctx.save_for_backward(grad_f, grad_g, grad_den)
        return costs

    @staticmethod
    def backward(ctx, grad_costs):
        grad_costs = grad_costs.view(-1, 1, 1)
        grad_f, grad_g, grad_den = ctx.saved_tensors

        if grad_f is not None:
            grad_f *= grad_costs

        if grad_g is not None:
            grad_g *= grad_costs

        if grad_den is not None:
            grad_den *= grad_costs

        return grad_f, grad_g, None, None, None, None, grad_den


class _LogMMExp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        track_l: bool = True,
        track_r: bool = True,
    ) -> torch.Tensor:
        mm = core.log_matmul(lhs, rhs)

        if track_l or track_r:
            ctx.save_for_backward(lhs, rhs, mm)
            ctx.track = [track_l, track_r]
        return mm

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        lhs, rhs, res = ctx.saved_tensors

        grad_lhs, grad_rhs = core.log_matmul_backward(
            grad_outputs, lhs, rhs, res, ctx.track
        )
        return grad_lhs, grad_rhs, None, None


def rnnt_loss_simple(
    f_enc: torch.Tensor,
    g_pred: torch.Tensor,
    labels: torch.Tensor,
    lf: torch.Tensor,
    ll: torch.Tensor,
    reduction: Literal["none", "sum", "mean"] = "mean",
    factor_den: float = 1.0,
    avg_length: bool = False,
):
    """CUDA-warp simple rnn-t loss with the joiner as an log-add op.

    Arguments:
        f_enc:  (N, T, V)   output of encoder, denoting log probs or scores
        g_pred: (N, U+1, V) output of predictor, denoting log probs or scores
        labels: (N, U)      target label seqs
        lf:     (N, )       lengths of f_enc
        ll:     (N, )       lengths of labels
        factor_den (float) : a weighting factor of denominator in log space
            e.g. 1.0 is a fully local normalized loss, 0.0 is fully non-local normalized loss.
    """
    assert torch.all(ll > 0), f"get invalid lengths of labels (=0): {ll}"
    lf = lf.to(device=f_enc.device, dtype=torch.int32)
    ll = ll.to(device=f_enc.device, dtype=torch.int32)
    labels = labels.to(device=f_enc.device, dtype=torch.int64)

    N, T, V = f_enc.shape
    U = labels.shape[1]

    track_f = f_enc.requires_grad and torch.is_grad_enabled()
    track_g = g_pred.requires_grad and torch.is_grad_enabled()
    if factor_den == 0.0:
        den = None
    else:
        den = _LogMMExp.apply(
            f_enc.float(), g_pred.float().transpose(1, 2), track_f, track_g
        )
        den = factor_den * den

    """
    gather the target label and the blank symbol
    after gathering:
      y(n, t, u)   = f_enc(n, t, u+1) + g_pred(n, u, 1), 0 <= t < T, 0 <= u < U
      blk(n, t, u) = f_enc(n, t, 0) + g_pred(n, u, 0),   0 <= t < T, 0 <= u <= U
    """
    index = labels.new_zeros((N, 1 + U))
    index[:, 1:] = labels
    # (N, T, V) -> (N, T, U)
    f = f_enc.gather(dim=2, index=labels.unsqueeze(1).expand(-1, T, -1))

    index = labels.new_zeros((N, U + 1, 2))
    index[:, :-1, 1] = labels
    # (N, U+1, V) -> (N, U, 2)
    g = g_pred.gather(dim=2, index=index)

    costs = _RNNTLossSimple.apply(f.float(), g.float(), lf, ll, track_f, track_g, den)
    if avg_length:
        costs /= lf + ll

    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum(dim=0)
    elif reduction == "mean":
        return costs.mean(dim=0)
    else:
        raise ValueError(
            f"Unknown reduction method: {reduction}, expected to be one of ['mean', 'sum', 'none']"
        )
