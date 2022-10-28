import torch
import warp_rnnt._C as core
from typing import *
from pkg_resources import get_distribution
from torch.cuda.amp import autocast

__version__ = get_distribution('warp_rnnt').version


class _RNNTLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, log_probs, labels, frames_lengths, labels_lengths, blank=0, fastemit_lambda=0.0):
        costs, ctx.grads = core.rnnt_loss(
            xs=log_probs, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
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
    def forward(ctx, log_probs, labels, frames_lengths, labels_lengths, blank=0, fastemit_lambda=0.0, enable_grad: bool = True):
        ctx.blank = blank

        costs, grads, loc = core.rnnt_loss_compact_forward(
            xs=log_probs, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
            blank=blank,
            fastemit_lambda=fastemit_lambda,
            require_grad=enable_grad
        )
        if enable_grad:
            expand_len = (frames_lengths * (labels_lengths+1))
            ctx.V = log_probs.size(-1)
            ctx.save_for_backward(grads, loc, expand_len)
        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads, loc, expand_len = ctx.saved_tensors

        if ctx.blank < 0:
            cumSum = torch.cumsum(expand_len, dim=0, dtype=torch.int32)
            grads_input = core.rnnt_loss_compact_backward(
                grads_output.contiguous(), grads, cumSum, loc, ctx.V, ctx.blank)
        else:
            expand_grads_output = grads_output.gather(
                dim=0, index=torch.repeat_interleave(expand_len.to(dtype=torch.long)))
            grads *= expand_grads_output.view(-1, 1)
            grads_input = grads

        return grads_input, None, None, None, None, None, None


class _RNNTLossFusion(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, frames_lengths, labels_lengths, blank=0, enable_grad: bool = True):
        if blank < 0:
            raise NotImplementedError(
                "Fusion with gather=True is not implemented.")
        else:
            ctx.blank = blank

        costs, grads = core.rnnt_loss_fused_forward(
            xs=logits, ys=labels,
            xn=frames_lengths, yn=labels_lengths,
            blank=blank, require_grad=enable_grad
        )
        if enable_grad:
            expand_len = (frames_lengths * (labels_lengths+1))
            ctx.V = logits.size(-1)
            ctx.save_for_backward(grads, expand_len)

        return costs

    @staticmethod
    def backward(ctx, grads_output):
        grads, expand_len = ctx.saved_tensors

        expand_grads_output = grads_output.gather(
            dim=0, index=torch.repeat_interleave(expand_len.to(dtype=torch.long)))
        grads *= expand_grads_output.view(-1, 1)

        return grads, None, None, None, None, None


def rnnt_loss(log_probs: torch.FloatTensor,
              labels: torch.IntTensor,
              frames_lengths: torch.IntTensor,
              labels_lengths: torch.IntTensor,
              average_frames: bool = False,
              reduction: Optional[AnyStr] = None,
              blank: int = 0,
              gather: bool = True,
              fastemit_lambda: float = 0.0,
              compact: bool = False) -> torch.Tensor:
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
    """

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)
    assert isinstance(gather, bool)

    assert not labels.requires_grad, "labels does not require gradients"
    assert not frames_lengths.requires_grad, "frames_lengths does not require gradients"
    assert not labels_lengths.requires_grad, "labels_lengths does not require gradients"

    if gather:
        if compact:
            blank = -(blank+1)  # cast [0, ) to ( , -1)

        else:
            N, T, U, V = log_probs.size()
            index = torch.full(
                [N, T, U, 2], blank,
                device=labels.device, dtype=torch.long
            )
            index[:, :, :U-1, 1] = labels.unsqueeze(dim=1)
            log_probs = log_probs.gather(dim=3, index=index)
            blank = -1

    enable_grad = (log_probs.requires_grad and torch.is_grad_enabled())
    with autocast(enabled=False):
        if compact:
            costs = _RNNTLossCompact.apply(log_probs.float(), labels, frames_lengths,
                                           labels_lengths, blank, fastemit_lambda, enable_grad)
        else:
            costs = _RNNTLoss.apply(log_probs.float(), labels, frames_lengths,
                                    labels_lengths, blank, fastemit_lambda)

    if average_frames:
        costs = costs / frames_lengths.to(log_probs)

    if reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    return costs


def rnnt_loss_fused(logits: torch.FloatTensor,
                    labels: torch.IntTensor,
                    frames_lengths: torch.IntTensor,
                    labels_lengths: torch.IntTensor,
                    average_frames: bool = False,
                    reduction: Optional[AnyStr] = None,
                    blank: int = 0) -> torch.Tensor:
    logit_clone = logits.clone()
    return rnnt_loss_fused_(logit_clone, labels, frames_lengths, labels_lengths, average_frames,
                            reduction, blank)


def rnnt_loss_fused_(logits: torch.FloatTensor,
                     labels: torch.IntTensor,
                     frames_lengths: torch.IntTensor,
                     labels_lengths: torch.IntTensor,
                     average_frames: bool = False,
                     reduction: Optional[AnyStr] = None,
                     blank: int = 0) -> torch.Tensor:

    assert average_frames is None or isinstance(average_frames, bool)
    assert reduction is None or reduction in ("none", "mean", "sum")
    assert isinstance(blank, int)

    assert not labels.requires_grad, "labels does not require gradients"
    assert not frames_lengths.requires_grad, "frames_lengths does not require gradients"
    assert not labels_lengths.requires_grad, "labels_lengths does not require gradients"

    assert logits.dim(
    ) == 2, f"Logits should be of shape (Stu, V), instead of {logits.size()}"
    if labels.dim() == 2:
        assert labels.size(
            1) == 1, f"Labels should be of size (Su, 1) of (Su,), instead of {labels.size()}"
    elif labels.dim() != 1:
        raise RuntimeError(
            f"Labels should be of size (Su, 1) of (Su,), instead of {labels.size()}")

    assert frames_lengths.dim() == 1
    assert labels_lengths.dim() == 1
    assert frames_lengths.size(0) == labels_lengths.size(0)

    with autocast(enabled=False):
        costs = _RNNTLossFusion.apply(
            logits.float(), labels, frames_lengths, labels_lengths, blank, (logits.requires_grad and torch.is_grad_enabled()))

    if average_frames:
        costs = costs / frames_lengths.to(logits)
    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum()
    elif reduction == "mean":
        return costs.mean()
    else:
        raise ValueError(
            f"Unknown reduction method: {reduction}, expected to be one of ['mean', 'sum', 'none']")

# NOTE (huahuan): to simplify the code, I hard-coded the blank=0 in rnnt_loss_simple()


class _RNNTLossSimple(torch.autograd.Function):

    @staticmethod
    def forward(ctx, f, g, lf, ll, track_grad_f: bool = True, track_grad_g: bool = True):
        costs, alphas, betas = core.rnnt_loss_simple_fwd(f, g, lf, ll)

        grad_f = core.rnnt_loss_simple_bwd_f(
            f, g, alphas, betas, lf, ll) if track_grad_f else None
        grad_g = core.rnnt_loss_simple_bwd_g(
            f, g, alphas, betas, lf, ll) if track_grad_g else None
        ctx.save_for_backward(grad_f, grad_g)
        return costs

    @staticmethod
    def backward(ctx, grad_costs):
        grad_costs = grad_costs.view(-1, 1, 1)
        (grad_f, grad_g) = ctx.saved_tensors
        if grad_f is not None:
            grad_f *= grad_costs

        if grad_g is not None:
            grad_g *= grad_costs
        return grad_f, grad_g, None, None, None, None


def rnnt_loss_simple(
        f_enc: torch.Tensor,
        g_pred: torch.Tensor,
        labels: torch.Tensor,
        lf: torch.Tensor,
        ll: torch.Tensor,
        reduction: Literal['none', 'sum', 'mean'] = 'mean'):
    """CUDA-warp simple rnn-t loss with the joiner as an log-add op.

    Arguments:
        f_enc:  (N, T, V)   output of encoder, denoting log probs or scores
        g_pred: (N, U+1, V) output of predictor, denoting log probs or scores
        labels: (N, U)      target label seqs
        lf:     (N, )       lengths of f_enc
        ll:     (N, )       lengths of labels
    """
    assert torch.all(ll > 0), f"get invalid lengths of labels (=0): {ll}"
    lf = lf.to(device=f_enc.device, dtype=torch.int32)
    ll = ll.to(device=f_enc.device, dtype=torch.int32)
    labels = labels.to(device=f_enc.device, dtype=torch.int64)

    N, T, V = f_enc.shape
    U = labels.shape[1]
    blanks = labels.new_zeros(1, 1)
    # gather the target label and the blank symbol
    # after gathering:
    #   y(n, t, u)   = f_enc(n, t, u+1) + g_pred(n, u, 1), 0 <= t < T, 0 <= u < U
    #   blk(n, t, u) = f_enc(n, t, 0) + g_pred(n, u, 0),   0 <= t < T, 0 <= u <= U
    ## (N, T, V) -> (N, T, 1+U)
    f_enc = torch.gather(f_enc, dim=2, index=torch.cat(
        (blanks.expand(N, -1), labels), dim=1).unsqueeze(1).expand(-1, T, -1))
    ## (N, U+1, U) -> (N, U+1, 2), 0 for blank, 1 for label at the position
    # fmt:off
    g_pred = torch.gather(g_pred, dim=2, index=torch.cat(
        (
            # (N, U+1, 1)
            blanks.unsqueeze(0).expand(N, U+1, -1), 
            # (N, U+1, 1)
            torch.cat(
                (
                    # (N, U)
                    labels, 
                    # (N, 1), this can be any token 0 <= and < V
                    blanks.expand(N, -1)
                ),
                dim=1
            ).unsqueeze(2)
        ),
        dim=2
    ))
    # fmt:on

    with autocast(enabled=False):
        costs = _RNNTLossSimple.apply(
            f_enc.float(), g_pred.float(), lf, ll,
            f_enc.requires_grad and torch.is_grad_enabled(),
            g_pred.requires_grad and torch.is_grad_enabled()
        )
    if reduction == "none" or reduction is None:
        return costs
    elif reduction == "sum":
        return costs.sum(dim=0)
    elif reduction == "mean":
        return costs.mean(dim=0)
    else:
        raise ValueError(
            f"Unknown reduction method: {reduction}, expected to be one of ['mean', 'sum', 'none']")
