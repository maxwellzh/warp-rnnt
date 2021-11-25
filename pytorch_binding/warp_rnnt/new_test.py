import torch
import time
import warp_rnnt._C as core
from warp_rnnt import rnnt_loss, fused_rnnt_loss, fused_rnnt_loss_


xs = torch.tensor([], dtype=torch.float32)
ys = torch.tensor([], dtype=torch.int)
xn = torch.tensor([], dtype=torch.int)
yn = torch.tensor([], dtype=torch.int)


def compactTensor(xs: torch.Tensor, ys: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor):

    assert xs.dim() == 4
    assert ys.dim() == 2

    N, T, Up, V = xs.size()
    assert ys.size() == (N, Up-1)
    assert xn.size(0) == N
    assert yn.size(0) == N

    _ys = torch.cat([ys[i, :yn[i]] for i in range(N)])
    _xs = [xs[i, :xn[i], :yn[i]+1, :].contiguous() for i in range(N)]
    _xs = torch.cat([x.view(-1, V) for x in _xs], dim=0)

    return _xs, _ys


def reverseCompact(xs: torch.Tensor, ys: torch.Tensor, xn: torch.Tensor, yn: torch.Tensor):

    N, T, U, V = xn.size(0), xn.max(), yn.max(), xs.size(-1)
    _xs = xs.new_zeros((N, T, U+1, V))
    _ys = ys.new_zeros((N, U))

    offset = 0
    offset_y = 0
    for n in range(N):
        Ti, Uip = xn[n], yn[n]+1
        _xs[n, :Ti, :Uip, :] = xs[offset:offset+Ti*Uip, :].view(Ti, Uip, V)

        _ys[n, :Uip-1] = ys[offset_y:offset_y+Uip-1].view(-1)
        offset += Ti*Uip
        offset_y += Uip-1

    return _xs, _ys


def test_calls():
    n = 20
    t = 32
    u = 16
    v = 3
    cnt = 0
    for i in range(1):
        torch.manual_seed(i)
        xn = torch.tensor([t] * n, dtype=torch.int, device=0)
        yn = torch.randint(1, u, (n,), dtype=torch.int, device=0)
        ys = torch.randint(1, v, (yn.sum(), ), dtype=torch.int, device=0)
        xs = torch.randn(((xn*(yn+1)).sum(), v), dtype=torch.float32,
                         device=0).log_softmax(dim=-1)

        cumSum = torch.cumsum(xn * (yn+1), dim=0)
        _costs, _grads, _, _ = core.rnnt_loss_compact_forward(xs, ys, xn, yn)
        real_grads = core.rnnt_loss_compact_backward(torch.ones_like(_costs),
                                                     _grads, cumSum.to(torch.int32), _grads, xs.size(-1), 0)

        costs, grads, loc, blank = core.rnnt_loss_compact_forward(
            xs, ys, xn, yn, -1)
        real_grads_gather = core.rnnt_loss_compact_backward(torch.ones_like(costs),
                                                            grads, cumSum.to(torch.int32), loc, xs.size(-1), blank)

        if not torch.all(real_grads == real_grads_gather):
            print(xn)
            print(yn)
            print(xs.size())
            print(ys)
            print(_costs)
            print(costs)
            print(_grads)
            print(real_grads)
            print(real_grads_gather)
            break

        cnt += torch.all(real_grads == real_grads_gather)

        xs.requires_grad = True
        torch.autograd.gradcheck(
            rnnt_loss, (xs, ys, xn, yn, False, 'mean', 0, False, 0.0, True))

    print("Gather mode produces {} same results as non-gather one.".format(cnt))


def test_compute():

    NTest = 3

    for seed in range(NTest):
        torch.manual_seed(seed)
        N = torch.randint(1, 20, (1,)).item()
        T = torch.randint(5, 512, (1,)).item()
        U = torch.randint(1, 512, (1,)).item()
        V = torch.randint(3, 128, (1,)).item()

        xs = torch.randn((N, T, U, V), dtype=torch.float32,
                         device=0).log_softmax(dim=-1)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
        xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
        xn = xn + T - xn.max()
        yn = yn + U-1 - yn.max()

        ys = ys.to(dtype=torch.int)
        xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
            dtype=torch.int, device=0)
        print("xs size: ", xs.size())
        print("ys size: ", ys.size())
        print("lx size: ", xn.size())
        print("ly size: ", yn.size())
        xs.requires_grad = True

        m_cost = rnnt_loss(xs, ys, xn, yn, gather=False, compact=False)
        m_cost.sum().backward()
        m_grad = xs.grad.data.detach()
        xs.grad = None

        _xs, _ys = compactTensor(xs, ys, xn, yn)

        t_cost = rnnt_loss(_xs, _ys, xn, yn, gather=True, compact=True)
        t_cost.sum().backward()
        t_grad = xs.grad.data.detach()

        print("backward diff 1-order norm: {:.4e}".format(
            torch.sum(torch.abs(m_grad - t_grad)).item()))

        print("correctness: forward | backward : {} | {}\n".format(
            torch.all(m_cost == t_cost).item(), torch.all(m_grad == t_grad).item()))


def test_compact_no_grad(N: int = 3):

    minicase = True
    if minicase:
        NTest = 1
    else:
        NTest = N

    for seed in range(NTest):
        torch.manual_seed(seed)
        if minicase:
            N = torch.randint(2, 4, (1,)).item()
            T = torch.randint(4, 8, (1,)).item()
            U = torch.randint(4, 8, (1,)).item()
            V = torch.randint(3, 5, (1,)).item()
            N, T, U, V = 2, 3, 3, 3
        else:
            N = torch.randint(1, 20, (1,)).item()
            T = torch.randint(5, 512, (1,)).item()
            U = torch.randint(1, 512, (1,)).item()
            V = torch.randint(3, 128, (1,)).item()

        xs = torch.randn((N, T, U, V), dtype=torch.float32,
                         device=0).log_softmax(dim=-1)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
        xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
        xn = xn + T - xn.max()
        yn = yn + U-1 - yn.max()

        ys = ys.to(dtype=torch.int)
        xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
            dtype=torch.int, device=0)
        print("xs size: ", xs.size())
        print("ys size: ", ys.size())
        print("lx size: ", xn.size())
        print("ly size: ", yn.size())

        xs, ys = compactTensor(xs, ys, xn, yn)

        for g in [True, False]:
            xs.requires_grad = True
            cost = rnnt_loss(xs, ys, xn, yn, gather=g, compact=True)
            print(
                "(gather={}, compact={}), cost w/ grad record\n{}".format(g, True, cost))

            xs.requires_grad = False
            cost = rnnt_loss(xs, ys, xn, yn, gather=g, compact=True)
            print(
                "(gather={}, compact={}), cost w/o grad record\n{}".format(g, True, cost))


def test_fusion_compute(minicase=False):

    NTest = 1 if minicase else 5

    for seed in range(NTest):
        torch.manual_seed(seed)
        if minicase:
            N = torch.randint(2, 4, (1,)).item()
            T = torch.randint(4, 8, (1,)).item()
            U = torch.randint(4, 8, (1,)).item()
            V = torch.randint(3, 5, (1,)).item()
            N, T, U, V = 2, 3, 3, 3
        else:
            N = torch.randint(2, 8, (1,)).item()
            T = torch.randint(16, 512, (1,)).item()
            U = torch.randint(16, 512, (1,)).item()
            V = torch.randint(10, 128, (1,)).item()

        xs = torch.randn((N, T, U, V), dtype=torch.float32, device=0)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
        xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
        xn = xn + T - xn.max()
        yn = yn + U-1 - yn.max()

        ys = ys.to(dtype=torch.int)
        xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
            dtype=torch.int, device=0)
        print("xs size: ", xs.size())
        print("ys size: ", ys.size())
        print("lx size: ", xn.size())
        print("ly size: ", yn.size())
        xs, ys = compactTensor(xs, ys, xn, yn)

        xs.requires_grad = True
        weighted = torch.randn((N, ), device=xs.device, dtype=xs.dtype)
        torch.cuda.reset_peak_memory_stats()

        time_beg = time.time()
        t_cost = rnnt_loss(xs.log_softmax(dim=-1), ys, xn,
                           yn, gather=True, compact=True)
        (t_cost*weighted).sum().backward()
        t_grad = xs.grad.data.detach()
        t_dur = time.time() - time_beg
        xs.grad = None
        t_mem = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()

        time_beg = time.time()
        l_cost = fused_rnnt_loss_(xs, ys, xn, yn)
        (l_cost*weighted).sum().backward()
        l_grad = xs.grad.data.detach()
        l_dur = time.time() - time_beg
        xs.grad = None
        l_mem = torch.cuda.max_memory_allocated()
        del xs

        print("{} run: fused mode: time={:.3f}ms | memory={:.0f}MB".format(
            seed, l_dur*1000, l_mem/1e6))
        print("{} run: native mode: time={:.3f}ms | memory={:.0f}MB".format(
            seed, t_dur*1000, t_mem/1e6))

        match_backward = torch.abs(l_grad-t_grad)
        print("Forward | backward 1st-order norm: {:.3e} | {:.3e}, max diff={:.3e}".format(torch.sum(
            torch.abs(l_cost-t_cost)).item(), torch.sum(match_backward).item(), match_backward.max().item()))

        if match_backward.max().item() > 1.0:
            print(weighted)
            print(l_grad[match_backward > 1.0])
            print(t_grad[match_backward > 1.0])
            break
        if minicase:
            print("Fused costs; ", l_cost.cpu().tolist())
            print("Compact costs; ", t_cost.cpu().tolist())
            print(l_grad-t_grad)


def test_fusion_no_grad(minicase=False):

    NTest = 1 if minicase else 5

    for seed in range(NTest):
        torch.manual_seed(seed)
        if minicase:
            N = torch.randint(2, 4, (1,)).item()
            T = torch.randint(4, 8, (1,)).item()
            U = torch.randint(4, 8, (1,)).item()
            V = torch.randint(3, 5, (1,)).item()
        else:
            N = torch.randint(2, 8, (1,)).item()
            T = torch.randint(16, 512, (1,)).item()
            U = torch.randint(16, 512, (1,)).item()
            V = torch.randint(10, 128, (1,)).item()

        xs = torch.randn((N, T, U, V), dtype=torch.float32, device=0)
        ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
        xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
        yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
        xn = xn + T - xn.max()
        yn = yn + U-1 - yn.max()

        ys = ys.to(dtype=torch.int)
        xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
            dtype=torch.int, device=0)
        print("xs size: ", xs.size())
        print("ys size: ", ys.size())
        print("lx size: ", xn.size())
        print("ly size: ", yn.size())
        xs, ys = compactTensor(xs, ys, xn, yn)

        xs.requires_grad = True
        l_cost_grad = fused_rnnt_loss(xs, ys, xn, yn)
        xs.grad = None
        xs.requires_grad = False
        l_cost = fused_rnnt_loss(xs, ys, xn, yn)
        print("Cost w/ gradient enable:\n{}".format(l_cost_grad))
        print("Cost w/o gradient:\n{}".format(l_cost))
        xs.requires_grad = True
        with torch.no_grad():
            l_cost = fused_rnnt_loss(xs, ys, xn, yn)
            print("Cost w/o gradient via torch.no_grad():\n{}".format(l_cost))


def test_bench_fusion():
    torch.manual_seed(0)

    N = torch.randint(2, 8, (1,)).item()
    T = torch.randint(16, 512, (1,)).item()
    U = torch.randint(16, 512, (1,)).item()
    V = torch.randint(10, 128, (1,)).item()
    xs = torch.randn((N, T, U, V), dtype=torch.float32, device=0)
    ys = torch.randint(1, V, (N, U-1), dtype=torch.int, device=0)
    xn = torch.randint(T // 2, T+1, (N,), dtype=torch.int, device=0)
    yn = torch.randint(U // 2, U, (N,), dtype=torch.int, device=0)
    xn = xn + T - xn.max()
    yn = yn + U-1 - yn.max()
    ys = ys.to(dtype=torch.int)
    xn, yn = xn.to(dtype=torch.int, device=0), yn.to(
        dtype=torch.int, device=0)
    xs, ys = compactTensor(xs, ys, xn, yn)

    t_beg = time.time()
    costs_grad = []
    xs.requires_grad = True
    for i in range(100):
        _cost = fused_rnnt_loss(xs, ys, xn, yn)
        costs_grad.append(_cost.sum().item())

    t_dur_grad = time.time() - t_beg

    t_beg = time.time()
    costs_nograd = []
    xs.requires_grad = False
    for i in range(100):
        torch.manual_seed(i)
        _cost = fused_rnnt_loss(xs, ys, xn, yn)
        costs_nograd.append(_cost.sum().item())

    t_dur_nograd = time.time() - t_beg

    print(
        "Time (ms): w/ | w/o grad = {:.2f} | {:.2f} ".format(t_dur_grad*10, t_dur_nograd*10))

    for cg, cng in zip(costs_grad[:10], costs_nograd[:10]):
        print(f"{cg:<10.2f}  {cng:<10.2f}")


if __name__ == "__main__":
    try:
        test_compute()
        test_fusion_compute(minicase=True)
        test_fusion_no_grad(minicase=True)
        test_compact_no_grad(1)

        test_fusion_compute(minicase=False)
        test_fusion_no_grad(minicase=False)
        test_compact_no_grad(10)

        test_bench_fusion()
    except Exception as e:
        print(e)
