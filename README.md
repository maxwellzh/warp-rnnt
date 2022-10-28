
# PyTorch bindings for CUDA-Warp RNN-Transducer

This is ported from Ivan Sorokin's [warp-rnnt](https://github.com/1ytic/warp-rnnt) with some features introduced:

1. Compact memory layout RNN-T loss support (No paddings in the output tensor of the joiner.)
2. Fuse the `log_softmax` along with the rnn-t loss. (This would consume more memory than the `gather=True`, so it's not recommended to be used.)
3. Un-normalized simple "rnn-t": there's no a such joiner, to obtain `z(t, u)` (i.e. the log prob at position (t, u) in trivial rnn-t) in the rnn-t lattice, we simply compute the log-add of `enc(t)` and `pred(u)`.

    AFAIK, this is first proposed in the original paper of rnn-t by Alex Graves ([paper](https://arxiv.org/pdf/1211.3711.pdf), Eq. 12-13). And there're already some efficient implementations: Awni Hannun's [transducer](https://github.com/awni/transducer) and Kaldi's [k2-fsa](https://k2-fsa.github.io/k2/python_api/api.html#rnnt-loss-simple).
    
    However, here in my implementation, the essential difference is that we don't do the normalization (the denominator in Eq. 13 in Graves' paper). Note that without the normalization factor, the rnn-t loss is no more a negative log probability and boundless. The most promising feature of "simple rnnt loss" is that the memory consumption is limited by `O(N*(T+U)*V)` instead of `O(N*T*U*V)` in the vanilla implementation.

## Usage

Please refer to the python module entry: `warp_rnnt/__init__.py`

## Requirements

- C++11 or C++14 compiler (tested with GCC 7.5).
- Python: 3.5+ (tested with version 3.9).
- [PyTorch](http://pytorch.org/) >= 1.0.0 (tested with version 1.12.1).
- [CUDA Toolkit](https://developer.nvidia.com/cuda-zone) (tested with version 11.5).



## Install

The following setup instructions compile the package from the source code locally.

```bash
git clone https://github.com/maxwellzh/warp-rnnt.git
# install to where you clone
python -m pip install -e .
# install to your python packages
# python -m pip install .
```

## Test
There is a unittest which includes tests for arguments and outputs as well.

```bash
python -m warp_rnnt.test
```


## Reference

- Ivan Sorokin [warp-rnnt](https://github.com/1ytic/warp-rnnt)

- Awni Hannun [transducer](https://github.com/awni/transducer)

- Alex Graves [Sequence transduction with recurrent neural network](https://arxiv.org/pdf/1211.3711.pdf)

- Kaldi k2-fsa [rnnt-loss-simple](https://k2-fsa.github.io/k2/python_api/api.html#rnnt-loss-simple)

- Speech recognition with rnn-t. [repo](https://github.com/maxwellzh/Transducer-dev)