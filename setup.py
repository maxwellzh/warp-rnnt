import os

import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    with open(readme_file, 'r') as f:
        return f.read()


if not torch.cuda.is_available():
    raise Exception("CPU version is not implemented")


requirements = [
    "pybind11",
    "numpy",
    "torch>=1.0.0"
]
long_description = get_long_description()

setup(
    name="warp_rnnt",
    version="0.8.0",
    description="PyTorch bindings for CUDA-Warp RNN-Transducer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxwellzh/warp-rnnt",
    author="Huahuan Zheng",
    author_email="maxwellzh@outlook.com",
    license="MIT",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="warp_rnnt._C",
            sources=[
                "csrc/binding.cpp",
                "csrc/gather.cu",
                "csrc/core.cu",
                "csrc/core_gather.cu",
                "csrc/core_compact.cu",
                "csrc/core_compact_gather.cu",
            ]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    setup_requires=requirements,
    install_requires=requirements
)
