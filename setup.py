import os
import sys

from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)


class CUDAArchListError(RuntimeError):
    """To raise if CUDA is found and TORCH_CUDA_ARCH_LIST is not set."""

    def __init__(self) -> None:
        super().__init__(
            "You must explicitly set TORCH_CUDA_ARCH_LIST to build from source if CUDA is found.\n"
            "Check you supported gpu architectures beforehand.\n"
            "For example: TORCH_CUDA_ARCH_LIST='7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX'"
        )


def get_extension() -> Extension:
    """Either CUDA or CPU extension."""
    use_cuda = CUDA_HOME is not None and sys.platform != "win32"
    if use_cuda and "TORCH_CUDA_ARCH_LIST" not in os.environ:
        raise CUDAArchListError
    sources = ["src/torchsoftdtw/csrc/softdtw.cpp"] + (
        ["src/torchsoftdtw/csrc/cuda/softdtw.cu"] if use_cuda else []
    )
    TORCH_TARGET_VERSION = "0x020A000000000000"
    extra_compile_args = {
        "cxx": [
            f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
            "-DTORCH_STABLE_ONLY",
            "-Werror",
        ],
        "nvcc": [f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}"],
    }
    extension = (CUDAExtension if use_cuda else CppExtension)(
        "torchsoftdtw._C",
        sources,
        define_macros=[("WITH_CUDA", None)] if use_cuda else [],
        extra_compile_args=extra_compile_args,
        py_limited_api=True,
    )
    if use_cuda:
        # Remove cudart so it does not appear in the .so's dependencies.
        # Cudart symbols are resolved at runtime from the cudart already loaded by PyTorch,
        # making the wheel compatible across CUDA major versions.
        extension.libraries = [
            lib for lib in extension.libraries if "cudart" not in lib
        ]
    return extension


setup(
    name="torchsoftdtw",
    version="0.1.0",
    ext_modules=[get_extension()],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp312"}},
)
