import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules = []

csrc_dir = os.path.join("src", "torchsoftdtw", "csrc")
sources = [os.path.join(csrc_dir, "softdtw.cpp")]

# Target PyTorch's libtorch stable ABI (torch/csrc/stable) at the minimum
# version required by this package (see pyproject.toml), so the compiled
# extension keeps working against newer libtorch releases without a rebuild.
TORCH_TARGET_VERSION = "0x0210000000000000ULL"
extra_compile_args = {
    "cxx": [
        f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}",
        "-DTORCH_STABLE_ONLY",
        "-Werror",
    ],
    "nvcc": [f"-DTORCH_TARGET_VERSION={TORCH_TARGET_VERSION}"],
}

cuda_available = False
try:
    import torch

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        from torch.utils.cpp_extension import CUDAExtension

        cuda_source = os.path.join(csrc_dir, "cuda", "softdtw.cu")
        if os.path.exists(cuda_source):
            sources.append(cuda_source)
            ext_modules.append(
                CUDAExtension(
                    name="torchsoftdtw._C",
                    sources=sources,
                    define_macros=[("WITH_CUDA", None)],
                    extra_compile_args=extra_compile_args,
                    py_limited_api=True,
                )
            )
            cuda_available = True
except Exception:
    pass

if not cuda_available:
    ext_modules.append(
        CppExtension(
            name="torchsoftdtw._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
        )
    )

setup(
    name="torchsoftdtw",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
