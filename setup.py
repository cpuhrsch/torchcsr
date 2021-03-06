import setuptools
import torch
import distutils.command.clean
import shutil
import os
import glob
import subprocess
import sys
import io
from build_tools import setup_helpers
from setuptools import setup, find_packages

from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
    BuildExtension,
)


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


version = "0.0.1"
sha = "Unknown"
package_name = "torchcsr"

cwd = os.path.dirname(os.path.abspath(__file__))

try:
    sha = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    pass

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
elif sha != "Unknown":
    version = version + "+" + sha[:7]

print("Building wheel {}-{}".format(package_name, version))


def write_version_file():
    version_path = os.path.join(cwd, "torchcsr", "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))
        f.write("from torchcsr import _C\n")
        f.write("if hasattr(_C, 'CUDA_VERSION'):\n")
        f.write("    cuda = _C.CUDA_VERSION\n")


write_version_file()

readme = open("README.md").read()

pytorch_dep = "torch"

requirements = [
    pytorch_dep,
]

if os.getenv("PYTORCH_VERSION"):
    pytorch_dep += "==" + os.getenv("PYTORCH_VERSION")


def get_extensions():

    extension = CppExtension

    define_macros = []

    extra_link_args = []
    extra_compile_args = {"cxx": ["-O3", "-g", "-std=c++14"]}
    if int(os.environ.get("DEBUG", 0)):
        extra_compile_args = {
            "cxx": ["-O0", "-fno-inline", "-g", "-std=c++14"]}
        extra_link_args = ["-O0", "-g"]
    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv(
        "FORCE_CUDA", "0"
    ) == "1":
        extension = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags == "":
            nvcc_flags = []
        else:
            nvcc_flags = nvcc_flags.split(" ")
        extra_compile_args["nvcc"] = nvcc_flags

    if sys.platform == "win32":
        define_macros += [("torchcsr_EXPORTS", None)]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "torchcsr", "csrc")
    utils_dir = os.path.join(extensions_dir, "utils")

    extension_sources = set(
        os.path.join(extensions_dir, p)
        for p in glob.glob(os.path.join(extensions_dir, "*.cpp"))
    )
    utils_sources = set(
        os.path.join(utils_dir, p) for p in glob.glob(os.path.join(utils_dir, "*.cpp"))
    )

    sources = list(set(extension_sources) | set(utils_sources))

    include_dirs = [extensions_dir, utils_dir]

    ext_modules = [
        extension(
            "torchcsr._C",
            sources,
            include_dirs=[this_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    return ext_modules


class clean(distutils.command.clean.clean):
    def run(self):
        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)


setup(
    name=package_name,
    version=version,
    author="Christian Puhrsch",
    author_email="cpuhrsch@fb.com",
    description="torchcsr for PyTorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/cpuhrsch/torchcsr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(exclude=["build*", "test*", "torchcsrc.csrc*", "third_party*", "build_tools*"]),
    ext_modules=setup_helpers.get_ext_modules(),
    cmdclass={
        "clean": clean,
        "build_ext": setup_helpers.CMakeBuild,
    },
    install_requires=requirements,
    zip_safe=False,
)
