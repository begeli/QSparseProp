import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, compiler: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
        self.compiler = compiler

    def get_args(self):
        if self.compiler == 'clang':
            return self._clang_args()
        elif self.compiler == 'gcc':
            return self._gcc_args()
        elif self.compiler == 'icc':
            return self._icc_args()

        return [], []

    def _clang_args(self):
        cmake_args = [
            "-DCMAKE_MAKE_PROGRAM=/Applications/CLion.app/Contents/bin/ninja/mac/ninja",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
            "-DCMAKE_CXX_FLAGS=-std=c++17 -march=icelake-client -mprefer-vector-width=512 -O3 -ffast-math -fno-finite-math-only -Xclang -fopenmp",
            "-G Ninja"
        ]
        build_args = ["--target", "all", "-j", "6"]

        return cmake_args, build_args

    def _gcc_args(self):
        cmake_args = [
            "-DCMAKE_MAKE_PROGRAM=/Applications/CLion.app/Contents/bin/ninja/mac/ninja",
            "-DCMAKE_C_COMPILER=/usr/local/bin/gcc-13",
            "-DCMAKE_CXX_COMPILER=/usr/local/bin/g++-13",
            "-DCMAKE_CXX_FLAGS=-std=c++17 -O3 -march=icelake-client -mprefer-vector-width=512 -ffast-math -fno-finite-math-only -fopenmp",
            "-G Ninja",
        ]
        build_args = ["--target", "all", "-j", "6"]

        return cmake_args, build_args

    def _icc_args(self):
        cmake_args = [
            "-DCMAKE_MAKE_PROGRAM=/Applications/CLion.app/Contents/bin/ninja/mac/ninja",
            "-DCMAKE_C_COMPILER=icc",
            "-DCMAKE_CXX_COMPILER=icpc",
            "-DCMAKE_CXX_FLAGS=-std=c++17 -O3 -march=icelake-client -ffast-math -fno-finite-math-only -qopenmp",
            "-G Ninja"
        ]
        build_args = ["--target", "all", "-j", "6"]

        return cmake_args, build_args


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",  # not used on MSVC, but no harm
            f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"
        ]
        cmake_args_, build_args = ext.get_args()
        cmake_args += cmake_args_

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["/usr/local/bin/cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["/usr/local/bin/cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in a file.
# Change the compiler value to one of the three supported compilers. (clang, gcc, icc)
compiler = "clang"
setup(
    name="_".join(("qsparseprop_backend", compiler)),
    version="0.0.1",
    author="Berke Egeli",
    author_email="begeli98@gmail.com",
    description="A project with AVX512 implementations for quantized sparse forward/backward propagation algorithms for convolutional/linear layers.",
    long_description="",
    ext_modules=[CMakeExtension("qsparseprop_backend", compiler)],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    extras_require={"test": ["pytest>=6.0"]},
    python_requires=">=3.7",
)