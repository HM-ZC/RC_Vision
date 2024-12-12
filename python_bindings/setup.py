from setuptools import setup, Extension
import pybind11
import sys
import os

from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "rc_vision_py",
        ["src/pybind_module.cpp"],
        include_dirs=[
            os.path.join(os.getcwd(), "../core_lib/include")
        ],
        library_dirs=[os.path.join(os.getcwd(), "../install_core/lib")],
        libraries=["rc_vision_core"],
        language="c++"
    ),
]

setup(
    name="rc_vision_py",
    version="0.1.0",
    author="HM_ZC",
    author_email="hmzc0327@gmail.com",
    description="Python bindings for rc_vision_core",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
