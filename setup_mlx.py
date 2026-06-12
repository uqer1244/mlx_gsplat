import os
import sys
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            
            # Locate MLX package path
            import mlx.core as mx
            mlx_pkg_dir = os.path.dirname(mx.__file__)
            mlx_include = os.path.join(mlx_pkg_dir, "include")
            mlx_include_metal_cpp = os.path.join(mlx_include, "metal_cpp")
            mlx_lib_dir = os.path.join(mlx_pkg_dir, "lib")
            
            ext.include_dirs.append(mlx_include)
            ext.include_dirs.append(mlx_include_metal_cpp)
            ext.library_dirs.append(mlx_lib_dir)
            ext.libraries.append("mlx")
            
            ext.extra_compile_args = [
                "-std=c++17", 
                "-O3", 
                "-stdlib=libc++",
                "-w"
            ]
            
            # Frameworks needed for Metal support
            ext.extra_link_args = [
                "-framework", "Metal", 
                "-framework", "Foundation",
                "-Wl,-rpath," + mlx_lib_dir
            ]
            
        super().build_extensions()

setup(
    name="mlx_gsplat_ext",
    version="0.1.0",
    ext_modules=[
        Extension(
            "mlx_gsplat_ext",
            sources=["src/mlx_extensions.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExt},
)
