from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='triplane_sampler_cuda',
    ext_modules=[
        CUDAExtension('triplane_sampler_cuda', [
            'TriplaneSampler.cpp',
            'TriplaneSampler_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
