from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='triplane_sampler_cuda',
    ext_modules=[
        CUDAExtension('triplane_sampler_cuda', [
            'myGridSampler.cpp',
            'myGridSampler_kernel.cu',
        ], include_dirs=[
            '/home/mil/noguchi/D3/triplane_grid_sample',
        ], )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
