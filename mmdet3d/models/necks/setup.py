from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='bev_pool_ext',
    ext_modules=[
        CppExtension(
            name='bev_pool_ext',
            sources=['bevpool.cpp'],
            extra_compile_args={'cxx': ['/std:c++14']}
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
