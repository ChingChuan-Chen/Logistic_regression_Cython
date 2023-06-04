#!/usr/bin/python
from setuptools import Extension, find_packages, setup

if __name__ == '__main__':
    try:
        from Cython.Distutils import build_ext
        sources = ['speed_logit_reg/_speed_logit_reg.pyx']
    except:
        from setuptools.command.build_ext import build_ext
        sources = ['speed_logit_reg/_speed_logit_reg.cpp']

    class CustomBuildExt(build_ext):
        """Custom build_ext class to defer numpy imports until needed.

        Overrides the run command for building an extension and adds in numpy
        include dirs to the extension build. Doing this at extension build time
        allows us to avoid requiring that numpy be pre-installed before
        executing this setup script.
        """

        def run(self):
            import numpy
            self.include_dirs.append(numpy.get_include())
            build_ext.run(self)

    cpp_ext = Extension(
        'speed_logit_reg._speed_logit_reg',
        sources=sources,
        libraries=[],
        include_dirs=[],
        language='c++',
    )

    setup(
        name='speed_logit_reg',
        version='0.1',
        description=__doc__,
        license='BSD 3 Clause',
        url='github.com:ChingChuan-Chen/Logistic_regression_Cython',
        author='Ching-Chuan Chen',
        author_email='zw12356@gmail.com',
        install_requires=[
            'setuptools>=18.0',
            'numpy>=1.11.2',
            'scikit-learn>=0.18.1',
        ],
        setup_requires=[
            'numpy>=1.11.2',
        ],
        packages=find_packages(),
        ext_modules=[cpp_ext],
        cmdclass={'build_ext': CustomBuildExt},
    )
