import sys
import os
import subprocess
from glob import glob
import setuptools
from setuptools.command.build_py import build_py


class lisa_build(build_py):
    """
    Customized setuptools build command
    From https://stackoverflow.com/a/45349660
    """
    def run(self):
        if subprocess.call(["make", "-ik", "all"]) != 0:
            sys.exit(-1)
        build_py.run(self)

# Read in version info
with open('./lisa/_version.py') as foo:
    VER = str(foo.read().split('=')[-1].strip().strip("'"))

# Read in the README
with open('README', 'r') as foo:
    long_description = foo.read()

docdir = os.path.join(os.path.dirname(__file__), 'doc')
moddir = os.path.join(os.path.dirname(__file__), 'lisa', 'modules')

setuptools.setup(
    name='lisa',
    version=VER,
    scripts=[],
    author='Michael Himes',
    author_email='mhimes@knights.ucf.edu',
    description='Large-selection Interface for Sampling Algorithms',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/exosports/lisa',
    packages=setuptools.find_packages(), 
    include_package_data=True, 
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        ],
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'matplotlib>=3.0', 
                      'pymultinest==2.9', 'ultranest>=2.2.1', 'dynesty>=1.0.1', 
                      'dnest4>=0.2.4', 'mpi4py>=3.0.3'], 
    cmdclass={'build_py': lisa_build}
    )
