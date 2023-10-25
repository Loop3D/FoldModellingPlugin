# setup cython code
import sys
import codecs

try:
    from setuptools import setup, find_packages
except:
    raise RuntimeError("Cannot import setuptools \n" "python -m pip install setuptools")
    sys.exit(1)

import numpy
import os

package_root = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(package_root, "FoldOptLib/version.py")) as fp:
    exec(fp.read(), version)
version = version["__version__"]

setup(
    name="FoldModellingPlugin",
    install_requires=[
        "loopstructural>=1.4.10",
        "scipy>=1.2.2",  # 1.2.2 is required to use vonmises_fisher() in scipy.stats
        "ipywidgets",
    ],
    description="Open source Fold Geometry Optimisers for LoopStructural and Map2Loop",
    long_description=codecs.open("README.md", "r", "utf-8").read(),
    author="Rabii Chaarani",
    author_email="rabii.chaarani@monash.edu",
    license=("MIT"),
    url="https://github.com/Loop3D/FoldOptLib",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Nonlinear Optimisation",
    ],
    version=version,
    packages=find_packages(),
    # ext_modules=
    # ),
    # include_dirs=[numpy.get_include()],
    # include_package_data=True,
    # package_data={
    #     "FoldOptLib": [
    #     ]
    # },
    keywords=[
        "earth sciences",
        "geology",
        "3-D modelling",
        "structural geology",
        "uncertainty",
        "fold geometry",
    ],
)
