# -*- coding: utf-8; -*-

# python3 setup.py
# python3 setup.py sdist bdist_wheel

import sys
# from distutils.core import setup
import setuptools
from setuptools import setup

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python >= 3.6 required.")

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="eppes",
    version="0.5.0",
    description="EPPES ensemble estimation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["eppes"],
#    packages=setuptools.find_packages(),
    install_requires=['numpy','scipy'],
    author='Marko Laine',
    author_email='marko.laine@fmi.fi',
    license='MIT',
    url="https://github.com/mjlaine/eppes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
