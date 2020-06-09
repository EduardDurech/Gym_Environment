from setuptools setup

setup(
    name="env_pkg",
    version="0.0"
)

##Note: this does *NOT* check dependencies
##For dependencies, include e.g.:
"""
from setuptools import find_packages

setup(
    ...
    install_requires=["gym>=0.2.3", "pandas", "cfg_load"],
    packages=find_packages(),
    ..
)
"""
