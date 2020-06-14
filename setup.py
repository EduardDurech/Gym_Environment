from setuptools import setup

setup(
    name="env_pkg",
    version="0.0",
    install_requires=[
          'tensorflow==2.2.0',
          'keras-rl==0.4.2',
          'flatland-rl==2.2.1',
          'numpy==1.18.5',
          'gym==0.14.0'
      ],

)

