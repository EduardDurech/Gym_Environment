from setuptools setup

setup(
    name="gym_banana",
    version="0.0.2",
    install_requires=["gym>=0.2.3", "pandas", "cfg_load"],
    packages=find_packages(),
)

/*Note, this does *NOT* check dependencies, include e.g.:
```
from setuptools import find_packages

setup(
    ...
    install_requires=["gym>=0.2.3", "pandas", "cfg_load"],
    packages=find_packages(),
    ..
)
```*/
