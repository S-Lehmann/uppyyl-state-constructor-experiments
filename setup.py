#!/usr/bin/env python

"""setup.py: Controls the setup process using setuptools."""

import re

from setuptools import setup

version = re.search(
    r'^__version__\s*=\s*"(.*)"',
    open('uppyyl_state_constructor_experiments/version.py').read(),
    re.M
).group(1)

with open("README.md", "rb") as f:
    long_description = f.read().decode("utf-8")

setup(
    name="uppyyl_state_constructor_experiments",
    packages=["uppyyl_state_constructor_experiments"],
    entry_points={
        "console_scripts": [
            'uppyyl_state_constructor_experiments = '
            'uppyyl_state_constructor_experiments.__main__:main',
            'uppyyl-state-constructor-experiments = '
            'uppyyl_state_constructor_experiments.__main__:main',
        ]
    },
    version=version,
    description="Experiments of Uppyyl State Constructor including a CLI tool.",
    long_description=long_description,
    author="Sascha Lehmann",
    author_email="s.lehmann@tuhh.de",
    url="",
    install_requires=[
        'uppyyl_state_constructor',
        'matplotlib~=3.3.0',
        'numpy==1.18.1',
        'pytest==5.3.5',
        'colorama~=0.4.3'
    ],
)
