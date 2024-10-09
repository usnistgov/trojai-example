#!/usr/bin/env python

from setuptools import setup

__author__ = "anonymous"
__version__ = "1.0"

setup(
    name="gentle",
    version=__version__,
    description="Library for safe RL projects",
    long_description=open("README.md").read(),
    author=__author__,
    author_email="Jared.Markowitz@jhuapl.edu",
    license="BSD",
    packages=["gentle"],
    keywords="deep reinforcement learning, risk-sensitive RL, constrained RL",
    classifiers=[],
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "scipy",
        "opencv-python",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "python-box",
        "gymnasium",
        "safety-gymnasium",
    ],
)
