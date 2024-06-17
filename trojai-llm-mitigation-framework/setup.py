from setuptools import setup, find_packages

setup(
    name='trojai_llm_mitigation_round',
    version='1.0',
    author='Neil Fendley, Joshua Carney',
    description='Code for running the TrojAI Mitigation Round',
    long_description='',
    keywords='development, setup, setuptools',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'configargparse',
        'numpy',
        'transformers',
        'peft',
        'datasets',
        'pyyaml'
    ],
)