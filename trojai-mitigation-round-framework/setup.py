from setuptools import setup, find_packages

setup(
    name='trojai_mitigation_round',
    version='1.0',
    author='Neil Fendley, Joshua Carney, Trevor Stout',
    description='Framework code for running the TrojAI Mitigation Round',
    url='https://gitlab.jhuapl.edu/trojai_v2/mitigationround',
    python_requires='>=3.8',
    packages=find_packages(include=['trojai_mitigation_round']),
    install_requires=[
        'torch',
        'torchvision',
        'tqdm',
        'pandas',
        'numpy',
        'torchmetrics'
    ],
)