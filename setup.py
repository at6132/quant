from setuptools import setup, find_packages

setup(
    name="quant",
    version="0.1.0",
    packages=find_packages() + ['Core', 'Core.indicators'],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "pyyaml",
        "flask",
        "redis",
    ],
) 