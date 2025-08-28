"""Setup script for mini_monai package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mini_monai",
    version="0.1.0",
    author="MONAI Team",
    author_email="info@monai.io",
    description="A lightweight implementation of MONAI for medical image processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Project-MONAI/MONAI",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.20.0",
        "simpleitk>=2.0.0",
        "nibabel>=3.2.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
)