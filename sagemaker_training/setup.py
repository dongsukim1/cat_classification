from setuptools import setup, find_packages

setup(
    name="wildlife-classification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "Pillow>=8.0.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0"
    ]
)