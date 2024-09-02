from setuptools import setup, find_packages

setup(
    name='deep3dface',
    version='0.0.1',
    description='deep3dface package', 
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',   
    ],
)