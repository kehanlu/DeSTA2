from setuptools import setup, find_packages

setup(
    name='desta',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        "transformers",
        "librosa"
    ],
    author='Ke-Han Lu',
    description='An simple module for inferencing DeSTA model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kehanlu/DeSTA2',
)