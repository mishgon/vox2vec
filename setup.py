from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as file:
    requirements = file.read().splitlines()

with open('README.md', encoding='utf-8') as file:
    long_description = file.read()

setup(
    name='vox2vec',
    version='0.0.1',
    description='vox2vec: Self-Superivsed Learning of Voxel-level Representations in Medical Images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/neuro-ml/vox2vec',
    author='M. Goncharov, V. Soboleva',
    author_email='m.goncharov@ira-labs.com',
    packages=find_packages(include=('vox2vec',)),
    python_requires='>=3.6',
    install_requires=requirements,
)
