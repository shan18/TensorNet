import os
import re
from setuptools import setup, find_packages


base_dir = os.path.dirname(os.path.abspath(__file__))


def version():
    with open(os.path.join(base_dir, 'tensornet', '__init__.py')) as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name='torch-tensornet',
    version=version(),
    author='Shantanu Acharya',
    author_email='thegeek.004@gmail.com',
    description='A high-level deep learning library build on top of PyTorch.',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/shan18/TensorNet',
    packages=find_packages(),
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=requirements()
)
