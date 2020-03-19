from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name='torch-tensornet',
    version='0.0.1a',
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
    install_requires=requirements()
)
