from setuptools import find_packages, setup

setup(
    name='fledge',
    version='0.0.1',
    author='An Awesome Coder',
    author_email='myungjle@cisco.com',
    packages=find_packages(),
    scripts=[],
    url='https://wwwin-github.cisco.com/eti/fledge/',
    license='LICENSE.txt',
    description=
    'This package is a python library to run ML workloads in the fledge system',
    long_description=open('README.md').read(),
    install_requires=[
        'cloudpickle', 'keras', 'numpy', 'scikit-learn', 'tensorflow', 'torch'
    ]
)
