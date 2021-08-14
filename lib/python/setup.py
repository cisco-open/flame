from setuptools import find_packages, setup

setup(
    name='fledge',
    version='0.0.1',
    author='Myungjin Lee',
    author_email='myungjle@cisco.com',
    include_package_data=True,
    packages=find_packages(),
    # TODO: remove data_files later as it is not essential
    data_files=[
        (
            'fledge/examples/hier_mnist/gaggr', [
                'fledge/examples/hier_mnist/gaggr/config.json'
            ]
        ),
        (
            'fledge/examples/hier_mnist/aggregator', [
                'fledge/examples/hier_mnist/aggregator/config_uk.json',
                'fledge/examples/hier_mnist/aggregator/config_us.json'
            ]
        ),
        (
            'fledge/examples/hier_mnist/trainer', [
                'fledge/examples/hier_mnist/trainer/config_uk.json',
                'fledge/examples/hier_mnist/trainer/config_us.json'
            ]
        ),
        (
            'fledge/examples/mnist/aggregator', [
                'fledge/examples/mnist/aggregator/config.json'
            ]
        ),
        (
            'fledge/examples/mnist/trainer', [
                'fledge/examples/mnist/trainer/config.json'
            ]
        ),
        (
            'fledge/examples/simple/bar', [
                'fledge/examples/simple/bar/config.json'
            ]
        ),
        (
            'fledge/examples/simple/foo', [
                'fledge/examples/simple/foo/config.json'
            ]
        )
    ],
    scripts=[],
    url='https://wwwin-github.cisco.com/eti/fledge/',
    license='LICENSE.txt',
    description=
    'This package is a python library to run ML workloads in the fledge system',
    long_description=open('README.md').read(),
    install_requires=['cloudpickle', 'paho-mqtt']
)
