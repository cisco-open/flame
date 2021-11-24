# Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
# All rights reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Package setup script."""

from setuptools import find_packages, setup

setup(
    name='fledge',
    version='0.0.5',
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
    url='https://github.com/cisco/fledge/',
    license='LICENSE.txt',
    description="This package is a python library"
    " to run ML workloads in the fledge system",
    long_description=open('README.md').read(),
    install_requires=['boto3', 'cloudpickle', 'mlflow', 'paho-mqtt']
)
