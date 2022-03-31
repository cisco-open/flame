# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Package setup script."""

from setuptools import find_packages, setup

setup(
    name='flame',
    version='0.0.7',
    author='Myungjin Lee',
    author_email='myungjle@cisco.com',
    include_package_data=True,
    packages=find_packages(),
    # TODO: remove data_files later as it is not essential
    data_files=[('flame/examples/hier_mnist/top_aggregator',
                 ['flame/examples/hier_mnist/top_aggregator/config.json']),
                ('flame/examples/hier_mnist/middle_aggregator', [
                    'flame/examples/hier_mnist/middle_aggregator/config_uk.json',
                    'flame/examples/hier_mnist/middle_aggregator/config_us.json'
                ]),
                ('flame/examples/hier_mnist/trainer', [
                    'flame/examples/hier_mnist/trainer/config_uk.json',
                    'flame/examples/hier_mnist/trainer/config_us.json'
                ]),
                ('flame/examples/mnist/aggregator',
                 ['flame/examples/mnist/aggregator/config.json']),
                ('flame/examples/mnist/trainer',
                 ['flame/examples/mnist/trainer/config.json']),
                ('flame/examples/simple/bar',
                 ['flame/examples/simple/bar/config.json']),
                ('flame/examples/simple/foo',
                 ['flame/examples/simple/foo/config.json'])],
    scripts=['scripts/flame-config'],
    url='https://github.com/cisco-open/flame/',
    license='LICENSE.txt',
    description="This package is a python library"
    " to run ML workloads in the flame system",
    long_description=open('README.md').read(),
    install_requires=[
        'boto3', 'cloudpickle', 'diskcache', 'mlflow', 'paho-mqtt'
    ])
