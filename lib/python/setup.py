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
    version='0.1.0',
    author='Flame Maintainers',
    author_email='flame-github-owners@cisco.com',
    include_package_data=True,
    packages=find_packages(),
    data_files=[],
    scripts=['scripts/flame-config'],
    url='https://github.com/cisco-open/flame/',
    license='LICENSE.txt',
    description="This package is a python library"
    " to run ML workloads in the flame system",
    long_description=open('README.md').read(),
    install_requires=[
        'aiostream',
        'boto3',
        'cloudpickle',
        'diskcache',
        'mlflow==2.0.1',
        'paho-mqtt',
        'protobuf==3.19.5',
        'grpcio==1.51.1',
        'pydantic',
    ],
    extras_require={
        'dev': [
            'pre-commit',
            'black',
            'flake8',
            'bandit',
            'mypy',
            'isort',
        ],
    },
)
