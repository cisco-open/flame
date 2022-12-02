#!/usr/bin/env bash
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


function macos_installation {
    echo "macos: to be implemented"
}

function ubuntu_installation {
    echo "ubuntu: to be implemented"
}

function amzn2_installation {
    echo "amzn2: to be implemented"
}

function main {
    os=${@:$OPTIND:1}
    shift;

    case $os in
	"macos")
	    echo "Starting installation for Mac OS"
	    macos_installation
	    ;;
	
	"ubuntu")
	    echo "Starting installation for Ubuntu"
	    ubuntu_installation
	    ;;
	"amzn2")
	    echo "Starting installation for Amazon Linux 2"
	    amzn2_installation
	    ;;
	*)
	    echo "usage: ./setup.sh <macos | ubuntu | amzn2>"
	    ;;
    esac
}

main "$@"
