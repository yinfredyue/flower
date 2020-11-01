#!/bin/bash

# Copyright 2020 Adap GmbH. All Rights Reserved.
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
# ==============================================================================
set -e

# echo "RELATIVE path: $(dirname ${BASH_SOURCE[0]})"
# echo "ABOSLUTE path: $( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )"
# echo "DEST: $( cd $( dirname ${BASH_SOURCE[0]} ) && pwd )/../.."

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"/../../
# echo "$(pwd)"

# How to read the command above
# https://stackoverflow.com/q/39340169/9057530
# BASH_SOURCE[0] returns the RELATIVE path of the script
# dirname XXX gives the direction portion of the path. `dir /usr/local` -> /usr/
# && is to execute the two commands sequentially
# So "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" returns the ABSOLUTE path
# cd $(cd $(dirname XXX/src/flower/src/py/flwr_example/quickstart_pytorch) && pwd)/../../..
# Which is the same as 

# Start a Flower server
python3 -m flwr_example.mnist_app.server
