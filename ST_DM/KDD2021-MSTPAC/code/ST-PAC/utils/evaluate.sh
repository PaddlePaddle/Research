#!/bin/bash 
################################################################################
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
################################################################################

function main() {
    infer_data="$1"
    ckpt_version="$2"
    echo "[TRACE] $(date) evaluate infer_data is ${infer_data}, checkpoint_version is ${ckpt_version}" >&2

    #startup script for eval
    #TODO: user-define metric here!!!

    return 0
}

main "$@"
