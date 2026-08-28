#!/usr/bin/env bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from https://github.com/vllm-project/vllm/tree/main/tools
#

set -euo pipefail

scversion="stable"

if [ -d "shellcheck-${scversion}" ]; then
    shellcheck_path="$(pwd)/shellcheck-${scversion}"
    PATH="$PATH:$shellcheck_path"
    export PATH
fi

if ! [ -x "$(command -v shellcheck)" ]; then
    if [ "$(uname -s)" != "Linux" ] || [ "$(uname -m)" != "x86_64" ]; then
        echo "Please install shellcheck: https://github.com/koalaman/shellcheck?tab=readme-ov-file#installing"
        exit 1
    fi

    wget -qO- "https://github.com/koalaman/shellcheck/releases/download/${scversion?}/shellcheck-${scversion?}.linux.x86_64.tar.xz" | tar -xJv
    shellcheck_path="$(pwd)/shellcheck-${scversion}"
    PATH="$PATH:$shellcheck_path"
    export PATH
fi

run_shellcheck() {
    local file
    local status=0
    for file in "$@"; do
        [[ -f "$file" ]] || continue
        if git check-ignore -q "$file"; then
            continue
        fi
        shellcheck -s bash "$file" || status=$?
    done
    return "$status"
}

# pre-commit passes the changed shell files. Keep the no-argument behavior for
# developers who intentionally request a repository-wide scan.
if (( $# > 0 )); then
    run_shellcheck "$@"
    exit
fi

while IFS= read -r -d '' file; do
    run_shellcheck "$file"
done < <(find . -path ./.git -prune -o -name "*.sh" -print0)
