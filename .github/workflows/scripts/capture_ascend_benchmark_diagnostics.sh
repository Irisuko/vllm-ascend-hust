#!/bin/bash
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
set -uo pipefail

phase=${1:?Usage: $0 <phase>}
npu_smi_bin=$(command -v npu-smi 2>/dev/null || true)
if [[ -z "$npu_smi_bin" ]]; then
  for candidate in /usr/local/bin/npu-smi /usr/local/sbin/npu-smi /usr/sbin/npu-smi /usr/bin/npu-smi; do
    if [[ -x "$candidate" ]]; then
      npu_smi_bin=$candidate
      break
    fi
  done
fi

echo "::group::Ascend NPU diagnostics ($phase)"
echo "runner=${RUNNER_NAME:-unknown}"
if [[ -z "$npu_smi_bin" ]]; then
  echo "npu-smi is unavailable; HBM diagnostics could not be captured." >&2
elif ! "$npu_smi_bin" info; then
  echo "npu-smi info failed; HBM diagnostics could not be captured." >&2
fi
echo "::endgroup::"
