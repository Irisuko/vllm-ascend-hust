#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Capability smoke gate (issue #198): fail closed on non-whitelisted fallbacks.

Every ``record_capability(<CAP_*>, STATUS_FALLBACK, ...)`` on a Dense hot path
must carry a whitelist entry with an owner and a removal condition.  Adding a
new fallback without whitelisting it makes this gate fail, so a runtime
downgrade can no longer live on forever as an anonymous warning.
"""

import re
from pathlib import Path

import vllm_ascend.diagnostics.capability_manifest as cm

ROOT = Path(__file__).resolve().parents[3]

# Instrumented Dense hot-path modules.
SOURCE_FILES = [
    ROOT / "vllm_ascend/worker/worker.py",
    ROOT / "vllm_ascend/sample/sampler.py",
    ROOT / "vllm_ascend/utils.py",
    ROOT / "vllm_ascend/compilation/passes/qknorm_rope_fusion_pass.py",
]

# Fallback whitelist: capability -> (owner, removal condition).
# Statuses other than FALLBACK (enabled / unavailable / disabled_by_policy)
# are not subject to this gate.
FALLBACK_WHITELIST = {
    cm.CAP_RUNNER_V2_MODEL_RUNNER: (
        "SuccinctPaul",
        "enable V2 model runner on vllm 0.23.0 (drop the vllm_version_is('0.23.0') guard)",
    ),
    cm.CAP_SAMPLER_TOP_K_TOP_P: (
        "SuccinctPaul",
        "ship aclnnApplyTopKTopPCustom symbols in every release/source-dev image",
    ),
    cm.CAP_SAMPLER_PENALTY_TRITON: (
        "SuccinctPaul",
        "ship Triton-Ascend in the base image used by Dense serving",
    ),
    cm.CAP_FUSION_QKNORM_ROPE: (
        "SuccinctPaul",
        "register QKNorm/RoPE custom ops for every supported model dtype/head_dim",
    ),
}

# Matches `record_capability(<CAP_CONST>, <STATUS_CONST>, ...)` including the
# multi-line form used across the instrumented modules.
_CALL_RE = re.compile(r"record_capability\(\s*([A-Z][A-Z0-9_]*)\s*,\s*(STATUS_[A-Z_]+)")


def _scan_fallbacks() -> dict[str, list[str]]:
    """capability -> list of source files that record it as FALLBACK."""
    fallbacks: dict[str, list[str]] = {}
    for path in SOURCE_FILES:
        src = path.read_text(encoding="utf-8")
        for cap_name, status_name in _CALL_RE.findall(src):
            cap = getattr(cm, cap_name, None)
            if cap is None:
                raise AssertionError(f"{path}: unknown capability constant {cap_name}")
            status = getattr(cm, status_name, None)
            if status != cm.STATUS_FALLBACK:
                continue
            fallbacks.setdefault(cap, []).append(str(path))
    return fallbacks


def test_manifest_records_all_six_known_capabilities():
    """Every tracked Dense hot-path capability must appear in the instrumented code."""
    wired = set()
    for path in SOURCE_FILES:
        src = path.read_text(encoding="utf-8")
        for cap_name, _ in _CALL_RE.findall(src):
            cap = getattr(cm, cap_name, None)
            if cap is not None:
                wired.add(cap)
    assert {
        cm.CAP_RUNNER_V2_MODEL_RUNNER,
        cm.CAP_SAMPLER_TOP_K_TOP_P,
        cm.CAP_SAMPLER_PENALTY_TRITON,
        cm.CAP_FUSION_ADD_RMS_NORM_BIAS,
        cm.CAP_FUSION_QKNORM_ROPE,
        cm.CAP_GRAPH_MODE,
    } <= wired


def test_fail_closed_on_non_whitelisted_fallbacks():
    fallbacks = _scan_fallbacks()
    unknown = sorted(set(fallbacks) - set(FALLBACK_WHITELIST))
    assert not unknown, (
        "non-whitelisted fallback(s) introduced on Dense hot paths: "
        f"{unknown}. Add an entry to FALLBACK_WHITELIST in this file with an "
        "owner and a removal condition before merging."
    )


def test_whitelist_entries_carry_owner_and_removal_condition():
    fallbacks = _scan_fallbacks()
    for cap, (owner, condition) in FALLBACK_WHITELIST.items():
        assert owner, f"{cap}: whitelist entry missing owner"
        assert condition, f"{cap}: whitelist entry missing removal condition"
        assert cap in fallbacks, f"{cap}: whitelist entry for a capability never recorded as FALLBACK"
