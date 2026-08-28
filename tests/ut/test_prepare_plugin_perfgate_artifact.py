# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / ".github/workflows/scripts/prepare_plugin_perfgate_artifact.py"
SPEC = importlib.util.spec_from_file_location("prepare_plugin_perfgate_artifact", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)

CORE_SHA = "1" * 40
PLUGIN_SHA = "2" * 40


def _artifact() -> dict:
    return {
        "metrics": {"throughput_tps": 1},
        "metadata": {
            "github_repository": "vLLM-HUST/vllm-hust",
            "github_ref": "main",
            "git_commit": CORE_SHA,
            "github_commit_url": (f"https://github.com/vLLM-HUST/vllm-hust/commit/{CORE_SHA}"),
            "runtime_provenance": {
                "engine": {
                    "repository": "vLLM-HUST/vllm-hust",
                    "ref": "main",
                    "commit": CORE_SHA,
                },
                "plugin": {
                    "engine": "vllm-ascend-hust",
                    "repository": "vLLM-HUST/vllm-ascend-hust",
                    "ref": "main",
                    "commit": PLUGIN_SHA,
                },
            },
        },
    }


def _prepare(payload: dict) -> dict:
    return prepare.prepare_plugin_artifact(
        payload,
        target_repository="vLLM-HUST/vllm-ascend-hust",
        target_ref="main",
        target_sha=PLUGIN_SHA,
    )


def test_prepares_copy_and_changes_only_target_metadata() -> None:
    source = _artifact()
    original = copy.deepcopy(source)

    result = _prepare(source)

    assert source == original
    expected = copy.deepcopy(original)
    expected["metadata"].update(
        {
            "github_repository": "vLLM-HUST/vllm-ascend-hust",
            "github_ref": "main",
            "git_commit": PLUGIN_SHA,
            "github_commit_url": (f"https://github.com/vLLM-HUST/vllm-ascend-hust/commit/{PLUGIN_SHA}"),
        }
    )
    assert result == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("github_repository", "other/core", "metadata.github_repository"),
        ("git_commit", "3" * 40, "top-level Core identity"),
    ],
)
def test_rejects_invalid_core_target_identity(field: str, value: str, message: str) -> None:
    source = _artifact()
    source["metadata"][field] = value

    with pytest.raises(ValueError, match=message):
        _prepare(source)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("repository", "other/plugin", "runtime_provenance.plugin.repository"),
        ("commit", "3" * 40, "Plugin SHA"),
    ],
)
def test_rejects_invalid_plugin_target_identity(field: str, value: str, message: str) -> None:
    source = _artifact()
    source["metadata"]["runtime_provenance"]["plugin"][field] = value

    with pytest.raises(ValueError, match=message):
        _prepare(source)


@pytest.mark.parametrize("target_sha", ["abc123", "A" * 40])
def test_rejects_noncanonical_sha_and_non_https_server(target_sha: str) -> None:
    with pytest.raises(ValueError, match="full lowercase 40-character"):
        prepare.prepare_plugin_artifact(
            _artifact(),
            target_repository="vLLM-HUST/vllm-ascend-hust",
            target_ref="main",
            target_sha=target_sha,
        )

    with pytest.raises(ValueError, match="HTTPS"):
        prepare.prepare_plugin_artifact(
            _artifact(),
            target_repository="vLLM-HUST/vllm-ascend-hust",
            target_ref="main",
            target_sha=PLUGIN_SHA,
            github_server_url="http://github.example",
        )
