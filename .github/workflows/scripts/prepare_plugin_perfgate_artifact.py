#!/usr/bin/env python3
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

"""Prepare a Plugin-targeted copy of a same-spec perfgate artifact."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

CORE_REPOSITORY = "vLLM-HUST/vllm-hust"
PLUGIN_REPOSITORY = "vLLM-HUST/vllm-ascend-hust"
LOWERCASE_HEX = frozenset("0123456789abcdef")


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _repository(value: Any, *, field: str, expected: str) -> str:
    normalized = str(value or "").strip()
    if normalized.lower() != expected.lower():
        raise ValueError(f"{field} must be {expected}, got {normalized or 'unset'}")
    return normalized


def _sha(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if len(normalized) != 40 or any(character not in LOWERCASE_HEX for character in normalized):
        raise ValueError(f"{field} must be a full lowercase 40-character Git SHA")
    return normalized


def _single_line(value: Any, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized or any(character in normalized for character in "\0\r\n"):
        raise ValueError(f"{field} must be a non-empty single-line value")
    return normalized


def prepare_plugin_artifact(
    payload: dict[str, Any],
    *,
    target_repository: str,
    target_ref: str,
    target_sha: str,
    github_server_url: str = "https://github.com",
) -> dict[str, Any]:
    """Return a validated copy whose top-level target is the Plugin commit."""
    target_repository = _repository(
        target_repository,
        field="target_repository",
        expected=PLUGIN_REPOSITORY,
    )
    target_ref = _single_line(target_ref, field="target_ref")
    target_sha = _sha(target_sha, field="target_sha")
    github_server_url = _single_line(github_server_url, field="github_server_url").rstrip("/")
    if not github_server_url.startswith("https://"):
        raise ValueError("github_server_url must be an HTTPS URL")

    prepared = copy.deepcopy(_object(payload, field="artifact"))
    metadata = _object(prepared.get("metadata"), field="metadata")
    runtime = _object(metadata.get("runtime_provenance"), field="metadata.runtime_provenance")
    engine = _object(runtime.get("engine"), field="runtime_provenance.engine")
    plugin = _object(runtime.get("plugin"), field="runtime_provenance.plugin")

    core_repository = _repository(
        metadata.get("github_repository"),
        field="metadata.github_repository",
        expected=CORE_REPOSITORY,
    )
    core_sha = _sha(metadata.get("git_commit"), field="metadata.git_commit")
    engine_repository = _repository(
        engine.get("repository"),
        field="runtime_provenance.engine.repository",
        expected=CORE_REPOSITORY,
    )
    engine_sha = _sha(engine.get("commit"), field="runtime_provenance.engine.commit")
    if core_repository.lower() != engine_repository.lower() or core_sha != engine_sha:
        raise ValueError("artifact top-level Core identity does not match runtime engine provenance")

    _repository(
        plugin.get("repository"),
        field="runtime_provenance.plugin.repository",
        expected=target_repository,
    )
    plugin_sha = _sha(plugin.get("commit"), field="runtime_provenance.plugin.commit")
    if plugin_sha != target_sha:
        raise ValueError(
            f"artifact Plugin SHA does not match requested target: expected {target_sha}, got {plugin_sha}"
        )

    metadata["github_repository"] = target_repository
    metadata["github_ref"] = target_ref
    metadata["git_commit"] = target_sha
    metadata["github_commit_url"] = f"{github_server_url}/{target_repository}/commit/{target_sha}"
    return prepared


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path}: {error}") from error
    return _object(payload, field=str(path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-repository", required=True)
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--github-server-url", default="https://github.com")
    args = parser.parse_args()

    if args.source.resolve() == args.output.resolve():
        parser.error("--output must differ from --source")

    try:
        prepared = prepare_plugin_artifact(
            _load_json_object(args.source),
            target_repository=args.target_repository,
            target_ref=args.target_ref,
            target_sha=args.target_sha,
            github_server_url=args.github_server_url,
        )
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
