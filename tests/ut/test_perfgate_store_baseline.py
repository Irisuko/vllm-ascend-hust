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

import hashlib
import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / ".github/workflows/scripts/perfgate_store_baseline.sh"
TARGET_SHA = "a" * 40
OTHER_SHA = "b" * 40


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_publication_fixture(tmp_path: Path) -> dict[str, str]:
    result_root = tmp_path / "results"
    submission_dir = result_root / "submissions" / "run"
    benchmark_repo = tmp_path / "benchmark"
    target_repo = tmp_path / "plugin"
    fake_bin = tmp_path / "bin"
    capture = tmp_path / "publisher"
    submission_dir.mkdir(parents=True)
    (benchmark_repo / "src/vllm_hust_benchmark").mkdir(parents=True)
    target_repo.mkdir()
    fake_bin.mkdir()

    warmup = []
    per_run = []
    for run_kind, run_index in (("warmup", 1), ("measured", 1), ("measured", 2), ("measured", 3)):
        relative = f"runs/warmup-{run_index}" if run_kind == "warmup" else f"runs/{run_index}"
        raw_result = result_root / relative / "raw_benchmark_result.json"
        raw_result.parent.mkdir(parents=True)
        raw_result.write_text(json.dumps({"kind": run_kind, "run_index": run_index}) + "\n", encoding="utf-8")
        evidence = {
            "run_index": run_index,
            "raw_result_sha256": _sha256(raw_result),
        }
        if run_kind == "warmup":
            warmup.append(evidence)
        else:
            per_run.append(
                {
                    **evidence,
                    "metrics": {
                        "throughput_tps": float(run_index),
                        "ttft_ms": 10.0,
                        "tbt_ms": 2.0,
                        "error_rate": 0.0,
                        "peak_mem_mb": None,
                    },
                }
            )

    measurement = {
        "schema_version": "perfgate-measurement/v2",
        "strategy": "warmup+primary-median-run",
        "warmup_runs": 1,
        "measured_runs": 3,
        "aggregation": "primary-median-run",
        "selection": {
            "primary_metric": "throughput_tps",
            "sort_direction": "ascending",
            "secondary_sort_key": "run_index",
            "ordered_run_indices": [1, 2, 3],
            "selected_position": 2,
            "selected_run_index": 2,
            "selected_raw_result_sha256": per_run[1]["raw_result_sha256"],
        },
        "warmup": warmup,
        "per_run": per_run,
    }
    measurement_file = submission_dir / "measurement.json"
    measurement_file.write_text(json.dumps(measurement) + "\n", encoding="utf-8")

    baseline_file = submission_dir / "plugin-run_leaderboard.json"
    baseline_file.write_text(
        json.dumps(
            {
                "same_spec": {
                    "scenario": "random-online",
                    "spec_id": "perfgate-ascend-qwen25-3b-910b2",
                    "resolved_spec_hash": "c" * 64,
                },
                "metadata": {
                    "github_repository": "vLLM-HUST/vllm-ascend-hust",
                    "git_commit": TARGET_SHA,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    provenance_file = submission_dir / "perfgate-provenance.json"
    provenance_file.write_text(
        json.dumps(
            {
                "vllm_hust_sha": "1" * 40,
                "vllm_ascend_hust_sha": TARGET_SHA,
                "benchmark_runner_sha": "2" * 40,
                "runtime_manager_sha": "3" * 40,
                "hardware_chip_model": "910B2",
                "cann_version": "8.5.0",
                "torch_version": "2.7.1",
                "torch_npu_version": "2.7.1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _write_executable(
        fake_bin / "git",
        """#!/bin/sh
set -eu
printf '%s\n' "$*" >> "$FAKE_GIT_LOG"
case "$*" in
  *"rev-parse origin/main"*) printf '%s\n' "$FAKE_MAIN_SHA" ;;
esac
""",
    )
    fake_python = fake_bin / "python"
    _write_executable(
        fake_python,
        """#!/bin/sh
set -eu
printf '%s\n' "$@" > "${PUBLISH_CAPTURE}.args"
printf '%s\n' "${GIT_ASKPASS:-}" > "${PUBLISH_CAPTURE}.askpass"
if [ -n "${PERFGATE_BASELINE_WRITER_TOKEN:-}" ]; then
  printf 'present\n' > "${PUBLISH_CAPTURE}.token"
fi
exit "${FAKE_PUBLISH_EXIT_CODE:-0}"
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "PYTHON_BIN": str(fake_python),
            "RUN_ID": "run",
            "RESULT_ROOT": str(result_root),
            "PERFGATE_BASELINE_SOURCE_FILE": str(baseline_file),
            "PERFGATE_MEASUREMENT_FILE": str(measurement_file),
            "PERFGATE_PROVENANCE_FILE": str(provenance_file),
            "PERFGATE_BENCHMARK_REPO_DIR": str(benchmark_repo),
            "PERFGATE_TARGET_GIT_REPOSITORY": str(target_repo),
            "PERFGATE_TARGET_REPOSITORY": "vLLM-HUST/vllm-ascend-hust",
            "PERFGATE_TARGET_SHA": TARGET_SHA,
            "PERFGATE_BASELINE_WRITER_TOKEN": "test-writer-token",
            "RUNNER_TEMP": str(tmp_path),
            "FAKE_MAIN_SHA": TARGET_SHA,
            "FAKE_GIT_LOG": str(tmp_path / "git.log"),
            "PUBLISH_CAPTURE": str(capture),
        }
    )
    return env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _capture_path(env: dict[str, str], suffix: str) -> Path:
    return Path(f"{env['PUBLISH_CAPTURE']}.{suffix}")


def test_publishes_tip_with_scoped_askpass_and_provenance(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    arguments = _capture_path(env, "args").read_text(encoding="utf-8").splitlines()
    assert arguments[:3] == ["-m", "vllm_hust_benchmark.perfgate_baselines", "publish"]
    assert "--runtime-manager-sha" in arguments
    assert "--update-latest-pointer" in arguments
    assert env["PERFGATE_BASELINE_WRITER_TOKEN"] not in arguments
    assert _capture_path(env, "token").read_text(encoding="utf-8") == "present\n"
    askpass = Path(_capture_path(env, "askpass").read_text(encoding="utf-8").strip())
    assert not askpass.exists()


def test_publishes_exact_baseline_without_pointer_when_main_advanced(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)
    env["FAKE_MAIN_SHA"] = OTHER_SHA

    result = _run(env)

    assert result.returncode == 0, result.stderr
    arguments = _capture_path(env, "args").read_text(encoding="utf-8").splitlines()
    assert "--update-latest-pointer" not in arguments
    assert "Target main advanced" in result.stdout


def test_rejects_missing_writer_token_before_publication(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)
    env["PERFGATE_BASELINE_WRITER_TOKEN"] = ""

    result = _run(env)

    assert result.returncode == 2
    assert "PERFGATE_BASELINE_WRITER_TOKEN is required" in result.stderr
    assert not _capture_path(env, "args").exists()


def test_rejects_artifact_target_mismatch_before_publication(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)
    baseline_file = Path(env["PERFGATE_BASELINE_SOURCE_FILE"])
    payload = json.loads(baseline_file.read_text(encoding="utf-8"))
    payload["metadata"]["git_commit"] = OTHER_SHA
    baseline_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _run(env)

    assert result.returncode == 2
    assert "Artifact SHA mismatch" in result.stderr
    assert not _capture_path(env, "args").exists()


def test_rejects_raw_checksum_mismatch_before_publication(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)
    raw_result = Path(env["RESULT_ROOT"]) / "runs/2/raw_benchmark_result.json"
    raw_result.write_text("tampered\n", encoding="utf-8")

    result = _run(env)

    assert result.returncode == 2
    assert "Measurement source checksum mismatch" in result.stderr
    assert not _capture_path(env, "args").exists()


def test_publisher_failure_propagates_and_removes_askpass(tmp_path: Path) -> None:
    env = _write_publication_fixture(tmp_path)
    env["FAKE_PUBLISH_EXIT_CODE"] = "23"

    result = _run(env)

    assert result.returncode == 23
    askpass = Path(_capture_path(env, "askpass").read_text(encoding="utf-8").strip())
    assert not askpass.exists()
