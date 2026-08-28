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
#

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "docker" / "source_dev_container.sh"


def _ascend_tree(tmp_path: Path) -> Path:
    root = tmp_path / "vllm-ascend"
    package = root / "vllm_ascend"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").touch()
    return root


def _git_vllm_tree(path: Path) -> tuple[Path, str]:
    package = path / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "test fixture",
        ],
        check=True,
    )
    sha = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return path, sha


def _run_shell(
    tmp_path: Path,
    command: str,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env["VLLM_ASCEND_REPO"] = str(_ascend_tree(tmp_path))
    if env:
        run_env.update(env)
    return subprocess.run(
        ["bash", "-c", 'source "$1"; eval "$2"', "bash", str(TOOL_PATH), command],
        env=run_env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_repository_discovery_has_structured_error(tmp_path: Path):
    script = tmp_path / "outside-git" / "source_dev_container.sh"
    script.parent.mkdir()
    shutil.copy2(TOOL_PATH, script)
    env = os.environ.copy()
    env.pop("VLLM_ASCEND_REPO", None)

    result = subprocess.run(
        ["bash", str(script), "status"],
        cwd=tmp_path,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == ("error: cannot locate the vllm-ascend checkout; set VLLM_ASCEND_REPO\n")


def test_invalid_local_vllm_repo_has_structured_error(tmp_path: Path):
    missing = tmp_path / "missing-vllm"
    result = _run_shell(
        tmp_path,
        "prepare_vllm_source",
        env={"VLLM_REPO": str(missing)},
    )

    assert result.returncode == 1
    assert result.stderr == f"error: VLLM_REPO is not a directory: {missing}\n"


def test_cache_path_must_be_a_directory(tmp_path: Path):
    cache_file = tmp_path / "cache-file"
    cache_file.touch()
    result = _run_shell(
        tmp_path,
        "prepare_vllm_source",
        env={"VLLM_CACHE_DIR": str(cache_file)},
    )

    assert result.returncode == 1
    assert result.stderr == (f"error: VLLM_CACHE_DIR exists but is not a directory: {cache_file}\n")


def test_local_checkout_records_local_ref_and_commit(tmp_path: Path):
    vllm_repo, expected_sha = _git_vllm_tree(tmp_path / "vllm")

    result = _run_shell(
        tmp_path,
        'prepare_vllm_source; printf "%s %s\\n" "$VLLM_SOURCE_REF" "$VLLM_SHA"',
        env={"VLLM_REPO": str(vllm_repo)},
    )

    assert result.returncode == 0
    assert result.stdout == f"local {expected_sha}\n"


def test_managed_cache_fetches_exact_requested_commit(tmp_path: Path):
    remote, expected_sha = _git_vllm_tree(tmp_path / "remote-vllm")
    cache = tmp_path / "cache" / "vllm"

    result = _run_shell(
        tmp_path,
        'prepare_vllm_source; printf "%s %s\\n" "$VLLM_SOURCE_REF" "$VLLM_SHA"',
        env={
            "VLLM_REMOTE": str(remote),
            "VLLM_REF": expected_sha,
            "VLLM_CACHE_DIR": str(cache),
        },
    )

    assert result.returncode == 0
    assert result.stdout == f"{expected_sha} {expected_sha}\n"
    assert (
        subprocess.check_output(
            ["git", "-C", str(cache), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        == expected_sha
    )


def test_managed_cache_refuses_dirty_checkout(tmp_path: Path):
    remote, expected_sha = _git_vllm_tree(tmp_path / "remote-vllm")
    cache = tmp_path / "cache" / "vllm"
    env = {
        "VLLM_REMOTE": str(remote),
        "VLLM_REF": expected_sha,
        "VLLM_CACHE_DIR": str(cache),
    }
    assert _run_shell(tmp_path, "prepare_vllm_source", env=env).returncode == 0
    dirty_file = cache / "local-change"
    dirty_file.touch()

    result = _run_shell(tmp_path, "prepare_vllm_source", env=env)

    assert result.returncode == 1
    assert result.stderr == f"error: managed vLLM cache is dirty: {cache}\n"


def test_image_reference_resolves_to_content_id(tmp_path: Path):
    digest = "0123456789abcdef" * 4
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == image && $2 == inspect ]]; then\n"
        f"    echo sha256:{digest}\n"
        "else\n"
        "    exit 99\n"
        "fi\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    result = _run_shell(
        tmp_path,
        'resolve_image; printf "%s\\n" "$IMAGE_ID"',
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0
    assert result.stdout == f"sha256:{digest}\n"


def test_raw_podman_image_id_is_normalized(tmp_path: Path):
    raw_image_id = "0" * 64
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $1 == image && $2 == inspect ]]; then\n"
        f"    echo {raw_image_id}\n"
        "else\n"
        "    exit 99\n"
        "fi\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    result = _run_shell(
        tmp_path,
        'resolve_image; printf "%s\\n" "$IMAGE_ID"',
        env={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
    )

    assert result.returncode == 0
    assert result.stdout == f"sha256:{raw_image_id}\n"


def test_custom_ops_verification_runs_the_live_probe(tmp_path: Path):
    result = _run_shell(
        tmp_path,
        'docker() { printf "%s\\n" "$*"; }; verify_custom_ops',
        env={"CONTAINER_NAME": "source-dev-custom-ops-test"},
    )

    assert result.returncode == 0
    assert result.stdout == (
        "exec -i source-dev-custom-ops-test bash /workspace/repo/tools/docker/verify_baked_custom_ops.sh\n"
    )


def test_failed_new_container_validation_removes_only_that_container(tmp_path: Path):
    docker_log = tmp_path / "docker.log"
    result = _run_shell(
        tmp_path,
        (
            "verify_custom_ops() { return 1; }; "
            'docker() { printf "%s\\n" "$*" >> "$DOCKER_LOG"; }; '
            "validate_new_container_custom_ops source-dev-broken"
        ),
        env={"DOCKER_LOG": str(docker_log)},
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert docker_log.read_text(encoding="utf-8") == "rm -f source-dev-broken\n"
    assert result.stderr == (
        "error: baked custom-op validation failed; removed newly created container source-dev-broken\n"
    )


def test_duplicate_npu_device_is_rejected(tmp_path: Path):
    result = _run_shell(
        tmp_path,
        "parse_npu_devices",
        env={"NPU_DEVICES": "2,2"},
    )

    assert result.returncode == 1
    assert result.stderr == "error: duplicate NPU device 2 in NPU_DEVICES=2,2\n"
