# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

import shlex
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
USE_SINGLE_ASCEND_ENV = REPO_ROOT / "scripts/use_single_ascend_env.sh"


def test_use_single_ascend_env_falls_back_to_cann_set_env_for_tbe() -> None:
    script = USE_SINGLE_ASCEND_ENV.read_text(encoding="utf-8")

    assert "ensure_cann_tbe_env()" in script
    assert "python_can_import_tbe()" in script
    assert "cann_tbe_python_bin()" in script
    assert "append_unique_path_var()" in script
    assert "enrich_cann_python_env()" in script
    assert "import tbe" in script
    assert 'local require_cann_tbe="${HUST_REQUIRE_CANN_TBE:-1}"' in script
    assert "export HUST_ASCEND_TBE_AVAILABLE=1" in script
    assert "export HUST_ASCEND_TBE_AVAILABLE=0" in script
    assert "${PYTHON_BIN:-}" in script
    assert "${ASCEND_HOME_PATH:-}/set_env.sh" in script
    assert "${ASCEND_TOOLKIT_HOME:-}/set_env.sh" in script
    assert "${ASCEND_TOOLKIT_LATEST_HOME:-}/set_env.sh" in script
    assert "${CONDA_PREFIX:-}/Ascend/cann/set_env.sh" in script
    assert "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" in script
    assert "/usr/local/Ascend/ascend-toolkit/set_env.sh" in script
    assert "${ASCEND_HOME_PATH:-}/python/site-packages" in script
    assert "${ASCEND_OPP_PATH:-}/built-in/op_impl/ai_core/tbe" in script
    assert "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages" in script
    assert 'python_prefix="$(cd "$(dirname "${python_bin}")/.." && pwd -P)"' in script
    assert 'append_unique_path_var LD_LIBRARY_PATH "${python_prefix}/lib"' in script
    assert 'source "${set_env_file}" || source_status=$?' in script
    assert "ensure_cann_tbe_env || return 1" in script
    assert 'append_unique_path_var PYTHONPATH "${candidate}"' in script
    assert 'append_unique_path_var LD_LIBRARY_PATH "${candidate}"' in script
    assert "hust_prioritize_conda_runtime_libs" in script
    assert 'if [[ "${require_cann_tbe}" != "1" ]]; then' in script
    assert 'continuing without strict TBE enforcement' in script
    assert 'echo "[ERROR] PYTHONPATH=${PYTHONPATH:-<unset>}" >&2' in script
    assert "Source the correct CANN set_env.sh" in script


def test_conda_runtime_library_priority_removes_stale_duplicate() -> None:
    helper = (REPO_ROOT / "scripts/hust_ascend_manager_helper.sh").read_text(
        encoding="utf-8"
    )

    assert "hust_prioritize_conda_runtime_libs()" in helper
    assert 'IFS=\':\' read -r -a ld_library_path_entries' in helper
    assert '"${entry}" == "${conda_lib_dir}"' in helper
    assert (
        'export LD_LIBRARY_PATH="${conda_lib_dir}'
        '${rebuilt_ld_library_path:+:${rebuilt_ld_library_path}}"'
        in helper
    )
    assert 'conda_libstdcpp="${conda_lib_dir}/libstdc++.so.6"' in helper
    assert (
        'export LD_PRELOAD="${conda_libstdcpp}${LD_PRELOAD:+:${LD_PRELOAD}}"'
        in helper
    )


def test_conda_runtime_library_priority_is_idempotent(tmp_path: Path) -> None:
    conda_prefix = tmp_path / "conda"
    conda_lib = conda_prefix / "lib"
    conda_lib.mkdir(parents=True)
    (conda_lib / "libstdc++.so.6").touch()
    helper = REPO_ROOT / "scripts/hust_ascend_manager_helper.sh"

    command = f"""
source {shlex.quote(str(helper))}
export LD_LIBRARY_PATH=/older/lib:{shlex.quote(str(conda_lib))}:/tail/lib:{shlex.quote(str(conda_lib))}
export LD_PRELOAD=/existing/lib.so
hust_prioritize_conda_runtime_libs {shlex.quote(str(conda_prefix))}
printf 'first_path=%s\n' "$LD_LIBRARY_PATH"
printf 'first_preload=%s\n' "$LD_PRELOAD"
hust_prioritize_conda_runtime_libs {shlex.quote(str(conda_prefix))}
printf 'second_path=%s\n' "$LD_LIBRARY_PATH"
printf 'second_preload=%s\n' "$LD_PRELOAD"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    expected_path = f"{conda_lib}:/older/lib:/tail/lib"
    expected_preload = f"{conda_lib}/libstdc++.so.6:/existing/lib.so"
    assert values["first_path"] == expected_path
    assert values["second_path"] == expected_path
    assert values["first_preload"] == expected_preload
    assert values["second_preload"] == expected_preload


def test_selected_runtime_wins_over_ambient_conda_prefix(tmp_path: Path) -> None:
    ambient_prefix = tmp_path / "ambient-conda"
    selected_prefix = tmp_path / "selected-conda"
    python_prefix = tmp_path / "python-selected"
    for prefix in (ambient_prefix, selected_prefix, python_prefix):
        (prefix / "lib").mkdir(parents=True)
        (prefix / "lib" / "libstdc++.so.6").touch()
    (python_prefix / "bin").mkdir()
    (python_prefix / "bin" / "python").touch()
    helper = REPO_ROOT / "scripts/hust_ascend_manager_helper.sh"

    command = f"""
source {shlex.quote(str(helper))}
export CONDA_PREFIX={shlex.quote(str(ambient_prefix))}
export VLLM_HUST_CONDA_PREFIX={shlex.quote(str(selected_prefix))}
export PYTHON_BIN={shlex.quote(str(python_prefix / "bin" / "python"))}
unset VLLM_HUST_PYTHON_BIN
unset LD_LIBRARY_PATH LD_PRELOAD
hust_prioritize_conda_runtime_libs
printf 'configured_path=%s\n' "$LD_LIBRARY_PATH"
printf 'configured_preload=%s\n' "$LD_PRELOAD"
unset VLLM_HUST_CONDA_PREFIX LD_LIBRARY_PATH LD_PRELOAD
hust_prioritize_conda_runtime_libs
printf 'python_path=%s\n' "$LD_LIBRARY_PATH"
printf 'python_preload=%s\n' "$LD_PRELOAD"
unset PYTHON_BIN LD_LIBRARY_PATH LD_PRELOAD
export VLLM_HUST_PYTHON_BIN={shlex.quote(str(python_prefix / "bin" / "python"))}
hust_prioritize_conda_runtime_libs
printf 'configured_python_path=%s\n' "$LD_LIBRARY_PATH"
printf 'configured_python_preload=%s\n' "$LD_PRELOAD"
"""
    result = subprocess.run(
        ["bash", "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)

    assert values["configured_path"] == str(selected_prefix / "lib")
    assert values["configured_preload"] == str(
        selected_prefix / "lib" / "libstdc++.so.6"
    )
    assert values["python_path"] == str(python_prefix / "lib")
    assert values["python_preload"] == str(
        python_prefix / "lib" / "libstdc++.so.6"
    )
    assert values["configured_python_path"] == str(python_prefix / "lib")
    assert values["configured_python_preload"] == str(
        python_prefix / "lib" / "libstdc++.so.6"
    )
