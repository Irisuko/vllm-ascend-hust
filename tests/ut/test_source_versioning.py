# SPDX-License-Identifier: Apache-2.0

import ast
import subprocess
from pathlib import Path

from packaging.version import Version
from setuptools_scm import get_version


ROOT = Path(__file__).resolve().parents[2]


def _describe_command_from_setup() -> list[str]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "SCM_GIT_DESCRIBE_COMMAND" for target in node.targets):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return value
    raise AssertionError("SCM_GIT_DESCRIBE_COMMAND is not defined in setup.py")


def test_source_version_ignores_namespaced_mirror_tags() -> None:
    command = _describe_command_from_setup()
    described = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    assert described.startswith("v")
    assert not described.startswith("upstream/")
    Version(get_version(root=ROOT, git_describe_command=command))
