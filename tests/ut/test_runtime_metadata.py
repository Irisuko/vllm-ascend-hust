# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


def _runtime_requirement(name: str) -> Requirement:
    requirements = Path(__file__).resolve().parents[2] / "requirements.txt"
    for line in requirements.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            requirement = Requirement(stripped)
            if requirement.name == name:
                return requirement
    raise AssertionError(f"Missing runtime requirement: {name}")


def test_fastapi_constraint_matches_paired_vllm_hust():
    specifier = _runtime_requirement("fastapi").specifier

    assert Version("0.133.0") in specifier
    assert Version("0.136.3") in specifier
    assert Version("0.137.0") not in specifier
