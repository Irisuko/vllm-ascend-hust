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

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "_selected_tests.yaml"
RUNNER_LABEL_PATH = REPO_ROOT / ".github" / "workflows" / "scripts" / "runner_label.json"


def test_a2_single_npu_container_uses_runner_scoped_runtime_contract() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    runner_labels = RUNNER_LABEL_PATH.read_text(encoding="utf-8")

    assert '"linux-aarch64-a2b3-1"' in runner_labels
    assert "matrix.group.runner == 'linux-aarch64-a2b3-1'" in workflow
    assert "--device /dev/davinci1:/dev/davinci0" in workflow
    assert "--device /dev/davinci_manager" in workflow
    assert "--device /dev/devmm_svm" in workflow
    assert "--device /dev/hisi_hdc" in workflow
    assert "/usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro" in workflow
    assert "/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro" in workflow
    assert "ASCEND_RT_VISIBLE_DEVICES: ${{ matrix.group.runner == 'linux-aarch64-a2b3-1' && '0' || '' }}" in workflow
    assert "--physical-device-ids" in workflow
    assert "matrix.group.runner == 'linux-aarch64-a2b3-1' && '1' || ''" in workflow


def test_npu_preflight_is_fail_closed_and_runs_before_package_install() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    preflight = "check_ascend_container_runtime.py"

    assert preflight in workflow
    assert workflow.index(preflight) < workflow.index("- name: Install packages")
    preflight_block = workflow[
        workflow.index("- name: Validate Ascend container runtime") : workflow.index("- name: Install packages")
    ]
    assert "continue-on-error" not in preflight_block
    assert "|| true" not in preflight_block
    assert "if: ${{ matrix.group.npu_type != 'cpu' }}" in preflight_block


def test_container_checkout_uses_runner_compatible_node_runtime() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow.count("uses: actions/checkout@v6.0.1") == 2
    assert "uses: actions/checkout@v7" not in workflow


def test_hosted_cpu_checkout_installs_git_before_disabling_submodules() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    install_git = workflow.index("- name: Install git for hosted checkout")
    checkout = workflow.index("- name: Checkout vllm-project/vllm-ascend repo")
    assert install_git < checkout
    install_block = workflow[install_git:checkout]
    assert "matrix.group.hosted_cpu == true" in install_block
    assert "apt-get install -y git" in install_block
    assert "submodules: ${{ matrix.group.hosted_cpu != true && 'recursive' || 'false' }}" in workflow


def test_repo_steps_use_the_checked_out_workspace() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    workspace = "working-directory: ${{ github.workspace }}"

    for step_name in (
        "Rebase on latest",
        "Get csrc hash",
        "Install vllm-project/vllm-ascend with device",
        "Install vllm-project/vllm-ascend no device",
        "Run selected tests with device",
        "Run selected tests without device",
    ):
        step_start = workflow.index(f"- name: {step_name}")
        step_end = workflow.find("\n      - ", step_start + 1)
        assert workspace in workflow[step_start:step_end]

    assert 'git config --global --add safe.directory "$GITHUB_WORKSPACE"' in workflow
    assert "/__w/vllm-ascend/vllm-ascend" not in workflow


def test_hosted_cpu_install_excludes_ascend_only_dependencies() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    install_start = workflow.index("- name: Install vllm-project/vllm-ascend no device")
    install_end = workflow.index("- name: Uninstall Triton for 310P tests", install_start)
    install_block = workflow[install_start:install_end]

    assert (
        "sed -E '/^-r (requirements-lint\\.txt|requirements\\.txt)$/d; "
        "/^(torch-npu|triton-ascend|memfabric_hybrid|memcache_hybrid|xlite)([<=>]|$)/d' requirements-dev.txt"
        in install_block
    )
    assert "torch-npu|triton-ascend|memfabric_hybrid|memcache_hybrid|xlite" in install_block
    assert 'uv pip install -r "$CPU_REQUIREMENTS"' in install_block
    assert "uv pip install --no-deps -e ." in install_block


def test_standalone_a2_runner_does_not_depend_on_cluster_local_package_cache() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "matrix.group.runner == 'linux-aarch64-a2b3-1' && 'https://pypi.org/simple'" in workflow
    install_block = workflow[
        workflow.index("- name: Install packages") : workflow.index("- name: Checkout vllm-project/vllm repo")
    ]
    assert (
        'if [ "${{ matrix.group.hosted_cpu }}" != "true" ] && '
        '[ "${{ matrix.group.runner }}" != "linux-aarch64-a2b3-1" ]; then' in install_block
    )
    assert "cache-service.nginx-pypi-cache.svc.cluster.local:8081" in install_block
