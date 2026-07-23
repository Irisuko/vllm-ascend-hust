<!-- markdownlint-disable MD013 MD033 MD041 -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-dark.png">
    <img alt="vLLM Ascend logo" src="https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/logos/vllm-ascend-logo-text-light.png" width="420">
  </picture>
</p>

<h3 align="center">
与 vLLM-HUST 配套维护的 Ascend/NPU 插件 fork
</h3>

<p align="center">
| <a href="https://www.hiascend.com/"><b>关于 Ascend</b></a> |
<a href="https://docs.vllm.ai/projects/ascend/zh-cn/latest/"><b>上游 Ascend 文档</b></a> |
<a href="https://github.com/vllm-project/vllm-ascend"><b>上游 vLLM Ascend</b></a> |
<a href="https://github.com/vLLM-HUST/vllm-hust"><b>HUST 核心 fork</b></a> |
<a href="README.md"><b>English</b></a> |
</p>

# vLLM-Ascend-HUST

`vLLM-Ascend-HUST` 是 HUST 维护的
[`vllm-project/vllm-ascend`](https://github.com/vllm-project/vllm-ascend)
后端插件 fork。它必须和
[`vllm-hust`](https://github.com/vLLM-HUST/vllm-hust) 这个核心 vLLM fork
成对使用、成对测试、成对发布。

本仓库保留上游 vLLM Ascend 插件身份，同时承载 HUST Ascend/NPU 实验所需的本地
补丁、脚本、自定义 kernel 和托管服务集成。

## 来自上游 vLLM Ascend 的能力

上游 `vllm-ascend` 项目为 vLLM 提供 Ascend 后端插件，主要包括：

- NPU 平台注册和设备集成
- Ascend 硬件上的模型执行支持
- 自定义算子和 kernel 集成路径
- worker、attention、sampling、分布式运行时适配
- 在 Ascend 设备上运行 vLLM 的官方文档
- vLLM Ascend SIG 社区支持

通用 Ascend 插件使用说明请优先参考上游文档：
<https://docs.vllm.ai/projects/ascend/zh-cn/latest/>。

## HUST 额外维护什么

HUST 侧改动主要服务本地科研实验和托管 NPU 服务流程：

- 和 `vllm-hust` 的成对兼容。
- 模型加载、工具调用解析、采样、调度、worker 行为等 Ascend 侧补丁。
- 本地自定义 kernel 编译开关和 kernel 开发辅助脚本。
- 单 NPU 本地测试环境脚本。
- 和 dev-hub `manage.sh` 托管服务流程配套的运行时 glue code。
- 上游 merge/version metadata，用来持续暴露 fork drift。

插件应尽量保持清晰。核心 vLLM 行为放在 `vllm-hust`；Ascend-only 运行时行为放在
本仓库。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `vllm_ascend/` | Python 插件包和 Ascend 运行时补丁。 |
| `csrc/` | 可选 C/C++/kernel 源码和第三方 kernel 依赖。 |
| `scripts/` | 本地安装、Ascend 环境和调试辅助脚本。 |
| `tests/` | 插件侧测试和 smoke 覆盖。 |
| `upstream_version.json` | 当前上游锚点和 HUST 版本基线。 |
| `AGENTS.md` | AI 辅助改动必须遵守的工作流规则。 |

## 配套 HUST 仓库

| 仓库 | 作用 |
| --- | --- |
| [`vllm-hust`](https://github.com/vLLM-HUST/vllm-hust) | 核心 vLLM fork。 |
| [`vllm-ascend-hust`](https://github.com/vLLM-HUST/vllm-ascend-hust) | Ascend/NPU 插件 fork。 |
| [`vllm-hust-dev-hub`](https://github.com/vLLM-HUST/vllm-hust-dev-hub) | 多仓开发工作区、托管服务脚本和 NPU smoke test 入口。 |
| [`vllm-hust-benchmark`](https://github.com/vLLM-HUST/vllm-hust-benchmark) | Benchmark 编排和结果导出。 |
| [`vllm-hust-perf-analyzer`](https://github.com/vLLM-HUST/vllm-hust-perf-analyzer) | 离线 profiler timeline 分析。 |

不要随意把这个插件装到不匹配的 vLLM checkout 上。这样容易掩盖 ABI、API 或运行时
契约的不一致。除非分支说明另有要求，否则两个 HUST 仓库都应使用配套的 `main`。

## 版本规则

本仓库和 `vllm-hust` 使用同一套上游锚定版本规则：

```text
<上游 release>.post1.dev<HUST-only commit 数>+g<短 sha>
```

`upstream_version.json` 是版本锚点来源：

- `upstream_commit`：已经包含进 fork commit graph 的精确上游提交。
- `upstream_version`：上游兼容版本号；如果上游处于 rc，就保留 rc 后缀。
- `release_version`：去掉 rc 后缀后的上游 release 行。

每次完成上游同步后，都要确认 fork 对上游 behind 为 0：

```bash
git fetch upstream main
git rev-list --left-right --count origin/main...upstream/main
# <HUST-only commits>  0
```

左侧数字是 HUST-only delta；右侧数字应该是 `0`。

## 开发安装

统一使用 `uv` 和配套 `vllm-hust` 的虚拟环境。不要使用系统 `python3` 或裸 `pip`。

先安装核心 fork：

```bash
cd /path/to/vllm-hust
uv venv --python 3.12
source .venv/bin/activate
uv pip install \
  -r requirements/common.txt \
  -r /path/to/vllm-ascend-hust/requirements.txt \
  -r requirements/build/empty.txt \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
  --index-strategy unsafe-best-match
VLLM_TARGET_DEVICE=empty uv pip install -e . \
  --no-build-isolation --no-deps
```

第一条命令在同一环境中安装 core 通用运行时依赖、Ascend 选定的 Torch/NPU
运行时，以及 `empty` 目标所需且不含 Torch 的构建工具。随后安装 core editable
包时可直接复用这些依赖，不会选择 CUDA wheel，也不会触发通用 PEP 517 中的
Torch 2.11 构建依赖。

再安装 Ascend 插件：

```bash
cd /path/to/vllm-ascend-hust
COMPILE_CUSTOM_KERNELS=0 uv pip install -e . --no-deps
```

插件仍使用正常的隔离构建。由于上一步已经统一解析并安装目标运行时依赖，
这里使用 `--no-deps` 不会遗漏运行时包。

在 HUST 本地机器上，优先使用仓库自带安装脚本：

```bash
cd /path/to/vllm-ascend-hust
bash scripts/install_local_ascend_plugin.sh /path/to/vllm-ascend-hust
```

如果当前改动涉及自定义 kernel，再根据本机 Ascend/CANN 环境设置对应编译变量。

## Ascend 环境

单设备本地测试可以先加载脚本：

```bash
cd /path/to/vllm-ascend-hust
source scripts/use_single_ascend_env.sh /usr/local/Ascend/ascend-toolkit/latest
```

然后确认插件能和配套 vLLM checkout 一起导入：

```bash
VLLM_PLUGINS=ascend \
VLLM_TARGET_DEVICE=npu \
.venv/bin/python -c "import vllm, vllm_ascend; print(vllm.__version__)"
```

在共享机器上只能使用分配给当前任务的设备。当前 HUST smoke workflow 默认只允许使用
NPU 1，除非操作者明确指定其他设备。

## 托管 NPU Smoke Test

接近生产服务的测试必须通过 dev-hub 启动：

```bash
cd /path/to/vllm-hust-dev-hub
./manage.sh status
./manage.sh restart
./manage.sh health --json
```

这样可以保证环境变量、设备选择、端口、日志和清理流程都和 HUST 实验服务一致。

## 校验清单

README-only 改动：

```bash
git diff --check -- README.md README.zh.md
```

插件 Python 改动：

```bash
.venv/bin/python -m py_compile path/to/file.py
pre-commit run --files path/to/file.py
```

上游 merge：

```bash
git diff --name-only --diff-filter=U
git rev-list --left-right --count origin/main...upstream/main
git submodule status --recursive
```

涉及 NPU 运行时的改动，还要通过 dev-hub 的 `manage.sh` 在 NPU 1 上做 smoke test。

## 上游同步流程

1. fetch upstream，从 `origin/main` 拉出 staging 分支。
2. 优先把 `upstream/main` 真实 merge 进 staging 分支。
3. 解决冲突时，Ascend-only delta 留在本仓库；核心行为尽量放回 `vllm-hust`。
4. 更新 `upstream_version.json` 和派生包版本。
5. 校验导入、语法、submodule 和 NPU smoke 行为。
6. 合并 staging PR 后拉取 `main`，确认对上游 behind 为 0。

日常同步尽量不要继续堆 cherry-pick/backport；能真实 merge upstream graph 时就真实
merge。

## 文档与社区

- 上游 vLLM Ascend 文档：<https://docs.vllm.ai/projects/ascend/zh-cn/latest/>
- 上游 vLLM Ascend 仓库：<https://github.com/vllm-project/vllm-ascend>
- 上游 vLLM 仓库：<https://github.com/vllm-project/vllm>
- HUST 核心 fork：<https://github.com/vLLM-HUST/vllm-hust>
- HUST 组织：<https://github.com/vLLM-HUST>
- HUST agent 工作流：[`AGENTS.md`](AGENTS.md)

English documentation is available in [`README.md`](README.md).

## License

本仓库继承上游 vLLM Ascend 的许可协议。详情见 [`LICENSE`](LICENSE) 和上游声明。
