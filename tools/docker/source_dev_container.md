# Live-source Ascend development container

The released vllm-ascend images contain editable installs backed by source
snapshots baked into the image. Mounting only a local vllm-ascend checkout can
therefore silently combine current plugin code with an old vLLM checkout.

`source_dev_container.sh` starts a reproducible toolchain container while
loading both Python packages from explicit source trees:

```text
vllm-hust        -> /workspace/vllm        (read-only)
vllm-ascend-hust -> /workspace/vllm-ascend (read-write)
PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend
```

The script locates the current vllm-ascend repository with Git; it does not
depend on a parent workspace layout.

## Quick start

From anywhere inside the vllm-ascend checkout:

```bash
tools/docker/source_dev_container.sh start
tools/docker/source_dev_container.sh verify
tools/docker/source_dev_container.sh verify-npu
tools/docker/source_dev_container.sh shell
```

The host must provide Docker, `/usr/local/Ascend/driver`,
`/etc/ascend_install.info`, the selected `/dev/davinci<N>` nodes, and
`/dev/davinci_manager`, `/dev/devmm_svm`, and `/dev/hisi_hdc`. The helper checks
these prerequisites before creating a container.

By default, the script fetches the latest `main` from
`https://github.com/vLLM-HUST/vllm-hust.git` into a script-owned checkout below
`${XDG_CACHE_HOME:-$HOME/.cache}`. The resolved commit is recorded on the
container and printed by `status` and `verify`.

The managed checkout is read-only inside the container. The script refuses to
update it if it is dirty, and refuses to take over an existing unmarked Git
checkout at the configured cache path.

The helper also resolves `IMAGE` to a local Docker image ID before creation,
records both the requested reference and resolved ID as labels, and starts the
container by ID. If the same tag later resolves to a different image, `start`
requires `recreate`. For a repeatable toolchain across hosts, set `IMAGE` to an
immutable registry digest instead of a mutable tag.

Docker-style `sha256:<digest>` and Podman-compatible raw 64-character image IDs
are normalized to the same value before the provenance comparison.

## Baked custom-op contract

`start` and `verify` validate the custom-op package owned by the image. When
custom kernels are enabled, the helper fails closed unless all of the following
are true:

- `ASCEND_CUSTOM_OPP_PATH` resolves to a vendor containing the
  `MoeInitRoutingCustom` header and `libcust_opapi.so`;
- the vendor's `op_api/lib` directory is active in `LD_LIBRARY_PATH`; and
- the package compiler version matches the active CANN version.

A successful probe prints the resolved vendor and library paths, package/CANN
versions, and SHA-256 checksums for the version metadata, routing header, and
op API library. This makes repeated containers auditable without copying or
installing generated artifacts into the live source tree. The image path is
immutable, so no per-container mutable installation directory needs cleanup.

Images deliberately built with `COMPILE_CUSTOM_KERNELS=0` print
`custom_ops.status=not-built` and must not export `ASCEND_CUSTOM_OPP_PATH`.
Missing activation metadata, missing artifacts, stale library paths, and CANN
version mismatches make `start` fail. A newly created invalid container is
removed before the command returns an error.

To develop vLLM and vllm-ascend together, select an existing local checkout:

```bash
VLLM_REPO=/path/to/vllm-hust \
    tools/docker/source_dev_container.sh recreate
```

The local checkout is also mounted read-only in the container. Host-side edits
remain visible immediately. Its `vllm_ref=local` and `vllm_sha` labels describe
the checkout when the container was created; later host-side commits and dirty
edits are live but do not rewrite Docker labels.

## Selecting NPUs

`NPU_DEVICES` accepts one physical ID, a comma-separated set, or `all`:

```bash
# One card.
NPU_DEVICES=0 tools/docker/source_dev_container.sh start

# Eight cards.
NPU_DEVICES=0,1,2,3,4,5,6,7 \
    tools/docker/source_dev_container.sh start

# A non-contiguous set.
NPU_DEVICES=2,5 tools/docker/source_dev_container.sh start

# Every /dev/davinci<N> node present on the host.
NPU_DEVICES=all tools/docker/source_dev_container.sh start
```

The physical device set determines the default container name. Inside the
container, selected devices are addressed by their logical IDs from zero.
`verify-npu` checks the visible count and performs a tensor round trip on every
logical device. The script deliberately does not synthesize
`ASCEND_RT_VISIBLE_DEVICES`; Docker device nodes remain the source of truth.

## Updating the managed vLLM baseline

`start` fetches the configured `VLLM_REF`. If an existing container records an
older managed commit, the script asks for an explicit recreation:

```bash
tools/docker/source_dev_container.sh recreate
```

Pin a specific development baseline when required:

```bash
VLLM_REF=<branch-tag-or-commit> \
    tools/docker/source_dev_container.sh recreate
```

The source provenance and NPU gates are intentionally separate. This makes a
wrong checkout distinguishable from a busy or misconfigured device.

## Python versus compiled extensions

Python edits are live immediately. C++ and AscendC edits still require
rebuilding the corresponding extension or custom-op package. Do not fall back
to the extension baked into the image when testing a changed binding: load the
newly built library explicitly or install the current checkout in editable
mode inside the container.

Containers created by the script carry an ownership label. Lifecycle commands
refuse to stop or delete an unmanaged container with the same name.
