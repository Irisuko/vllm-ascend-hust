#!/usr/bin/env bash

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

die() {
    echo "error: $*" >&2
    exit 1
}

compile_mode=${COMPILE_CUSTOM_KERNELS:-}
custom_opp_path=${ASCEND_CUSTOM_OPP_PATH:-}

if [[ $compile_mode == 0 ]]; then
    [[ -z $custom_opp_path ]] || die \
        "COMPILE_CUSTOM_KERNELS=0 but ASCEND_CUSTOM_OPP_PATH is still exported: $custom_opp_path"
    echo "custom_ops.mode=disabled"
    echo "custom_ops.compile_custom_kernels=0"
    echo "custom_ops.status=not-built"
    exit 0
fi

[[ -z $compile_mode || $compile_mode == 1 ]] || \
    die "unsupported COMPILE_CUSTOM_KERNELS value: $compile_mode"
[[ -n $custom_opp_path ]] || die \
    "custom ops are not activated: ASCEND_CUSTOM_OPP_PATH is empty"

vendor_path=
header_rel=op_api/include/aclnnop/aclnn_moe_init_routing_custom.h
library_rel=op_api/lib/libcust_opapi.so
IFS=: read -r -a vendor_candidates <<<"$custom_opp_path"
for candidate in "${vendor_candidates[@]}"; do
    [[ -d $candidate ]] || continue
    candidate=$(cd -- "$candidate" && pwd -P)
    if [[ -f $candidate/$header_rel && -f $candidate/$library_rel ]]; then
        vendor_path=$candidate
        break
    fi
done
[[ -n $vendor_path ]] || die \
    "no activated custom-op vendor contains $header_rel and $library_rel: $custom_opp_path"

header_path=$vendor_path/$header_rel
library_path=$vendor_path/$library_rel
library_dir=$(cd -- "$(dirname -- "$library_path")" && pwd -P)
library_active=0
IFS=: read -r -a library_candidates <<<"${LD_LIBRARY_PATH:-}"
for candidate in "${library_candidates[@]}"; do
    [[ -d $candidate ]] || continue
    candidate=$(cd -- "$candidate" && pwd -P)
    if [[ $candidate == "$library_dir" ]]; then
        library_active=1
        break
    fi
done
((library_active == 1)) || die \
    "custom op API library is not activated in LD_LIBRARY_PATH: $library_dir"

custom_version_file=$vendor_path/version.info
[[ -f $custom_version_file ]] || die \
    "custom-op version metadata is missing: $custom_version_file"
custom_version=$(sed -n 's/^custom_opp_compiler_version=//p' "$custom_version_file")
[[ -n $custom_version ]] || die \
    "custom-op compiler version is missing from $custom_version_file"

cann_version_file=${1:-}
if [[ -z $cann_version_file ]]; then
    for candidate in \
        "${ASCEND_OPP_PATH:-}/version.info" \
        "${ASCEND_HOME_PATH:-}/opp/version.info" \
        /usr/local/Ascend/ascend-toolkit/latest/opp/version.info \
        /usr/local/Ascend/cann-*/opp/version.info; do
        if [[ -f $candidate ]]; then
            cann_version_file=$candidate
            break
        fi
    done
fi
[[ -f $cann_version_file ]] || die "cannot locate CANN version metadata"
cann_version=$(sed -n 's/^Version=//p' "$cann_version_file")
[[ -n $cann_version ]] || die "CANN version is missing from $cann_version_file"
[[ $custom_version == "$cann_version" ]] || die \
    "custom-op compiler version $custom_version does not match CANN $cann_version"

version_sha256=$(sha256sum "$custom_version_file" | awk '{print $1}')
header_sha256=$(sha256sum "$header_path" | awk '{print $1}')
library_sha256=$(sha256sum "$library_path" | awk '{print $1}')

echo "custom_ops.mode=enabled"
echo "custom_ops.compile_custom_kernels=${compile_mode:-inferred-enabled}"
echo "custom_ops.vendor_path=$vendor_path"
echo "custom_ops.library_path=$library_path"
echo "custom_ops.custom_version=$custom_version"
echo "custom_ops.cann_version=$cann_version"
echo "custom_ops.version_sha256=$version_sha256"
echo "custom_ops.header_sha256=$header_sha256"
echo "custom_ops.library_sha256=$library_sha256"
echo "custom_ops.status=ready"
