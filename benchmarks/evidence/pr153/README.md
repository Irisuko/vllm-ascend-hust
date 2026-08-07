# PR 153 mapped-host gather evidence

## 2026-08-03 high-prefix crossover update

[`highprefix-20260803`](highprefix-20260803) contains the two-card ABBA A/B
requested to isolate mapped restore under the repository's established
high-prefix, severely HBM-constrained workload. Eight fresh lifecycles completed
1,600/1,600 requests with identical workload and clean-source provenance.

Mapped reduced mean fixed-workload duration by 1.88%, increased throughput by
1.92%, reduced mean TTFT by 2.33%, and reduced p99 TTFT by 2.18%. The duration
advantage appeared in all four reverse-order pairs and independently on both
cards. Mapped H2D processed 0.90% more bytes while increasing pooled aggregate
bandwidth from 0.99 to 63.85 GB/s and reducing pooled p99 device-event latency
from 689.06 to 7.36 ms. D2H remains asynchronous native copy in both modes;
the small scheduler-dependent D2H count and volume differences are disclosed
rather than treated as exact parity. This supports a scoped end-to-end claim
for this workload, not a default-path recommendation.

## 2026-08-02 pressure update

[`pressure-20260802`](pressure-20260802) contains the corrected production-path
pressure A/B requested in the latest review: three fresh lifecycles per mode,
H2D/D2H bandwidth and device-event latency, TTFT/TPOT/p99, peak NPU memory,
raw per-request and transfer samples, replay tooling, and source provenance.

That update also corrects an intermediate local experiment which had compared
a pinned tiering primary with a whole-primary mmap topology. PR 153 itself did
not change D2H to mapped access: its D2H path remains asynchronous native copy,
and only H2D restore uses mapped gather. At the corrected head, both variants
completed the same 123 D2H jobs and 28,764,536,832 D2H bytes; aggregate D2H
bandwidth was neutral (+1.17% mapped), while mapped H2D bandwidth improved from
3.34 to 29.23 GB/s. The end-to-end three-run mean favored mapped, but the gain
was not consistent across every reverse-order pair, so the feature remains
opt-in experimental infrastructure.

This directory records the focused validation used for PR 153 after addressing
review feedback.  The code under test is the clean, detached local
`vllm-ascend-hust` commit
`3b029f6d54694175e834488bed2f01c2c063a7ce`, with Git tree
`ebdf2c042b344c9c3479978400b9d8a348dce802`, paired with the clean
`vllm-hust` commit `a30addc7548a9a8b9b3323a7bc3eb7d7c4895d1c`.

The new host did not have the user's SSH key, so the already-authorized GitHub
connector recreated the signed-off commit metadata in the personal fork.  The
published code parent is `a77efd8828acaa1555ee607a08b33bcfda6e13de`;
its Git tree is exactly the same `ebdf2c0...` tree tested here.  The following
published commit changes evidence only.

## Runtime

- Ascend 910B2, physical devices 0 and 1 (logical `npu:0` and `npu:1`)
- CANN 9.0.0
- Python 3.12.13
- PyTorch 2.10.0+cpu and torch-npu 2.10.0
- Podman image
  `quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, image ID
  `sha256:0fc116f43369c0dd71bc253dff83d7702dc44747d10ca1c8e9569e7265e10731`

The live source paths in the container were `/workspace/vllm` and
`/workspace/vllm-ascend`.  See
[`validation/repository-state.txt`](validation/repository-state.txt),
[`validation/runtime-versions.json`](validation/runtime-versions.json), and
[`validation/container-runtime.txt`](validation/container-runtime.txt).

Host-side `npu-smi` snapshots are included before initialization, while the
task process was initialized, while the benchmark was running, and after all
runtime work.  The running snapshots show only this task on the selected
devices; the final snapshot shows no device processes.

## Build and static validation

The targeted custom operator was packaged with:

```bash
cd /workspace/vllm-ascend/csrc
bash build.sh --pkg --ops=kv_cache_block_gather \
  --soc=ascend910b -j16 -O2
```

The full `vllm_ascend_C` extension and `vllm_ascend_kernels` library were then
configured from the exact source checkout and built with CMake.  The complete
configuration, commands, result, and artifact hashes are in
[`validation/build-validation.txt`](validation/build-validation.txt).

Ruff 0.14.0 passed over all 14 changed Python files:

```bash
uv tool run --from ruff==0.14.0 ruff check <changed Python files>
uv tool run --from ruff==0.14.0 ruff format --check <changed Python files>
```

Result: all checks passed and all 14 files were already formatted.  See
[`validation/ruff.txt`](validation/ruff.txt).

## Correctness and lifecycle validation

Focused host tests:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend \
pytest -q --confcutdir=/workspace/vllm-ascend/tests/ut \
  tests/ut/_tools/test_benchmark_kv_gather_vs_span.py \
  tests/ut/kv_offload/test_experimental_mapped.py \
  tests/ut/ops/test_kv_cache_block_gather_int8_source.py \
  tests/ut/test_custom_op_package.py
```

Result: **33 passed**.  The handler tests include partial registration
rollback, submission failure with possible work in flight, failed completion
event recording and synchronization, unregister retry, and idempotent
shutdown.  See [`validation/focused-unit.txt`](validation/focused-unit.txt).

One-device NPU tests:

```bash
PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend \
pytest -q \
  --confcutdir=/workspace/vllm-ascend/tests/e2e/pull_request/one_card \
  tests/e2e/pull_request/one_card/test_experimental_mapped_offload.py \
  tests/e2e/pull_request/one_card/test_kv_cache_block_gather.py \
  tests/e2e/pull_request/one_card/test_mapped_host_pool_registration.py
```

Result: **9 passed**.  See
[`validation/npu-one-card-e2e.txt`](validation/npu-one-card-e2e.txt).

Two-device validation of the new cross-device ID-tensor guards:

```bash
PYTHONPATH=/workspace/vllm:/workspace/vllm-ascend \
pytest -q \
  --confcutdir=/workspace/vllm-ascend/tests/e2e/pull_request/two_card \
  tests/e2e/pull_request/two_card/test_kv_cache_block_gather_devices.py
```

Result: **2 passed**, covering mismatched source and destination block-ID
tensors.  See
[`validation/npu-two-card-e2e.txt`](validation/npu-two-card-e2e.txt).

The standalone NPU smoke test also passed; see
[`validation/npu-smoke.txt`](validation/npu-smoke.txt).

## Exact-head microbenchmark

The two directories under [`microbench`](microbench) contain manifests,
per-iteration raw samples (`results.jsonl` and `results.csv`), and summaries
for both backend orders.  Each run used:

- one shared pinned host allocation for mapped gather and span copy;
- two tensor parts (K and V), 512 selected blocks, and 4096 host/device blocks;
- 4, 16, and 64 KiB blocks;
- requested span lengths 1, 2, 4, 8, 16, 32, 64, and 512;
- 5 warmups and 30 measured iterations per backend and case; and
- the exact extension and OPAPI artifacts whose hashes appear in each
  manifest.

The full replayable command is stored in each manifest.  The runs are
explicitly labeled `python-per-span-microbenchmark` and
`production_backend_equivalent: false`: the span baseline intentionally issues
one Python-level copy per span and is **not** the native production transfer
backend.

The mapped-first manifest reports the `vllm-ascend` checkout as dirty only
because the preceding span-first run had already written its untracked result
directory.  Its `status_porcelain` list contains those four result files and no
source change.  The clean source state before and after validation is recorded
in `validation/repository-state.txt`, and both runs used the same Git tree named
at the top of this document.

Across both orders, mapped gather was strongly faster for highly fragmented
cases.  For example, 512 individual 16 KiB spans measured 18.48-18.57 GB/s for
mapped gather versus 0.39-0.41 GB/s for the Python per-span baseline.  At the
contiguous end, the result was close or mixed: for 4 KiB blocks the mapped path
was about 1% slower, while the larger block cases favored mapped gather in
these two observations.  The raw samples should be used rather than treating
the 10% decision column as a production claim.

## Historical end-to-end observation

[`mooncake-p256-repeated-observation.json`](mooncake-p256-repeated-observation.json)
is intentionally marked as historical and was **not rerun at the review-fix
head**.  It records all four observed repetitions from two same-card,
reverse-order pairs, their source commits, workload hashes, and result hashes.

For those exact runs, mapped restore reduced measured H2D handler time by
85.54%, but the mean end-to-end duration changed by only -0.058% (208.77 s
mapped versus 208.89 s span).  The 0.12 s mean difference was smaller than the
0.81-1.57 s within-variant ranges, so the end-to-end observation is neutral.
No broader performance claim is made from those four samples.

## Integrity

[`SHA256SUMS`](SHA256SUMS) covers every evidence file except itself.
