# PR 153 matched pressure evidence (2026-08-02)

## Correction to the earlier M0 interpretation

The first local M0 A/B did not represent this PR. It compared
`TieringOffloadingSpec` with a pinned CPU primary against the same spec with a
whole-primary mmap region. That second topology also changed the D2H store
destination, so its D2H regression cannot be attributed to PR 153.

PR 153 has a narrower directional contract. `MappedOffloadingSpec` keeps NPU
to CPU stores on asynchronous
`swap_blocks_batch(..., DIRECTION_D2H)` and changes only CPU to NPU restore to
mapped-host gather. The evidence here tests that actual contract.

## Source under test

- `vllm-ascend-hust`: clean
  `53fa793590107f58a8552a18e889a211fe6b4bd3`, tree
  `74009c6b355e6146645b89c2b4461eaf0a50b578`.
- vLLM-HUST core: `6901c26b23fa60ca1a2450b0999f803f6ee6c335`,
  based on `e4ce33646` with two instrumentation-only commits:
  opt-in transfer event tracing and stable benchmark request IDs.
- Runtime image:
  `quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, image ID
  `sha256:0fc116f43369c0dd71bc253dff83d7702dc44747d10ca1c8e9569e7265e10731`.
- Ascend 910B2, physical NPU 4, CANN 9.0.0, Python 3.12.13,
  PyTorch 2.10.0+cpu, torch-npu 2.10.0.

The replayable core patch is
[`reproduction/core-instrumentation.patch`](reproduction/core-instrumentation.patch).
Neither instrumentation commit changes an offload primitive. Each run config
records the full core/backend SHA, backend tree, clean state, mode, and workload.

The native and mapped variants intentionally use the **same clean source
head**, rather than different Git commits. This is a stricter feature-off / 
feature-on control: it removes unrelated base/head source drift and changes
only `spec_name` (`NPUOffloadingSpec` versus `MappedOffloadingSpec`).

## Workload and protocol

- `Qwen/Qwen2.5-14B-Instruct`, float16, max model length 32,768.
- 8 GiB device KV cache and 16 GiB pinned CPU KV tier.
- 40 deterministic prefix-repetition requests at 1 request/s.
- 3,840 prefix + 256 suffix input tokens and 256 output tokens.
- Canonical request-set SHA-256:
  `43edb2018fdf3ab73d75fedbfe6197fbf04bfa1ea69866b682fadac6be9a0499`.
- Three fresh service lifecycles per mode in reverse-order pairs:
  native→mapped, mapped→native, native→mapped.
- One task-specific 24 GiB-SHM source container; every lifecycle starts a
  fresh API/engine process and checks that the selected NPU is idle before and
  after the run.

Transfer bandwidth and latency come from the handler's NPU timing events in
each completed `TransferResult`. The trace records every transfer's direction,
bytes, and device-event interval. Aggregate bandwidth is total bytes divided
by summed device-event time; p99 uses nearest-rank observed samples. Peak NPU
memory is the maximum per-process device-memory sample reported by `npu-smi`.

## Correctness and validation

All six retained lifecycles completed 40/40 requests with zero request
failures (240/240 total), used the byte-identical request-set file, observed
both transfer directions, and left no process on physical NPU 4.

The current-vLLM API port was validated before packaging:

- Ruff 0.14.0 check and format check: passed.
- Focused host suite: **33 passed**.
- One-card mapped handler/operator/lifecycle suite: **9 passed**.

The exact outputs and before/after NPU snapshots are in [`reproduction`](reproduction).
The custom operator and extension source did not change in the API-port commit;
the previously validated exact-source artifacts remained in use.

After the measured client completes, the harness explicitly terminates the
task's multiprocess server and verifies NPU cleanup. vLLM can log an
`EngineDeadError` in the API output handler during this forced post-client
teardown; it occurs after the saved 40/40 benchmark result and is not counted
as a request or transfer failure.

## Results

| Run | Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | H2D GB/s | H2D p99 (ms) | Peak NPU MB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | native | 74.06 | 5,818.00 | 20,563.50 | 113.35 | 143.93 | 9.51 | 78.84 | 4.20 | 121.66 | 38,382 |
| 2 | mapped | 75.27 | 6,180.62 | 20,948.92 | 114.10 | 147.48 | 10.25 | 75.12 | 36.27 | 8.72 | 38,422 |
| 3 | mapped | 74.23 | 6,112.86 | 20,857.57 | 112.06 | 144.92 | 7.88 | 70.72 | 38.49 | 8.64 | 38,422 |
| 4 | native | 76.02 | 6,053.60 | 20,768.59 | 113.86 | 145.75 | 8.13 | 96.47 | 3.27 | 177.98 | 38,378 |
| 5 | native | 75.16 | 6,067.69 | 21,145.12 | 115.18 | 147.04 | 7.96 | 97.38 | 2.83 | 205.79 | 38,382 |
| 6 | mapped | 67.82 | 4,377.67 | 17,342.14 | 100.08 | 126.35 | 7.99 | 66.90 | 20.72 | 27.47 | 38,418 |

Three-lifecycle aggregates:

| Mode | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | D2H GB/s | D2H p99 (ms) | H2D GB/s | H2D p99 (ms) | Peak NPU MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 75.08 | 5,979.76 | 20,825.74 | 114.13 | 145.57 | 8.48 | 96.47 | 3.34 | 205.79 | 38,381 |
| mapped | 72.44 | 5,557.05 | 19,716.21 | 108.75 | 139.59 | 8.58 | 70.72 | 29.23 | 27.47 | 38,421 |

Both modes completed exactly 123 D2H jobs totaling 28,764,536,832 bytes.
Mapped D2H measured 8.58 GB/s versus 8.48 GB/s native (+1.17%), neutral within
the run-to-run spread. This is the expected result for a PR that retains
asynchronous D2H copies. Mapped H2D measured 29.23 GB/s versus 3.34 GB/s
native, while pooled H2D p99 device-event latency fell from 205.79 ms to
27.47 ms. Mean peak NPU process memory changed by +0.10%.

The three-run means favor mapped for duration (−3.52%), mean TTFT (−7.07%),
p99 TTFT (−5.33%), mean TPOT (−4.72%), and p99 TPOT (−4.11%). The end-to-end
advantage is not stable across every reverse-order pair: pair 1 is slightly
worse, pair 2 is essentially neutral, and pair 3 supplies most of the mean
gain. The evidence therefore supports the narrow H2D component claim and D2H
parity, not a default-path end-to-end guarantee. The feature remains opt-in
experimental infrastructure.

Machine-readable per-run data, pooled statistics, standard deviations, and
pair deltas are in [`retained-summary.json`](retained-summary.json). The compact
human table is in [`retained-summary.md`](retained-summary.md).

## Replay and integrity

The harness and summarizer are under [`reproduction`](reproduction). With the
documented source container running, replay one lifecycle from the backend
repository root with:

```bash
benchmarks/evidence/pr153/pressure-20260802/reproduction/run_pr153_pressure_lifecycle.sh \
  native 1 /absolute/output/retained-01-native \
  source-dev-pr153-pressure-20260802
```

Select `mapped` for the feature-on variant. Each raw lifecycle directory in
this bundle includes the run config, detailed benchmark result, per-request
latencies, merged transfer events, Prometheus snapshots, resource samples,
server/client logs, and NPU before/after snapshots.

[`SHA256SUMS`](SHA256SUMS) covers every evidence file in this directory except
itself.
