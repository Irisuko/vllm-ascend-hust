# PR 153 high-prefix crossover evidence (2026-08-03)

## Decision

This matched two-card ABBA experiment supports a **scoped end-to-end claim**
for PR 153's mapped H2D restore under the repository's high-prefix, severely
HBM-constrained workload. It does not establish that mapped restore should be
the default for other workloads.

Across four fresh lifecycles per mode, mapped reduced fixed-workload duration
from 868.15 s to 851.81 s (-1.88%), increased request and token throughput by
1.92%, reduced mean TTFT by 2.33%, and reduced p99 TTFT by 2.18%. The direction
was consistent in all four reverse-order pairs and on both physical cards:

- pair duration deltas (mapped minus native): -27.47, -18.52, -8.74, and
  -10.65 s;
- NPU 1 duration: -2.06%, mean TTFT: -2.77%; and
- NPU 2 duration: -1.70%, mean TTFT: -1.89%.

The mean duration separation (16.34 s) exceeds both the mapped run standard
deviation (2.42 s) and native run standard deviation (6.15 s). TPOT is mixed:
mean TPOT improved 1.47%, while mean-of-run p99 TPOT regressed 0.52%. The claim
is therefore specifically about fixed-workload completion and TTFT, not every
latency statistic.

## Directional transfer result

The production contract remains directional: D2H stores use native
asynchronous copy in both modes, while only mapped-mode H2D restore uses
mapped-host gather.

Mapped H2D processed slightly more data (+0.90%, 57.50 versus 56.99 GB) yet
raised pooled aggregate H2D bandwidth from 0.99 to 63.85 GB/s and reduced
pooled p99 device-event latency from 689.06 to 7.36 ms. Mapped D2H processed
0.15% less data (126.18 versus 126.37 GB); aggregate D2H bandwidth varied from
7.08 to 6.35 GB/s. The D2H job counts and volumes are close but not identical
because scheduling outcomes differ, so this experiment does not claim exact
D2H event parity or attribute its D2H bandwidth spread to the PR. The mapped
end-to-end advantage did not come from faster D2H: its pooled D2H device-event
time was about 2.04 s longer despite the slightly smaller volume.

## Source and workload

- Clean `vllm-ascend-hust` commit
  `d48784c2e0803df4f4eb6212f49d11f219aa13c0`, tree
  `372508902acfb5ed02c1bdd143fac98a36543c11`.
- Clean vLLM-HUST core commit
  `6901c26b23fa60ca1a2450b0999f803f6ee6c335`, tree
  `829202c4f054e11527dd6867878c76e36859a1bb`. Its two extra commits provide
  opt-in transfer tracing and deterministic benchmark request IDs only.
- `Qwen/Qwen2.5-7B-Instruct`, float16, graph mode, prefix caching with SHA-256.
- 256 MiB device KV and 1 GiB pinned CPU KV tier.
- 200 deterministic requests at 1 request/s: 3,840 shared-prefix + 256 suffix
  input tokens, 256 output tokens, and 10 prefixes (95% nominal reuse
  opportunity).
- Canonical request-set SHA-256:
  `c0498f87ca1c221fa82f5811619ff1001d7921ddf9d907735746eabe3c4d9a82`;
  byte-level request-set file SHA-256:
  `607912ba4ad105124d220fd278df81e87f5b9e1d5b3cca592b863cb7d1dedc23`.
- Ascend 910B2 physical NPUs 1 and 2, each with two native and two mapped
  lifecycles in an ABBA crossover.
- Runtime image
  `quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`, image ID
  `sha256:0fc116f43369c0dd71bc253dff83d7702dc44747d10ca1c8e9569e7265e10731`.

Native and mapped use the same clean source and differ only in offloading spec:
`NPUOffloadingSpec` versus `MappedOffloadingSpec`. Each lifecycle starts a
fresh API/engine process. All eight retained lifecycles completed 200/200
requests with zero failures (1,600/1,600 total), observed both transfer
directions, used the identical request-set file, and left no NPU process after
shutdown.

The experiment contract, including pre-observation device-availability
amendments, is in [`EXPERIMENT_CONTRACT.md`](EXPERIMENT_CONTRACT.md). Pilot and
failed-preflight attempts are excluded from this retained evidence bundle.

## Results

| Mode | Duration (s) | Req/s | Tok/s | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | H2D GB/s | H2D p99 (ms) | Peak NPU MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| native | 868.15 | 0.230 | 1002.63 | 308,940.56 | 615,587.66 | 108.77 | 1,775.07 | 0.99 | 689.06 | 16,066 |
| mapped | 851.81 | 0.235 | 1021.84 | 301,748.83 | 602,145.28 | 107.17 | 1,784.26 | 63.85 | 7.36 | 16,026 |

Machine-readable per-run data, pooled transfer statistics, standard
deviations, reverse-order pair deltas, and per-device crossovers are in
[`retained-summary.json`](retained-summary.json). The compact complete table is
in [`retained-summary.md`](retained-summary.md).

## Replay and integrity

The harness, summarizer, transfer-event conversion tools, and exact core
instrumentation patch are under [`reproduction`](reproduction). With the
documented task-specific source container running, replay one lifecycle from
the backend repository root with:

```bash
PHYSICAL_DEVICE_ID=1 \
benchmarks/evidence/pr153/highprefix-20260803/reproduction/run_pr153_highprefix_lifecycle.sh \
  native 1 /absolute/output/retained-01-npu1-native \
  source-dev-pr153-highprefix-npu1
```

Select `mapped` for the feature-on variant. Every retained directory includes
the exact run config, raw detailed benchmark result, per-request latencies,
merged transfer events, Prometheus snapshots, resource samples, server/client
logs, and before/after NPU snapshots.

[`SHA256SUMS`](SHA256SUMS) covers every evidence file in this directory except
itself.
