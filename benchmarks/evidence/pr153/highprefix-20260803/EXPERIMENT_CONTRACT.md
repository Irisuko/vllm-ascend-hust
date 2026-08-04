# PR 153 high-prefix mapped/native A/B contract

## Claim and decision

Claim under test: under the repository's high-prefix, severely HBM-constrained
workload, replacing native asynchronous H2D restore with mapped-host gather
improves end-to-end serving while preserving the native asynchronous D2H path.

The result decides whether PR 153 can add a scoped end-to-end claim for this
workload. It does not decide whether mapped restore should become the default.

## Fixed conditions

- Same clean PR 153 backend source and paired instrumented vLLM core.
- Qwen/Qwen2.5-7B-Instruct, float16, one Ascend 910B2 per lifecycle.
- 256 MiB device KV and 1 GiB CPU KV tier.
- Prefix caching enabled with SHA-256 hashing.
- 200 deterministic requests at 1 request/s: 3,840 prefix + 256 suffix input
  tokens, 256 output tokens, and 10 prefixes (95% nominal reuse opportunity).
- Graph mode, max model length 4,608, max 16 sequences.
- Native and mapped differ only by offloading spec name.
- Physical NPUs 1 and 2 run in an ABBA crossover: NPU 1 uses
  native/mapped/mapped/native and NPU 2 mapped/native/native/mapped.
- Fresh API/engine process for every lifecycle; identical request-set hash;
  selected NPU must be idle before and after every run.

## Evidence threshold

Primary metrics are duration, mean/p99 TTFT, and throughput-equivalent fixed
workload completion time. TPOT, D2H/H2D event bandwidth and latency, peak NPU
memory, correctness, per-card effects, and paired-round deltas are safeguards.

Evidence supports the claim only if the mapped advantage is larger than
run-to-run and card-to-card variation and does not come from changed D2H work,
request failures, or mismatched request sets. A mean improvement supplied by
one round, one card, or a changed transfer volume counts against a stable
end-to-end claim.

## Budget and stop conditions

Run four concurrent rounds (eight fresh lifecycles, four per mode) after source,
device, model, and request-set verification. Stop or repair the contract before
retaining results if either card becomes occupied, source parity changes, the
request set differs, the workload cannot complete, or one variant lacks either
transfer direction.

## Pre-observation amendment

The initially selected idle NPUs 3 and 4 had empty process tables but could not
initialize from a fresh task container (`torch.npu.device_count() == 0`), most
likely because established containers still held their device contexts. NPU 1
passed a fresh tensor round trip. The contract therefore uses NPU 1 plus the
next independently usable idle card; no measured lifecycle had started when
this resource-only amendment was made. After Fletcher authorized deletion of
the abandoned containers holding stale device contexts, NPU 2 passed the same
fresh tensor round trip and became the second crossover card. NPU 1's first
native lifecycle had completed before the exact second-card ID was recorded;
the mode, workload, source, and crossover schedule were unchanged.

After the first complete NPU 1/NPU 2 pair, NPU 1 retained a stale driver-side
process record even though the API/engine process and container were gone. To
avoid treating an unhealthy card lifecycle as retained evidence, that pair and
the earlier NPU 1 setup run are labeled as pilots and excluded. With the
abandoned context-owning containers removed, NPUs 2 and 3 both passed fresh
tensor round trips, but an unrelated multi-card workload occupied NPUs 0--5
before a retained lifecycle could start. After that workload ended, NPUs 1 and
2 again had empty process tables and idle-baseline memory. The retained
experiment therefore restarts from lifecycle index 1 on the original NPU 1/2
pair with the original four-round ABBA schedule. Fresh containers were created
without an extra NPU preflight process, and this amendment was recorded before
starting or aggregating any retained result.
