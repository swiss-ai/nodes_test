# Focused NCCL results — Aug 24, 2026

## Summary

Some small two-node evaluations showed an approximately 25% slowdown. To
isolate it, the investigation moved from top-down debugging through the full
Megatron stack to minimal two-node communication benchmarks.

The custom P2P probe used the corrected launch: one rank per GPU on two
four-GPU nodes, PMIx, and `--network=disable_rdzv_get`. It measures all 28
unique GPU pairs with a 64 MiB ping-pong transfer and repeats the full matrix
for 10 sweeps. On nodes associated with slower evaluations, it caught
intermittent drops both within and across nodes:

- `nid006907`: about 30% lower intra-node bandwidth and 14–16% lower
  cross-node bandwidth in isolated sweeps.
- `nid007195`: about 37–42% lower intra-node bandwidth in isolated sweeps.

The failures usually appear in only one sample for an affected pair during a
10-sweep run. A clean rerun is therefore a quiet window, not enough to clear
the node.

Stock `nccl-tests`, also using the corrected launch, reproduced the same
intermittent behavior in the three-mode test (intra node A, intra node B, and
all eight GPUs across both nodes):

- With `nid006907`, one cross-node reduce-scatter pass dropped by about 64%
  and recovered on the next pass.
- With `nid007195`, one cross-node scatter pass dropped by 73–75% and then
  recovered.

Preliminary stock cross-pair results (runs omitted
`--network=disable_rdzv_get`): `nid007133+nid007195` had isolated lows of
14.52 GB/s at 64 MiB and 21.24 GB/s at 256 MiB. The same-anchor and independent
controls had minima of 22.03–22.25 GB/s at 64 MiB and 22.56–23.05 GB/s at
256 MiB. This is suggestive only; corrected reruns are pending.

Current verdict: the P2P and stock NCCL tests consistently support an
intermittent communication-path problem on `nid006907` and `nid007195`.
Because it is episodic, a single clean run does not overturn earlier failures.

## Logs to inspect

- Custom P2P: [nid006907 flagged samples](logs/p2psb-6907-6856-3173446.log#L176-L181)
  and [nid007195 flagged samples](logs/p2psb-7195-6856-3173448.log#L176-L178).
- Stock three-mode: [nid006907 reduce-scatter drop](logs/nccltsb-6907-6856-3173447.log#L418)
  and [nid007195 scatter drop](logs/nccltsb-7195-6856-3173449.log#L502-L503).
- Stock cross-pairs (preliminary; runs omitted
  `--network=disable_rdzv_get`): [nid007195 64 MiB low](attic/logs/nccltx-float-7195-3168415.log#L534),
  [256 MiB low](attic/logs/nccltx-float-7195-3168415.log#L1831),
  [same-anchor control](attic/logs/nccltx-7133-ctrl-3168822.log#L533-L535),
  and [independent control](attic/logs/nccltx-float-ctrl-3168416.log#L533-L535).

## Pending follow-ups

`3174200`, the path-trace smoke gate, timed out while Nsight generated its
report; jobs `3174259` and `3174351–3174353` remain pending behind its failed
`afterok` dependency.
`3174098` is the delayed clean-control intra-pair rerun. These attic diagnostics
are not included in the verdict above.
Corrected cross-pair reruns are queued as `3174640–3174642`.
