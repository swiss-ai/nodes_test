# Clariden node health summary — GPU p2p bandwidth checks, 2026-08-21

Method: NCCL ping-pong over all 28 GPU pairs of a 2-node allocation (4 GPU
each), inside the production eval container (NCCL 2.30.7, libfabric/ofi).
Healthy baselines, 64 MiB one-way bandwidth: intra-node NVLink ~114 GB/s,
cross-node ~23.8 GB/s (Slingshot line rate). Tools:
`nccl_p2p_matrix.sh` (single sweep, ~35 s) and `nccl_p2p_matrix_rep.sh`
(10 sweeps in one job, mean/std/min; ~70 s) in
`probes/` (this repo). Control noise floor (10-sweep pair
nid006856+006857): std <= 0.5, min >= 113.4 intra / 22.3 cross.

## Verdicts

| Node      | Runs | Verdict  | Evidence |
|-----------|------|----------|----------|
| nid007195 | 4    | **SICK** | severe NVLink cell in 3/3 afternoon runs (69 / 48 / 78 GB/s, always involving GPU0, pair migrates); depressed ingress 15.7-22.6; quiet in the 21:02 10-sweep run (episodic) |
| nid007162 | 2    | **SICK** | NVLink g1-g2 45.2; ingress 13.7-19.7; 10-sweep run: two cross collapses (16.6, 20.5) + g1-g2 wobble min 95.9 std 5.9 |
| nid007434 | 3    | **SICK** | run 1 mild (cross 19.5-21.2); run 2 severe: NVLink g0-g2 44.8 AND g1-g3 70.5, 10/16 cross cells 15.6-21.6; 10-sweep: known cells (GPU2) wobble min ~100.5 std ~4.5 |
| nid007133 | 3    | clean    | flat 23.8 cross block vs healthy partner; NVLink clean in all suspect-pair runs |
| nid007416 | 1    | clean    | flat |
| nid007418 | 2    | clean    | one-off 88.8 intra cell vanished on rerun (artifact band) |
| nid007420 | 2    | clean    | one-off 87.0 intra cell vanished on rerun (artifact band) |
| nid007425 | 1    | clean    | flat |
| nid007429 | 1    | clean    | flat |
| nid007432 | 1    | clean    | flat |
| nid007434 |      | (see SICK above) | |
| nid007437 | 1    | clean    | flat |
| nid006856 | many | clean (reference) | flat in every run; carries the largest NVLink error count (95 replays) while performing perfectly |
| nid006857 | many | clean (reference) | flat in every run |

## Run-by-run evidence, sick nodes (all 2026-08-21)

| Time  | Job     | Pair                  | Worst intra (GB/s)   | Worst cross (GB/s) | Note |
|-------|---------|-----------------------|----------------------|--------------------|------|
| 18:42 | 3141978 | 007133 + **007195**   | 69.1 (7195 g0-g2)    | 17.4               | first conviction |
| 18:45 | 3142040 | 007133 + **007195**   | 47.8 (7195 g0-g1)    | 15.7               | defect MOVED, still GPU0 |
| 18:47 | 3142051 | 006857 + **007195**   | 78.1 (7195 g0-g2)    | 18.6               | isolation: follows 7195 |
| 19:07 | 3142136 | 006857 + **007162**   | 45.2 (7162 g1-g2)    | 13.7               | conviction |
| 20:39 | 3142445 | 006856 + **007434**   | clean                | 19.5               | mild round |
| 20:42 | 3142456 | 006856 + **007434**   | 44.8 (g0-g2), 70.5 (g1-g3) | 15.6         | severe; 10/16 cross cells low |
| 21:02 | 3143136 | 006857 + **007195**   | min 106.8 (10-sweep) | clean              | quiet episode |
| 21:03 | 3143137 | 006857 + **007162**   | min 95.9 g1-g2       | 16.6, 20.5 (1-sweep collapses) | still failing |
| 21:02 | 3143138 | 006856 + **007434**   | min ~100.5 (GPU2 cells) | clean           | quiet, fingerprint visible |

Controls: 3141979 (18:42) and 3143135 (21:01) on nid006856+006857 — flat.
Clean-node sweep: jobs 3142437-3142446 + reruns 3142454/55 (20:37-20:42).

## Key characteristics

1. **Episodic**: severe events cluster in bursts (all three 7195 catches
   within 5 min; quiet 2.5 h later). A single clean health check does NOT
   clear a node.
2. **Location-stable**: the degraded cells return at the same GPU pairs
   across hours (7162 g1-g2, 7434 GPU2 cells, 7195 always GPU0).
3. **Both fabrics at once**: every sick node degrades intra-node NVLink AND
   cross-node ingress together; clean nodes show neither.
4. **First-line diagnostics are clean** on all three: all 72 NVLink links at
   full 26.56 GB/s, no meaningful CRC/replay accumulation (the healthy
   reference has more link errors than the sick nodes), zero ECC/remapped
   rows, no throttle flags. Root cause is below/above what user space sees —
   needs out-of-band diagnostics (BMC/HMC, dcgmi diag -r 3/4, fabric-side
   counters).
5. **Production impact measured**: with nid007195 in a 2-node vLLM eval,
   wall time +25% (404 s -> 506/508 s, reproduced 3x) and ~15% more energy
   per eval.

## Collective + burst experiments (Aug 21 late night)

Production-shaped probes (jobs `nccl-collrep-31438xx`, `nccl-burst-31439xx`;
scripts `nccl_collective_rep.sh`, `nccl_burst_rep.sh`): all-reduce /
all-gather / reduce-scatter on 3 groups (each node's 4 GPUs intra + all 8),
64 MiB, 20 barriered samples resp. 10 pipelined 10-op bursts (barrier
outside the loop; different process groups = different NCCL streams, so the
"simultaneous" phase overlaps both intra meshes for a contention test).

Healthy baselines (control pair): intra AR ~290-300 busbw, intra AG/RS
~250-260, all-node AR ~95-108 (adjacent pair higher; AR is
placement-sensitive, AG/RS are NOT — which is why the eval, which runs
AG/RS, measured placement-neutral), all-node AG/RS ~84-85. Pipelined bursts
repeat within ~0.5%; simultaneity costs nothing.

Sick trio during the same active episode: individual collectives collapse
to 23-46 GB/s all-node and 28-58 intra (2-4x slow); all-node AR bimodal on
7195 (half the samples ~60 vs ~93); one slow op stalls a whole pipelined
burst (7434 all-node RS burst 8.3 -> 20.6 ms; 7162 intra AG 3.1 -> 13.2 ms);
collapses occur even in the intra-only simultaneous phase -> fault is
node-internal. Clean nodes: tight, with rare one-off dips (007418, 007420
single-sample AG dips) that do not reproduce.

## Raw NCCL probe — no PyTorch/Python/MPI (Aug 22, ~03:05)

`nccl_raw_collrep.cu` + `nccl_raw_rep.sh` (`probes/`): plain C harness,
one process per GPU via srun, file-based ncclUniqueId rendezvous, compiled
in-container against the SAME pip libnccl 2.30.7 the vLLM benchmarks load,
same untouched vllm-openai image + aws-ofi hook.  30 barriered 64 MiB
samples, world (8-rank) + per-node intra groups.  Job 3146535 on
006857+**007195**: steady-state world allgather is a dead-flat 91.2 GB/s
busbw, but sweep 16 collapses ALL THREE world collectives (AG 60.7 / AR
54.0 / RS 83.8) while 7195's OWN intra reduce-scatter drops to 196 in the
same sweep and 006857's intra stays clean (std 2.1) -> event localizes
inside 7195; further hits at sweeps 17 (74/78), 21 (AR 47.0, RS 60.4), 30
(AG 64.6).  ~4/30 samples affected within one minute.  Caveat: sample 1 of
each group is communicator-warmup (world AG reads 1.41, intra AG ~231 on
BOTH nodes) - discard it.

Clean-pair control (job 3146564, nid006856+nid007133, disjoint nodes,
same 30 sweeps): ZERO world samples below 85 GB/s; world reduce-scatter
std 0.04 (dead flat), allreduce min 87.  Side by side (warmup excluded):

| world collective | sick pair (with 007195)      | clean pair            |
|------------------|------------------------------|-----------------------|
| allgather        | dips 60.7 / 74.4 / 64.6      | none (all ~91.2)      |
| allreduce        | dips 54.0 / 78.2 / 47.0      | min 87.1              |
| reducescatter    | dips 83.8 / 60.4             | std 0.04              |

(One isolated intra-AG sample at 83.2 on 006856, sweep 3: no recurrence,
no world-side effect that sweep - consistent with the known one-off
artifact band, watch-list only.)  This removes PyTorch/Python/MPI from
the evidence chain entirely; the NGC 26.01 image also ships prebuilt
nccl-tests (/usr/local/bin/all_gather_perf, OpenMPI 5.0.9) for an
admin-blessed second opinion.

## Standard-stack reproduction (Aug 21-22 night)

Independent confirmation with zero custom software: untouched upstream
`vllm/vllm-openai:v0.27.1` image (pinned digest, no fork overlay), stock
`lm_eval 0.4.12` from PyPI, public `Qwen/Qwen3-30B-A3B` (MoE, 128 experts),
2 nodes, TP=8 + `enable_expert_parallel` (EP=8 expert dispatch =
allgather/reduce-scatter over all 8 ranks) via a Ray cluster, plain NCCL
2.30.7 for every collective, weights+datasets fully prefetched (HF-offline
timed runs).  Suite: piqa, arc_challenge, arc_easy, hellaswag, winogrande,
xcopa, xnli, xwinograd, pawsx, mmlu (0-shot).  Scripts:
`qwen3-bench/qwen3_standard_{setup,eval}.sbatch` + `vllm-standard-v0271.toml`.

| Job     | Pair                  | lm-eval wall | loglik phase (276682 reqs) | vs clean |
|---------|-----------------------|--------------|----------------------------|----------|
| 3144734 | **007195** + 006857   | 1126 s       | 636 s (434.4 it/s)         | +13.9% / +12.2% |
| 3144735 | 006856 + 006857       |  989 s       | 567 s (487.6 it/s)         | baseline |
| 3144736 | **007195** + 006857   | 1129 s       | 642 s (430.4 it/s)         | +14.2% / +13.2% |

The "loglik phase" column is the strongest single comparison: the
`Running loglikelihood requests` progress bar covers pure inference only
(no engine startup, tokenization, or scoring), over an identical request
count - swapping nid006856 for nid007195 costs ~55 it/s, reproducibly.

Jobs serialized on the shared partner 006857, so the only variable is
007195 vs 006856.  The two sick runs agree to 0.3%.  Accuracies are
identical across all three runs (<=0.4 pt jitter) - the fault costs time,
not correctness.  This removes "your custom stack" as an explanation:
a vanilla vLLM + lm-eval user loses ~14% wall time with nid007195 in the
allocation (vs +25% for the Apertus2 DP=8/EP=8 eval - consistent with the
episodic severity and the different workload mix).

### Production-shaped serve benchmark (Aug 22, 4 nodes)

Same standard image/venv, pure vLLM tooling: `vllm serve` with TP=4
(NVLink-local) x DP=4 (one replica per node) x EP=16 via
`--data-parallel-backend ray`, benched with `vllm bench serve` (seeded
random workload, 1024 in / 512 out, 3000 prompts, concurrency 512, 2
passes/job).  Script: `qwen3-bench/qwen3_standard_bench.sbatch`.  Jobs share
nid006856,006857,007432; the 4th node is the variable.

| Metric (avg of 2 passes)  | 3145399 (**007195**) | 3145400 (007437) | delta |
|---------------------------|----------------------|------------------|-------|
| Output throughput (tok/s) | 7314                 | 7656             | -4.5% |
| Mean TPOT (ms)            | 66.3                 | 63.3             | +4.7% |
| P99 TPOT (ms)             | 69.6                 | 66.2             | +5.1% |
| P99 ITL (ms)              | ~291                 | ~288             | same  |

Passes agree within 1% inside each job.  The identical P99 ITL tail shows
the config-inherent decode interruptions are NOT node-related; the sick
signature is a uniform ~3 ms inflation of EVERY decode step - the shape
slower EP dispatch produces.

Second layout, same serve workload (Aug 22, 2 nodes): TP=1 x DP=8 x EP=8 -
the original Apertus2 eval geometry, ALL comm is 8-way cross-node expert
dispatch, sick node owns 4/8 EP ranks.  Jobs share nid006857; variable node
in the non-head role both times.

| Metric (avg of 2 passes)  | 3145755 (**007195**) | 3145756 (007437) | delta |
|---------------------------|----------------------|------------------|-------|
| Output throughput (tok/s) | 11856                | 12676            | -6.5% |
| Mean TPOT (ms)            | 38.2                 | 35.6             | +7.3% |
| P99 TPOT (ms)             | 41.9                 | 39.2             | +6.9% |

The sick job is also NOISIER: its passes disagree by 3.3% tok/s / 6.4%
TPOT (clean: 1.6% / 0.6%) and sick pass 1 caught a P99 ITL of 314 ms vs
~110-146 ms everywhere else - episodic stalls land inside passes.

Impact ladder (one node swapped, standard tools): serve TP4xDP4xEP16 +4.7%
< serve TP1xDP8xEP8 +7.3% < TP=8 eval +13% < original DP=8/EP=8 eval +25%
(louder episode) - impact scales with the sick node's share of cross-node
EP traffic; a conservative layout shrinks but cannot eliminate the damage.
(Aside: DP=8/TP=1 is also ~3.3x more per-GPU throughput than TP=4 x DP=4
for this model - 12.7k tok/s on 8 GPUs vs 7.7k on 16.)

Prefill-only variant (same TP=1 x DP=8 layout, in=4096 out=1, 10k prompts,
jobs 3146087 sick 01:30-01:53 / 3146088 clean 01:53-02:15): NULL - 122.1k
vs 121.9k total tok/s, passes agree to 0.01%.  A p2p probe right after
(job 3146348, ~02:45) found 007195 near-quiet: its g1-g2 NVLink cell
wobbling mildly (min 89.6, std 8.0) + three one-off cross dips ~19.5, no
severe cells.  Read: the run landed in a QUIET phase (the pristine pass
agreement is the tell - the loud decode bench 40 min earlier had 3-6% pass
scatter).  Confirms intermittency: a node can benchmark perfectly clean
for 23 straight minutes and still be sick; impact numbers are only
interpretable next to a concurrent loudness probe.

Positive control (job 3146391): same prefill bench, same CLEAN pair, NCCL
forced to one channel (NCCL_MAX_NCHANNELS=1, verified in NCCL_DEBUG=INFO
output) -> 121.9k drops to 42.8k tok/s (2.85x).  The bench is violently
wire-sensitive (comm-dominated), so the null is NOT instrument blindness:
007195's fabric genuinely ran at full speed in that window.  Corollary: a
prefill-4k serve bench during a LOUD episode should show a very large
delta.

Back-to-back probe pair on 006857+007195 (~02:17-02:20, jobs 3146348 p2p /
3146356 collective) settles which instrument to trust: p2p said NEAR-QUIET
(mild g1-g2 wobble min 89.6, three one-off cross dips ~19.5) while the
collective test, minutes later, caught the fault ACTIVE - intra-7195
reducescatter down to 37.3 GB/s busbw (6.6x slow), allreduce to 77.9, and
the all-node 8-rank ALLGATHER (= vLLM's EP dispatch op) to 15.7 / 42.7 in
2/20 samples vs a steady 82-83 otherwise; partner 006857 pristine (std
1.8-2.1).  Also a between-collapses fingerprint: 7195's intra allgather is
BIMODAL (healthy ~246 band + ~214 band, half the samples ~13% low, std
30.8) where 006857 is unimodal 242-251.  Runbook consequence: the 20-sweep
COLLECTIVE test is the sensitive detector (drives all links at once); p2p
ping-pong only localizes cells and can read near-clean while collectives
are collapsing.

Repro friction worth noting in a ticket: anonymous HF API calls from the
compute-node NAT IP are rate limited (needs HF_TOKEN or retries), and
multi-node TP in stock vLLM v0.27.1 requires `VLLM_ALLREDUCE_USE_SYMM_MEM=0`
plus `disable_custom_all_reduce=True` - both intra-node-only optimizations
whose capability probes deadlock across nodes on GH200 (py-spy stacks in
jobs 3144552, 3144711).

## Mitigation

`--exclude=nid007195,nid007162,nid007434` on timing-sensitive jobs.
Checker for any new suspect: `NCCL_BENCH_NODELIST=nidX,nid006856
probes/nccl_collective_rep.sh` FIRST (20-sweep collectives, the sensitive
detector - convict on collapse samples, high std, or a bimodal intra
band vs the partner), then `probes/nccl_p2p_matrix_rep.sh` (~70 s) to localize
which GPU pairs carry the defect.  A clean p2p run does NOT clear a node
(02:17 vs 02:20 pair above).

Logs: `nccl-p2pmatrix-{3141978,3141979,3142040,3142050,3142051,3142136,
3142437..3142446,3142454..3142456}.log`, `nccl-p2prep-{3143135..3143138}.log`
(rep-series now in `logs/` here; the older single-sweep p2pmatrix logs are no longer on disk). Full narrative: `apertus2_scaling_ladder_evals_vllm/TIMING_PLACEMENT_STUDY.md`.
