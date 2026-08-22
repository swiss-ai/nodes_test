# node-health — Clariden sick-node investigation (Aug 21–22, 2026)

Self-contained archive of the cluster-health study that convicted three
nodes of degrading cross-node NCCL collectives, plus everything needed to
re-run any measurement. The full evidence narrative (ticket-facing) is
[NODE_HEALTH_SUMMARY.md](NODE_HEALTH_SUMMARY.md); this README is the map.

## Verdict (TLDR)

- **Sick nodes: nid007195, nid007162, nid007434** — flaky NVLink cells
  (45–78 GB/s vs ~114 clean) plus sub-line-rate ingress; the fault is
  **episodic** (loud/quiet windows, minutes apart). Mitigation:
  `--exclude=nid007195,nid007162,nid007434` on timing-sensitive jobs.
- Impact scales with the sick node's share of cross-node EP traffic
  (one node swapped, everything else identical):

  | workload (2–4 nodes, Qwen3-30B-A3B)      | penalty        |
  |------------------------------------------|----------------|
  | vllm serve TP=4 x DP=4 x EP=16           | +4.7% TPOT     |
  | vllm serve TP=1 x DP=8 x EP=8            | +7.3% TPOT     |
  | lm-eval TP=8 + EP=8                      | +13–14% wall   |
  | Apertus2 production DP=8/EP=8 (episode)  | ~+25%          |

- **Detector hierarchy**: the 20-sweep *collective* probe convicts
  (collapse samples, high std, bimodal intra band); the p2p matrix only
  *localizes* bad GPU pairs — a clean p2p run does NOT clear a node
  (proved back-to-back at 02:17 vs 02:20).
- The raw C probe (`probes/nccl_raw_collrep.cu`, no PyTorch/Python/MPI,
  same libnccl 2.30.7) reproduced the collapses on the sick pair while a
  disjoint clean pair stayed dead flat — the fault lives below every
  software stack we run.
- Instrument sanity: `NCCL_MAX_NCHANNELS=1` positive control cut the
  prefill bench 2.85x, so a null on a sick pair means a genuine quiet
  window, not benchmark blindness.

## Layout

```
NODE_HEALTH_SUMMARY.md   full evidence write-up (ticket-facing)
probes/                  NCCL-level probes (all self-submitting sbatch)
  nccl_collective_rep.sh   20-sweep AG/AR/RS, world+intra  <- convicting detector
  nccl_p2p_matrix.sh       single-sweep p2p bandwidth matrix
  nccl_p2p_matrix_rep.sh   10-sweep p2p matrix (mean/std/min)
  nccl_burst_rep.sh        bursty small-collective variant
  nccl_allreduce_bench.sh  first-generation allreduce probe
  nccl_raw_rep.sh          raw C probe wrapper (compiles in-container)
  nccl_raw_collrep.cu      the C harness: no torch/python/MPI, file-based
                           ncclUniqueId rendezvous, ncclCommSplit intra groups
qwen3-bench/             standard-stack workload repro (stock everything)
  import_vllm_standard.sbatch   one-time: import vllm/vllm-openai:v0.27.1 sqsh
  vllm-standard-v0271.toml      EDF (aws-ofi hook, symm-mem off, HF_HOME)
  qwen3_standard_setup.sbatch   one-time: stock lm-eval venv + dataset prefetch
  qwen3_standard_eval.sbatch    timed lm-eval, TP=8 + EP over 2 nodes
  qwen3_standard_bench.sbatch   vllm serve + vllm bench serve (TP/DP/len knobs)
vllm-bench/              Apertus2 22B standard vLLM benchmarks (own README):
                         `vllm bench throughput` + serve/bench-serve on
                         ShareGPT, TP=4+EP single node, eval-overlay image
                         d094597b; dataset fetched by download_sharegpt.sh
results/                 per-job outputs (bench_pass*.json, serve.log, evals)
logs/                    all Slurm job logs (nccl-* probes, qwen3-std-* runs)
```

## Running the probes

All probes self-submit (reservation `SD-69241-apertus-1-5-0`, account
`infra01`; override with `NCCL_BENCH_RESERVATION=...`):

```
# convict / clear a suspect (pair it with a known-clean node):
NCCL_BENCH_NODELIST=nidSUSPECT,nid006856 probes/nccl_collective_rep.sh

# localize which GPU pairs carry the defect:
NCCL_BENCH_NODELIST=nidSUSPECT,nid006856 probes/nccl_p2p_matrix_rep.sh

# stack-free confirmation (raw C, same libnccl as vLLM):
NCCL_BENCH_NODELIST=nidSUSPECT,nid006856 probes/nccl_raw_rep.sh
# knobs: NCCL_RAW_SWEEPS=30 NCCL_RAW_SIZE_MB=64
```

Per the A/B rule: every sick-pair run needs its disjoint clean-pair
control before drawing conclusions. Raw-probe trap: sample 1 of each
group is communicator warmup (world AG can read ~1.4 GB/s) — discard it.

## Running the workload benches

One-time setup: `sbatch qwen3-bench/import_vllm_standard.sbatch`, then
`sbatch qwen3-bench/qwen3_standard_setup.sbatch` (venv + HF prefetch;
after that the timed jobs run offline). Then:

```
# lm-eval leg (2 nodes, TP=8 + EP):
LABEL=sick-7195 sbatch --nodelist=nidA,nidB qwen3-bench/qwen3_standard_eval.sbatch

# serve+bench leg (default 4 nodes, TP=4 x DP=4 x EP=16):
LABEL=sick-7195 sbatch --nodelist=nidA,nidB,nidC,nidD qwen3-bench/qwen3_standard_bench.sbatch

# original Apertus2 layout (TP=1 x DP=8 x EP=8, all comm cross-node):
TP_SIZE=1 DP_SIZE=8 LABEL=dp8-sick sbatch --nodes=2 --nodelist=nidA,nidB \
    qwen3-bench/qwen3_standard_bench.sbatch

# prefill-only shape (lm-eval loglikelihood-like):
IN_LEN=4096 OUT_LEN=1 NUM_PROMPTS=10000 LABEL=prefill4k-sick sbatch ... 
```

`NUM_PROMPTS=300` (bench) / `LIMIT=32` (eval) give a prewarm/smoke run
that also warms the vLLM compile cache. Results land in
`results/$LABEL-$JOBID/`.

## Lives outside this repo (on purpose — heavy or shared)

- `../vllm-play/hf_home_std/` — model weights + datasets (HF_HOME in the EDF)
- `../vllm-play/lmeval-std-venv/` — stock lm-eval venv
- `../vllm-play/vllm_cache_std/` — vLLM compile cache
- `../vllm-play/vllm-v0260-eval-multinode.toml` — default EDF for the
  torch-based probes (NCCL_BENCH_ENV overrides)
- `../.edf_imagestore/vllm+vllm-openai+v0.27.1.aarch64.sqsh` — the image
- `../check_nccl_transport.sh` — verifies NCCL picked AWS Libfabric/GDRDMA
  (not TCP) in any container

## Watch list (not convicted)

- nid007133: one-off p2p cell, later exonerated by the 8-node sweep
- nid006856: single intra-AG 83.2 GB/s at raw-probe sweep 3 (job 3146564),
  no recurrence, no world-group effect
