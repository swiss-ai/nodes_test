# vLLM benchmarks for the 22B MoE (iter_0036689)

Minimal, self-contained benchmark setup for the HF export at
`hf-export/scaling-ladder/22b-moe-256e-latent-swa15-nope-.../iter_0036689`
(Apertus2ForCausalLM, 22.4B params bf16, 256 routed experts top-8, SWA 1:5,
NoPE at the 4 full-attention layers, max context 8192).

Uses the **standard vLLM benchmark CLI** (`vllm bench ...`) — nothing custom —
so every number is comparable to what people publish for other models.

## Quick start

```
cd /iopsstor/scratch/cscs/mvasilev/node-health/vllm-bench
./download_sharegpt.sh              # once, ~640 MB, login node is fine
sbatch offline_throughput.sbatch    # offline: max batch throughput
sbatch online_serving.sbatch        # online: latency under load
```

Progress: `squeue --me`, logs in `slurm/`, numbers in `results/` and at the
bottom of the slurm log.

## Files

```
vllm-bench.toml            container env (EDF): the baked v0.27.1+apertus2 image
download_sharegpt.sh       one-time dataset download
offline_throughput.sbatch  job 1: `vllm bench throughput` (engine in-process)
online_serving.sbatch      job 2: submits online_inner.sh into the container
online_inner.sh            job 2 body: vllm serve + wait + vllm bench serve
slurm/                     job logs (+ server-<jobid>.log for the online job)
results/                   JSON results from both benchmarks
```

## Offline vs online — what's the difference and why both

```
OFFLINE  (vllm bench throughput)          ONLINE  (vllm serve + vllm bench serve)

  benchmark process                         client process        server process
  +---------------------+                   +--------------+      +--------------+
  |  LLM engine         |                   | open 500     | HTTP | OpenAI API   |
  |  <- 500 prompts     |                   | requests on  |----->| /v1/         |
  |  all queued at t=0  |                   | a schedule   |<-----| completions  |
  |  no HTTP, no        |                   | (all at once |stream|  LLM engine  |
  |  streaming          |                   | or X req/s)  |      +--------------+
  +---------------------+                   +--------------+
  measures: tokens/second                   measures: TTFT, TPOT, ITL, E2E
  (one number: peak batch                   percentiles + achieved throughput
  throughput)                               (what a user would feel)
```

- **Offline** answers: *"how fast can this model chew through a pile of
  work?"* The engine is driven directly in Python, every prompt is available
  from t=0, the scheduler always has a full queue, and there is no HTTP /
  streaming / detokenization-to-network overhead. This is the number that
  matters for batch workloads (evals, synthetic data generation,
  distillation) and it is the *upper bound* on throughput. It produces **no
  latency numbers** — nobody is waiting.

- **Online** answers: *"what does serving this model feel like?"* A real
  `vllm serve` process exposes the OpenAI API; the benchmark is a plain HTTP
  client. Requests arrive on a schedule (`--request-rate`), responses are
  streamed, and per-request timing is recorded. This is the number that
  matters for interactive use. It is always somewhat below the offline
  number (API server overhead, request arrival gaps, streaming), and the
  interesting experiment is *sweeping the request rate* to find where
  latency blows up — the capacity "knee" of the deployment.

Rule of thumb: offline = capacity planning for batch jobs; online = SLA
planning for a service. With `--request-rate inf` (our default) the online
run degenerates into "offline + HTTP overhead", which is exactly why
comparing the two is instructive.

## The dataset: ShareGPT

`ShareGPT_V3_unfiltered_cleaned_split.json` is *the* canonical vLLM benchmark
dataset — real user/assistant conversations, used since the original vLLM
paper and the default in its benchmark suite. What the loader does:

1. Reads the JSON, keeps conversations with >= 2 turns.
2. prompt = first human turn, expected output length = token length of the
   real assistant reply (so output lengths follow a realistic distribution
   instead of a fixed number).
3. Filters to prompt <= 1024 tokens and prompt+output <= 2048 tokens,
   shuffles with a fixed seed, takes `--num-prompts`.

So this is a **short-context chat workload** (fits easily in our 8192
context). For long-context or controlled-length experiments you'd switch to
`--dataset-name random --random-input-len ... --random-output-len ...` —
same scripts, different flags.

## What actually happens, step by step

### Offline job (`offline_throughput.sbatch`)

1. SLURM allocates 1 GH200 node (4 GPUs), `srun --environment=vllm-bench.toml`
   starts one task inside the container image
   (`vllm-apertus2-eval+acb3dfe5+lm0788350b` = upstream vLLM v0.27.1 arm64 +
   the 6-file Apertus2 overlay from swiss-ai/vllm@acb3dfe5d).
2. `vllm bench throughput` builds the engine (TP=4 — see the topology
   section for why the offline job cannot use DP): loads the 42 GB of bf16
   weights (sharded 4-way), runs torch.compile + CUDA graph capture (first
   run: ~10-20 min, cached in `vllm-cache/` for next time), allocates the
   rest of GPU memory as paged KV cache.
3. All 500 prompts are enqueued at once. The engine runs **continuous
   batching**: every ~iteration it schedules whichever sequences fit,
   prefills new ones (chunked), decodes running ones, retires finished ones.
4. Prints `Throughput: X requests/s, Y total tokens/s, Z output tokens/s`
   and writes the same to `results/offline-*.json`.

### Online job (`online_serving.sbatch` -> `online_inner.sh`)

1. Same allocation + container. Inside, `vllm serve` starts in the
   background (same engine build as above) and exposes the OpenAI API on
   `127.0.0.1:8000`. Its output goes to `slurm/server-<jobid>.log`.
2. The script polls `/health` every 5 s until the server is ready (or dies —
   in which case it prints the server log tail and fails loudly).
3. `vllm bench serve` loads the same ShareGPT sample, then opens 500 HTTP
   requests against `/v1/completions` per the arrival schedule
   (`--request-rate inf` = all at t=0) and records per-token streaming
   timestamps for each request.
4. Prints the metric table, saves the full result JSON to `results/`, and
   shuts the server down.

### Model-parallel layout

The online job defaults to the same **TP/PP/DP/EP = 1/1/4/4** shape as the
megatron-vs-vllm benchmark matrix, the ladder-eval leg, and training's
expert sharding:

```
ONLINE default: TP=1, DP=4, EP=4         one `vllm serve`, one endpoint
+--------------+--------------+--------------+--------------+
| engine 0     | engine 1     | engine 2     | engine 3     |
| full attn +  | full attn +  | full attn +  | full attn +  |  attention/dense:
| dense wts    | dense wts    | dense wts    | dense wts    |  REPLICATED (DP)
| own KV cache | own KV cache | own KV cache | own KV cache |
| 64 experts   | 64 experts   | 64 experts   | 64 experts   |  experts: SHARDED (EP=4)
+--------------+--------------+--------------+--------------+
        ^ API server load-balances requests across the 4 engines;
          the ONLY cross-GPU communication is the MoE all2all
          (allgather_reducescatter — the default backend, and the only
          one that works on Alps: no InfiniBand, no DeepEP)
```

How DP works in `vllm serve`: `--data-parallel-size 4` spawns 4 engine-core
processes behind a single HTTP endpoint. The front-end balances requests by
queue depth; each engine batches and schedules independently with its own
KV cache. Because the expert layers are one EP group spanning all ranks,
the forward passes must stay aligned — a DP coordinator process keeps ranks
in lockstep and idle ranks run empty "dummy" passes so the MoE collective
never blocks. `--max-num-seqs` applies PER RANK.

Why DP attention + EP experts for a MoE like this one: ~90% of the 22B
params are experts, so replicating attention/dense 4x costs only a few GB
per GPU, and in exchange attention/dense run with ZERO communication and
full-size GEMMs. Plain TP=4 instead slices every matmul 4-ways and inserts
two allreduces per layer. TP=4 wins on single-request latency (4 GPUs
cooperate on each token); DP=4/EP=4 wins on throughput under load. You can
measure exactly this trade-off: `TP=4 DP=1 sbatch online_serving.sbatch`.

The OFFLINE job uses TP=4/EP=4 instead: single-process offline vLLM
(`vllm bench throughput`) refuses `data_parallel_size > 1` — offline DP
needs one torchrun-launched process per GPU ("external launcher"), which is
exactly what the custom harness in vllm-play/benchmark_apertus2_vllm.py
implements. Expert sharding is identical (EP = TP x DP = 4 either way);
only the attention treatment differs.

## Reading the online metrics

- **TTFT** (time to first token): queueing + prefill. What "the model is
  thinking..." feels like.
- **TPOT** (time per output token): mean decode step time per request,
  excluding the first token. 1/TPOT = tokens/s a single user sees.
- **ITL** (inter-token latency): the individual gaps between streamed
  tokens. Same thing as TPOT but as a distribution — spikes show scheduler
  stalls (e.g. a big prefill preempting decodes).
- **E2EL**: full request latency, arrival to last token.
- **Request/output-token throughput**: the aggregate rate actually achieved.

Look at medians and p99, not means. With `--request-rate inf` expect ugly
TTFT (everything queues at t=0 by design) and good throughput; with a finite
rate expect the opposite.

## Experiments to try (env knobs, no editing)

```
# Smaller/larger sample:
NUM_PROMPTS=100 sbatch offline_throughput.sbatch

# Latency under a steady load instead of a thundering herd
# (Poisson arrivals at 8 req/s, at most 64 in flight):
REQUEST_RATE=8 MAX_CONCURRENCY=64 sbatch online_serving.sbatch

# Sweep the knee: run REQUEST_RATE = 2, 4, 8, 16, ... and plot p99 TTFT/TPOT
# from the results/ JSONs against achieved throughput.

# DP-attention vs TP-attention head-to-head (same EP=4 expert sharding):
TP=4 DP=1 sbatch online_serving.sbatch

# Reproduce the old benchmark-matrix workload (512-token prompts, 64-token
# outputs, 64 in flight) with the standard CLI instead of ShareGPT — edit
# the dataset flags in online_inner.sh to:
#   --dataset-name random --random-input-len 512 --random-output-len 64
# and submit with MAX_CONCURRENCY=64.
```

## Notes / gotchas

- First run of each job shape pays torch.compile (~10-20 min) before any
  token is produced. It is cached under `vllm-cache/`, so back-to-back runs
  are much cheaper. The benchmark numbers are not affected (timing starts
  after engine build), only the job wall time.
- `HF_HUB_OFFLINE=1` everywhere: model and dataset are local files; the jobs
  can never silently download anything.
- The model needs `--trust-remote-code` (it ships its own
  `modeling_apertus2.py` targeting transformers 5.8.1, which is what the
  container image has).
- Results JSONs are the ground truth; the slurm log has the human-readable
  table at the end.
- Don't compare offline vs online numbers from different `NUM_PROMPTS` /
  dataset seeds; the sampled request mix changes.
