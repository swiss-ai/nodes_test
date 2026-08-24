# node-health — focused NCCL probes

The active surface is intentionally limited to three two-node probes. See
[RESULTS.md](RESULTS.md) for the short verdict and direct links to the useful
log rows. The tracked files in `logs/` are corrected-launch evidence;
historical and pre-correction runs are under `attic/`.

## Active probes

- `probes/nccl_p2p_matrix_rep.sh` — custom 64 MiB P2P matrix, all 28 GPU
  pairs, repeated sweeps.
- `probes/nccl_tests_3mode.sh` — stock nccl-tests on intra node A, intra node
  B, and all eight GPUs across both nodes.
- `probes/nccl_tests_cross_pairs.sh` — stock `sendrecv_perf` on each of the 16
  cross-node GPU pairs.

All three allocate four GPUs per node and use PMIx plus
`--network=disable_rdzv_get`. The P2P matrix and three-mode test use one Slurm
task per GPU; the cross-pair test launches one rank per node for each selected
GPU pair while retaining the full-node GPU allocation.

## Run

```bash
# Custom repeated P2P matrix
NCCL_BENCH_NODELIST=nidSUSPECT,nidCONTROL \
  probes/nccl_p2p_matrix_rep.sh

# Stock intra-A / intra-B / cross-8-GPU test
NCCL_BENCH_NODELIST=nidSUSPECT,nidCONTROL \
  probes/nccl_tests_3mode.sh

# Stock test of every individual cross-node GPU pair
NCCL_BENCH_NODELIST=nidSUSPECT,nidCONTROL \
  probes/nccl_tests_cross_pairs.sh
```

The scripts self-submit with `sbatch`. Override the default reservation with
`NCCL_BENCH_RESERVATION`; use `NCCL_BENCH_SWEEPS` or `NCCL_BENCH_PASSES` to
change repetition counts.

Always pair a suspect with a known-clean control. These failures are episodic,
so compare repeated samples and do not clear a node from one quiet run.
