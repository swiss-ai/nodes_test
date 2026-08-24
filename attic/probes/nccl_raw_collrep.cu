// =============================================================================
// Raw NCCL collective probe -- no PyTorch, no Python, no MPI.
// =============================================================================
//
// One process per GPU (launched by plain srun, 4 tasks/node x 2 nodes).
// Rendezvous: rank 0 writes ncclUniqueId to a file on shared scratch
// (path in $NCCL_RAW_ID_FILE), everyone else polls for it.  Communicators:
//   world  = all 8 ranks (cross-node)  -- vLLM's EP-dispatch group
//   intra  = ncclCommSplit by SLURM_NODEID (each node's 4 GPUs, run
//            concurrently on both nodes, like the torch probe's intra phase)
// Per sample and collective (allgather / allreduce / reducescatter, 64 MiB
// per-rank payload, fresh pseudo-random device data each sweep): barrier
// (1-float allreduce + stream sync), then one timed op.  Group leaders print
// every sample and a mean/std/min/max summary -- same busbw conventions as
// nvidia nccl-tests: AG/RS scale by (n-1)/n, AR by 2(n-1)/n.
//
// Build (inside the vllm-standard container; see nccl_raw_rep.sh):
//   nvcc -O2 -o nccl_raw_collrep nccl_raw_collrep.cu \
//     -I$NCCL/include -L$NCCL/lib -l:libnccl.so.2 -Xlinker -rpath=$NCCL/lib

#include <nccl.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <math.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "rank %d CUDA error %s:%d: %s\n", g_rank, __FILE__, __LINE__, \
            cudaGetErrorString(e)); exit(1); } } while (0)
#define CHECK_NCCL(x) do { ncclResult_t r = (x); if (r != ncclSuccess) { \
    fprintf(stderr, "rank %d NCCL error %s:%d: %s\n", g_rank, __FILE__, __LINE__, \
            ncclGetErrorString(r)); exit(1); } } while (0)

static int g_rank = -1;

__global__ void fill_kernel(float *buf, size_t n, unsigned seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        unsigned v = (unsigned)i * 2654435761u + seed * 40503u + 12345u;
        buf[i] = (float)(v & 0xffffff) / 16777216.0f;
    }
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return v && *v ? atoi(v) : dflt;
}

// 1-float allreduce as a group barrier
static void group_barrier(ncclComm_t comm, float *tiny, cudaStream_t stream) {
    CHECK_NCCL(ncclAllReduce(tiny, tiny, 1, ncclFloat, ncclSum, comm, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
}

typedef struct { double sum, sumsq, mn, mx; int n; } stat_t;
static void stat_add(stat_t *s, double v) {
    s->sum += v; s->sumsq += v * v; s->n++;
    if (s->n == 1 || v < s->mn) s->mn = v;
    if (s->n == 1 || v > s->mx) s->mx = v;
}

int main(void) {
    const char *idfile = getenv("NCCL_RAW_ID_FILE");
    if (!idfile) { fprintf(stderr, "NCCL_RAW_ID_FILE not set\n"); return 1; }
    g_rank        = env_int("SLURM_PROCID", -1);
    int nranks    = env_int("SLURM_NTASKS", -1);
    int local     = env_int("SLURM_LOCALID", -1);
    int nodeid    = env_int("SLURM_NODEID", -1);
    int sweeps    = env_int("NCCL_RAW_SWEEPS", 20);
    size_t mib    = (size_t)env_int("NCCL_RAW_SIZE_MB", 64);
    if (g_rank < 0 || nranks < 2 || local < 0 || nodeid < 0) {
        fprintf(stderr, "not under srun (PROCID/NTASKS/LOCALID/NODEID)\n"); return 1;
    }
    char host[64] = {0};
    gethostname(host, sizeof(host) - 1);

    CHECK_CUDA(cudaSetDevice(local));

    // ---- rendezvous ------------------------------------------------------
    ncclUniqueId uid;
    if (g_rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&uid));
        char tmp[4096];
        snprintf(tmp, sizeof(tmp), "%s.tmp", idfile);
        FILE *f = fopen(tmp, "wb");
        if (!f || fwrite(&uid, sizeof(uid), 1, f) != 1) {
            fprintf(stderr, "rank 0: cannot write %s\n", tmp); return 1;
        }
        fclose(f);
        if (rename(tmp, idfile) != 0) { perror("rename idfile"); return 1; }
    } else {
        int waited = 0;
        FILE *f = NULL;
        while (waited < 180) {
            f = fopen(idfile, "rb");
            if (f) break;
            sleep(1); waited++;
        }
        if (!f || fread(&uid, sizeof(uid), 1, f) != 1) {
            fprintf(stderr, "rank %d: no uid file after %ds\n", g_rank, waited); return 1;
        }
        fclose(f);
    }

    ncclComm_t world, intra;
    CHECK_NCCL(ncclCommInitRank(&world, nranks, uid, g_rank));
    CHECK_NCCL(ncclCommSplit(world, nodeid, local, &intra, NULL));
    int intra_n = 0;
    CHECK_NCCL(ncclCommCount(intra, &intra_n));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // ---- buffers (sized for the world group = worst case) ---------------
    size_t count = mib * 1024 * 1024 / sizeof(float);   // per-rank payload
    float *send, *recv, *tiny;
    CHECK_CUDA(cudaMalloc(&send, count * nranks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&recv, count * nranks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&tiny, sizeof(float)));
    CHECK_CUDA(cudaMemset(tiny, 0, sizeof(float)));

    if (g_rank == 0)
        printf("raw NCCL probe: %d ranks (%d/node), %zu MiB/rank, %d sweeps\n",
               nranks, intra_n, mib, sweeps);

    enum { AG, AR, RS, NCOLL };
    const char *cname[NCOLL] = { "allgather", "allreduce", "reducescatter" };
    // [group 0=intra 1=world][coll]
    stat_t stats[2][NCOLL];
    memset(stats, 0, sizeof(stats));

    for (int sweep = 0; sweep < sweeps; sweep++) {
        for (int grp = 0; grp < 2; grp++) {
            ncclComm_t comm = grp ? world : intra;
            int n = grp ? nranks : intra_n;
            int leader = grp ? (g_rank == 0) : (local == 0);
            const char *gname = grp ? "world" : "intra";
            double bytes = (double)count * sizeof(float);

            group_barrier(world, tiny, stream);  // align everyone between phases
            for (int c = 0; c < NCOLL; c++) {
                size_t fill_n = count * (c == RS ? n : 1);
                fill_kernel<<<256, 256, 0, stream>>>(send, fill_n,
                    (unsigned)(sweep * 31 + c * 7 + g_rank));
                CHECK_CUDA(cudaStreamSynchronize(stream));
                group_barrier(comm, tiny, stream);
                double t0 = now_s();
                switch (c) {
                case AG: CHECK_NCCL(ncclAllGather(send, recv, count, ncclFloat,
                                                  comm, stream)); break;
                case AR: CHECK_NCCL(ncclAllReduce(send, recv, count, ncclFloat,
                                                  ncclSum, comm, stream)); break;
                case RS: CHECK_NCCL(ncclReduceScatter(send, recv, count, ncclFloat,
                                                      ncclSum, comm, stream)); break;
                }
                CHECK_CUDA(cudaStreamSynchronize(stream));
                double dt = now_s() - t0;
                if (leader) {
                    double factor = (c == AR) ? 2.0 * (n - 1) / n
                                              : (double)(n - 1) / n;
                    // nccl-tests busbw: AG/RS use total data = n*bytes
                    double busbw = (c == AR ? bytes * factor
                                            : bytes * n * factor) / dt / 1e9;
                    stat_add(&stats[grp][c], busbw);
                    printf("S %-5s %-9s %-13s %2d %8.2f GB/s\n",
                           gname, host, cname[c], sweep + 1, busbw);
                    fflush(stdout);
                }
            }
        }
    }

    // ---- summaries (each leader for its group) ---------------------------
    for (int grp = 0; grp < 2; grp++) {
        int leader = grp ? (g_rank == 0) : (local == 0);
        if (!leader) continue;
        for (int c = 0; c < NCOLL; c++) {
            stat_t *s = &stats[grp][c];
            double mean = s->sum / s->n;
            double var = s->sumsq / s->n - mean * mean;
            printf("SUM %-5s %-9s %-13s mean %7.2f std %6.2f min %7.2f max %7.2f\n",
                   grp ? "world" : "intra", host, cname[c],
                   mean, sqrt(var > 0 ? var : 0), s->mn, s->mx);
        }
    }
    fflush(stdout);

    ncclCommDestroy(intra);
    ncclCommDestroy(world);
    if (g_rank == 0) unlink(idfile);
    return 0;
}
