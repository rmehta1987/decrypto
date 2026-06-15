# GPU and node requirements for the Decrypto serving run on Polaris

This note states how much GPU and node capacity the three-model Decrypto
cross-play required on Polaris, how that requirement was derived, and which
resources bound the configuration. It records the realized run (2026-06-12 for
the per-model and orchestration proofs, 2026-06-14 for the production run). It is
a companion to `polaris_pbs_notes.md`, which holds the job ids, logs, and
per-job measurements.

The pipeline serves each model as a separate vLLM HTTP server and runs the
Decrypto game loop as a separate process that issues requests to those servers.
This is an inference workload: the model weights are loaded once and held; there
is no training state, optimizer memory, or weight synchronization.

---

## The configuration that ran

Three models were served concurrently: Qwen2.5-72B-Instruct, Qwen3-8B, and
Qwen3-4B. Qwen2.5-72B occupied all four GPUs of one node under tensor
parallelism (the four cards hold one quarter of the weights each and act as a
single device). Qwen3-8B and Qwen3-4B each occupied a single GPU on a separate
node. The game-loop process ran on a fourth node and used only CPU and the
network; the GPUs on that node were idle because Polaris allocates whole nodes.

The run therefore used four nodes (sixteen A100 GPUs), of which twelve held model
weights and four were unused. The experiment was the full cross-play matrix: with
all three models available to all three roles, the encoder × decoder ×
interceptor product is 27 role combinations, run over 15 environment seeds, for
405 games.

---

## What changed from the single-model test

| | Single-model smoke (2026-06-07) | Three-model run (2026-06-14) |
|---|---|---|
| Models served | 1 (Qwen2.5-0.5B) | 3 (Qwen2.5-72B, Qwen3-8B, Qwen3-4B) |
| Largest model | 0.5 billion parameters | 72 billion parameters |
| Weight placement | one GPU holds a full copy | the 72B is split across four GPUs (tensor parallelism); the two Qwen3 models fit one GPU each |
| Nodes used | 1 (server and experiment fused in one job) | 4 (three server nodes plus one experiment node) |
| Queue | `debug` (single job, ≤1 h) | `capacity` (single multi-node job, no preemption) |
| Games | 1 self-play episode | 405 cross-play games |

---

## How the GPU count was derived

Per-GPU weight memory in half precision is approximately `2 × parameters / TP`
bytes, where `TP` is the tensor-parallel degree. Tensor parallelism also requires
that `TP` divide both the number of attention heads and the number of key/value
heads; these were read from each model's `config.json` rather than assumed.

| Model | Parameters | Heads (attention / key-value) | TP | Weight memory per GPU | KV-cache size (measured) |
|---|---|---|---|---|---|
| Qwen2.5-72B | 72 billion | 64 / 8 | 4 | 33.98 GiB (measured) | 18,800 tokens |
| Qwen3-8B | 8 billion | 32 / 8 | 1 | ~16 GiB (derived) | 133,312 tokens |
| Qwen3-4B | 4 billion | 32 / 8 | 1 | ~8 GiB (derived) | 189,648 tokens |

A Polaris GPU holds 40 GiB. The 72B half-precision weights total about 136 GiB,
which exceeds one GPU and three GPUs but fits four (33.98 GiB per GPU, measured at
load). Tensor-parallel degree 4 is therefore the only single-node placement for
this model on 40 GiB cards; degree 2 would place about 68 GiB on two GPUs and does
not fit, and no 80 GiB partition exists on Polaris (confirmed against the node
specification and `nvidia-smi`). The two Qwen3 models fit one GPU each at
tensor-parallel degree 1.

---

## The binding constraints

The constraint was not the weight capacity of the small models, both of which
load on one GPU with most of the 40 GiB free. Two other limits determined the
configuration.

The first is key/value cache headroom for the 72B at tensor-parallel degree 4.
After the 33.98 GiB of weights, the remaining per-GPU memory is small, and vLLM's
startup profiling — which measures peak activation and sampler memory before
sizing the cache — consumed it. At a memory-utilization fraction of 0.95 or below,
and at the default profiling batch and scheduler-slot counts, the engine computed
zero free memory for the cache and aborted on load (jobs 7197372 and 7197374). The
configuration that loaded set the memory-utilization fraction to 0.97, capped the
profiling and prefill batch at 2,048 tokens, and capped the scheduler at 64
sequences; this produced an 18,800-token cache (job 7197375). The two caps reduce
the memory that profiling reserves; chunked prefill, enabled by default, still
serves prompts up to the 8,192-token context limit by processing them in segments.

The second is request concurrency against the 72B. An 18,800-token cache supports
about 2.3 concurrent requests at the 8,192-token context limit, and proportionally
more for shorter prompts. The game-loop runner starts one operating-system process
per game; issuing all 405 games at once produced more concurrent requests than the
72B server could hold, and the excess requests timed out. In that run the 72B
server aborted 610 of 1,929 requests and the client recorded 312 timeouts; 160 of
the 405 games were lost, distributed across the combinations that used the 72B.
Restricting the number of concurrent games to 24 removed the timeouts (zero
aborted requests of 1,736) and the run completed, at the cost of longer wall-clock
time. The sustainable concurrency was established by the earlier 27-game
single-seed run, which the 72B served without aborts.

---

## Node count and job structure

At one model per node, three server nodes and one experiment node run
concurrently, so the workload requires four nodes scheduled at the same time.
Placing the two single-GPU models on one node, on separate GPUs and ports, would
reduce this to three nodes; this was not implemented because the four-node layout
was sufficient.

The four components must be co-scheduled. A two-job structure — separate server
jobs and a dependent experiment job — was tested and abandoned: on a contended
queue the server jobs reached their walltime before the experiment job was
allocated a node, so no game ran. The realized structure is a single four-node
job that starts one server per node through `mpiexec` and runs the experiment on
the head node. A single job removes the inter-job scheduling gap and is eligible
for the `capacity` queue, which does not preempt running jobs, allows up to 168
hours, and permits one running job per user.

---

## The production run

The production run executed the 405-game matrix as one four-node `capacity` job.
The game loop reported 2 hours 27 minutes of run time; the whole job, including a
360-second server startup, was 2 hours 34 minutes. All 405 games produced results
after the concurrency cap (above) and a parser fix for empty model responses, both
recorded in `polaris_pbs_notes.md`. The allocation cost is small relative to the
account: the production run used about 10 node-hours (four nodes for 2.5 hours),
and the production run together with all probes, smokes, and failed attempts used
a few tens of node-hours against approximately 17,000 available.

---

*Quantities and sources: Polaris node = 4 × A100-SXM4, 40 GiB per GPU (node
specification; `nvidia-smi`, job 7197265). Qwen2.5-72B = 72 billion parameters,
64 attention / 8 key-value heads, 80 layers; weights 33.98 GiB per GPU at
tensor-parallel degree 4, 136 GiB on disk; serving configuration
memory-utilization 0.97, context limit 8,192, profiling/prefill batch 2,048,
scheduler 64 sequences; cache 18,800 tokens, 2.29 concurrent requests at the
context limit (job 7197375). Qwen3-8B and Qwen3-4B = 32 attention / 8 key-value
heads, tensor-parallel degree 1; caches 133,312 and 189,648 tokens (job 7197265).
Overload run (job 7199012): 405 concurrent games, 610 aborted requests of 1,929,
312 client timeouts, 160 games lost. Throttled run (job 7199082): 24 concurrent
games, 0 aborted requests of 1,736, run time 8,851 s, 403 of 405 games; the
remaining 2 recovered by job 7199219 for 405 of 405. Job ids, logs, and the full
ledger: `polaris_pbs_notes.md`.*
