# hybridKV: SnapKV + systematic SANTA (pilot)

**Hypothesis.** SnapKV keeps `B` prompt positions per KV head, and dense decode
then reads `B` K rows and `B` V rows per step (traffic `2B`). SANTA computes
exact scores over every key it is given but reads only the `U ≤ 4S` V rows it
samples (union across a 4-head GQA group). So at the same traffic, the
hybrid can keep **more keys**: `B' + U ≈ 2B` with `B' > B`. If the extra keys
recover attention mass that SnapKV evicts, and SANTA's V-sampling error is
smaller than that eviction error, the hybrid wins at matched traffic.

**Caveat to keep in view.** The hybrid still *stores* `B'` K and V rows. It
matches SnapKV on bandwidth per decode step, not on KV memory footprint. If
the idea holds up, a later step is to store V for only a subset, or quantise
the V rows that are rarely sampled.

## What is implemented

| file | role |
|---|---|
| `hybrid_backend.py` | HF attention backend. Prefill runs exact SDPA plus SnapKV ranking (window 32, softmax summed over the window, GQA group mean, max-pool kernel 7, top-k, window always kept). Decode runs `dense`, `sys{S}` (systematic SANTA, random or `-mid` offset) or `top{m}` (renormalised top-m on V, the deterministic control) over the kept set. Counts K and V rows per KV head per step. Oracle mode returns exact attention while measuring each variant's error. |
| `run_hybrid.py` | One exact prefill per example, shared by every method (paired comparison). RULER + LongBench (official templates, middle truncation, LongBench `metrics.py`). Modes `e2e` and `oracle`. |
| `validate_hybrid.py` | GPU correctness gate (manual loop == `generate`, no-op budget == full, oracle sanity, passkey). |
| `prepare_longbench.py` | Builds LongBench v1 English JSONL with official prompts and max_gen. |
| `analyze_hybrid.py` | Pareto plots and a **matched.csv**: each hybrid's metric minus SnapKV's metric interpolated at the same measured traffic. **The `delta` column is the merit test.** |
| `ruler_metrics.py` | RULER substring scoring (copied from `adaptive-SANTA`). |
| `setup_env.sh` | Builds the job venv in `/tmp` and prints the GPU. |
| `k8s/` | Finite Nautilus Jobs: validate, oracle, end-to-end RULER, LongBench prep and run, results export. |
| `scripts/fetch-results.bat` | Pulls results to `.\results\` through the export Job's log. |
| `tests/` | CPU tests: `python tests/test_backend.py` and `python tests/test_pipeline.py` (the latter builds a tiny random Llama). |

RULER data comes from the PVC `kevin-ruler-shared` (`/shared/ruler/data/4096_100`), prepared by `adaptive-SANTA`. Results go to `/shared/ruler/results/hybridkv/`.

Method labels: `full`, `snap{B}`, `snap{B}+sys{S}`, `snap{B}+sys{S}-mid`, `snap{B}+top{m}`, `full+sys{S}`.

Already checked locally on CPU with transformers 5.22.2:
- SANTA-sys is unbiased (the mean of 4000 draws is within 0.6% of exact) and its error falls with S.
- GQA union counts are exact.
- SnapKV ranking matches a naive per-head reference.
- Budgets are nested.
- On a tiny random Llama, the manual decode loop matches `model.generate` token for token, and a no-op budget leaves the output unchanged.

## Running (from the repo root)

The jobs clone `main` of this repo (it is public), so push before launching.

```bat
kubectl apply -f k8s\job-hybrid-validate.yaml
kubectl wait --for=condition=complete job/hybridkv-validate --timeout=60m
kubectl logs job/hybridkv-validate
```
Continue only on `HYBRID VALIDATION: PASS`. Then:

1. **Oracle merit test** (cheap, ~1-2 h): `kubectl apply -f k8s\job-hybrid-oracle-ruler.yaml`
   Runs 4 RULER-4K tasks × 25 examples, measures operator error against K+V rows, with no compounding.
2. **End-to-end RULER 4K**: `kubectl apply -f k8s\job-hybrid-e2e-ruler.yaml` (50 examples per task).
3. **LongBench (SnapKV Table 1 reproduction)**:
   `kubectl apply -f k8s\job-longbench-prepare.yaml`, then
   `kubectl apply -f k8s\job-hybrid-e2e-longbench.yaml`.
   The default model is Mistral-7B-Instruct-v0.2, the paper's model (gated: accept its licence on HF).
   Reference numbers come from SnapKV Table 1 (Mistral, All KV vs 1024/2048/4096).
   Our `full` and `snap{1024,2048,4096}` rows should land near them before the hybrid rows mean anything.
4. `scripts\fetch-results.bat` and then `python analyze_hybrid.py results\hybridkv`.

Wait on jobs with `kubectl wait`, never `sleep`. Every job here is finite and exits when done.

## Reading the results

- `mean_KV_rows` is the x-axis: K+V rows per KV head per decode step. Plot against this, not against S, because V reuse within a GQA group makes `U` data-dependent.
- `mass_kept_mean` (oracle) is the fraction of true softmax mass that SnapKV's kept set holds. This is what extra K budget buys.
- `sys` vs `top` at the same `U` tells you whether *stochastic* sampling matters, or whether any V sparsification would do.
- `-mid` vs random offset repeats the earlier midpoint ablation inside the hybrid.

## Known limitations of this pilot

- Batch size 1, HF eager-ish Python decoders: slow, and measures rows, not latency.
- SnapKV selection is per KV head (GQA group mean). The original code predates GQA handling, so `--group-agg max` is offered as an ablation.
- The first generated token comes from the exact prefill for every method, as in SnapKV.
- RULER JSONL on the PVC is templated for Llama-3. Regenerate it for a Mistral run.
