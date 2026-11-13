"""Run SnapKV / SnapKV+SANTA sweeps on RULER or LongBench JSONL data.

Two modes:

  --mode e2e     One shared exact prefill per example, then for every method a
                 compressed cache is built and greedy-decoded.  Scores task
                 accuracy and counts K/V rows read per KV head per decode step.

  --mode oracle  Dense greedy decode (trajectory = full KV), but at every
                 decode step and layer each variant's attention output is
                 computed and compared to exact.  Reports relative L2 error,
                 cosine, softmax mass kept, and K/V rows.  Cheap, no
                 compounding: the first test of whether a bigger K budget plus
                 V sampling beats SnapKV at the same row traffic.

Method grammar (see hybrid_backend.parse_method):
    full | snap{B} | {full|snap{B}}+sys{S}[-mid] | {full|snap{B}}+top{m}
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from hybrid_backend import BACKEND_NAME, CONTROLLER, parse_method, register_backend  # noqa: E402

RULER_TOKENS = {"fwe": 50, "niah_multivalue": 128, "qa_1": 32, "qa_2": 32}
LONGBENCH_NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
LONGBENCH_FIRST_LINE = {"trec", "triviaqa", "samsum", "lsht"}
LONGBENCH_METRIC = {
    "narrativeqa": "qa_f1_score", "qasper": "qa_f1_score", "multifieldqa_en": "qa_f1_score",
    "hotpotqa": "qa_f1_score", "2wikimqa": "qa_f1_score", "musique": "qa_f1_score",
    "gov_report": "rouge_score", "qmsum": "rouge_score", "multi_news": "rouge_score",
    "trec": "classification_score", "triviaqa": "qa_f1_score", "samsum": "rouge_score",
    "passage_count": "count_score", "passage_retrieval_en": "retrieval_score",
    "lcc": "code_sim_score", "repobench-p": "code_sim_score",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["ruler", "longbench"], required=True)
    p.add_argument("--dataset", required=True, help="JSONL file")
    p.add_argument("--task", required=True, help="RULER task or LongBench dataset name")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["e2e", "oracle"], default="e2e")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--methods", required=True, help="comma-separated method labels")
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--kernel", type=int, default=7)
    p.add_argument("--pool", choices=["max", "avg"], default="max")
    p.add_argument("--group-agg", choices=["mean", "max"], default="mean")
    p.add_argument("--max-examples", type=int, default=50)
    p.add_argument("--max-input-tokens", type=int, default=32768,
                   help="RULER: refuse longer prompts. LongBench: middle-truncate to this.")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--oracle-steps", type=int, default=None,
                   help="decode steps measured per example in oracle mode (default: task max)")
    p.add_argument("--longbench-dir", default=None, help="dir containing LongBench metrics.py")
    p.add_argument("--seed", type=int, default=1690)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Data / prompts
# --------------------------------------------------------------------------

def load_rows(path, n):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(n), f)]


def ruler_ids(tok, rec, max_tokens):
    prompt = rec.get("input") or rec.get("prompt")
    prefix = rec.get("answer_prefix")
    if prefix and prefix.strip() and prefix.strip() not in prompt[-max(256, len(prefix) + 16):]:
        prompt = prompt + prefix
    pre = "<|start_header_id|>" in prompt or prompt.startswith("<|begin_of_text|>") or prompt.startswith("<s>")
    ids = tok(prompt, return_tensors="pt", add_special_tokens=not pre)["input_ids"]
    if ids.shape[-1] > max_tokens:
        raise ValueError(f"RULER prompt has {ids.shape[-1]} tokens > {max_tokens}; refusing to truncate")
    return ids


def longbench_ids(tok, rec, dataset, max_tokens):
    """LongBench pred.py convention: middle-truncate the raw prompt, then chat-wrap."""
    prompt = rec["prompt"]
    raw = tok(prompt, add_special_tokens=False)["input_ids"]
    if len(raw) > max_tokens:
        half = max_tokens // 2
        prompt = tok.decode(raw[:half], skip_special_tokens=True) + tok.decode(raw[-half:], skip_special_tokens=True)
    if dataset in LONGBENCH_NO_CHAT:
        return tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


class Scorer:
    def __init__(self, args):
        self.args = args
        if args.benchmark == "ruler":
            from ruler_metrics import score_prediction, metric_for_task
            self.metric = metric_for_task(args.task)
            self.fn = score_prediction
        else:
            if not args.longbench_dir:
                raise ValueError("--longbench-dir is required for LongBench scoring")
            sys.path.insert(0, args.longbench_dir)
            import metrics as lb_metrics  # LongBench/metrics.py
            self.fn = getattr(lb_metrics, LONGBENCH_METRIC[args.task])

    def __call__(self, pred, rec):
        if self.args.benchmark == "ruler":
            return 100.0 * self.fn(pred, rec.get("outputs", []), self.metric)
        if self.args.task in LONGBENCH_FIRST_LINE:
            pred = pred.lstrip("\n").split("\n")[0]
        best = 0.0
        for gt in rec["answers"]:
            best = max(best, self.fn(pred, gt, all_classes=rec.get("all_classes")))
        return 100.0 * best


# --------------------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------------------

def cache_layers(cache):
    if hasattr(cache, "layers"):
        return [(lay.keys, lay.values) for lay in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))          # transformers 4.x


def build_cache(model, full_layers, budget):
    cache = DynamicCache(config=model.config)
    for i, (k, v) in enumerate(full_layers):
        idx = CONTROLLER.keep_indices(i, budget, k.shape[-2], k.device)
        if idx is not None:
            d = k.shape[-1]
            gi = idx.unsqueeze(-1).expand(-1, -1, -1, d)
            k, v = torch.gather(k, 2, gi), torch.gather(v, 2, gi)
        else:
            k, v = k.clone(), v.clone()
        cache.update(k, v, i)
    return cache


def stop_ids(model, tok, task):
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    eos.add(tok.eos_token_id)
    if task == "samsum":  # LongBench pred.py stops samsum on newline
        eos.add(tok.encode("\n", add_special_tokens=False)[-1])
    return {e for e in eos if e is not None}


@torch.inference_mode()
def prefill(model, ids):
    CONTROLLER.begin_example(ids.shape[-1])
    CONTROLLER.oracle_variants = None
    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    return cache, out.logits[0, -1].argmax().item()


@torch.inference_mode()
def greedy(model, cache, first_tok, n_prompt, max_new, stops):
    toks = [first_tok]
    dev = model.device
    while len(toks) < max_new and toks[-1] not in stops:
        pos = n_prompt + len(toks) - 1
        CONTROLLER.full_context_len = pos + 1
        out = model(input_ids=torch.tensor([[toks[-1]]], device=dev), past_key_values=cache,
                    position_ids=torch.tensor([[pos]], device=dev), use_cache=True)
        toks.append(out.logits[0, -1].argmax().item())
    if toks and toks[-1] in stops:
        toks = toks[:-1]
    return toks


def load_model(name, dtype):
    register_backend()
    tok = AutoTokenizer.from_pretrained(name, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        name, token=os.getenv("HF_TOKEN"), dtype=getattr(torch, dtype),
        attn_implementation=BACKEND_NAME, low_cpu_mem_usage=True,
    ).cuda().eval()
    return tok, model


def configure(args, methods):
    budgets = [m.budget for m in methods if m.budget is not None]
    CONTROLLER.window, CONTROLLER.kernel = args.window, args.kernel
    CONTROLLER.pool, CONTROLLER.group_agg = args.pool, args.group_agg
    CONTROLLER.max_budget = max(budgets) if budgets else 0
    if budgets and min(budgets) <= args.window:
        raise ValueError("every SnapKV budget must exceed --window")
    CONTROLLER.init_stats([m.label for m in methods])


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [parse_method(x) for x in args.methods.split(",") if x.strip()]
    configure(args, methods)

    tok, model = load_model(args.model, args.dtype)
    rows = load_rows(args.dataset, args.max_examples)
    scorer = Scorer(args)
    if args.benchmark == "ruler":
        max_new = args.max_new_tokens or RULER_TOKENS.get(args.task, 64)
    else:
        max_new = args.max_new_tokens  # per-row max_gen otherwise
    stops = stop_ids(model, tok, args.task)
    print(f"model={args.model} gpu={torch.cuda.get_device_name(0)} mode={args.mode} "
          f"methods={[m.label for m in methods]}", flush=True)

    scores = {m.label: [] for m in methods}
    pred_files = {}
    if args.mode == "e2e":
        pred_files = {m.label: open(out_dir / f"{args.task}_{m.label}.jsonl", "w", encoding="utf-8")
                      for m in methods}
    t_start = time.time()
    lengths = []

    for i, rec in enumerate(rows):
        if args.benchmark == "ruler":
            ids = ruler_ids(tok, rec, args.max_input_tokens)
        else:
            ids = longbench_ids(tok, rec, args.task, args.max_input_tokens)
        ids = ids.to(model.device)
        n = ids.shape[-1]
        lengths.append(n)
        mx = max_new or int(rec.get("max_gen", 64))
        torch.cuda.reset_peak_memory_stats()
        cache, first = prefill(model, ids)

        if args.mode == "oracle":
            CONTROLLER.set_oracle(methods, args.seed + i)
            steps = args.oracle_steps or mx
            toks = greedy(model, cache, first, n, steps, stops)
            pred = tok.decode(toks, skip_special_tokens=True).strip()
            print(f"[oracle {i+1}/{len(rows)}] n={n} steps={len(toks)} score={scorer(pred, rec):.1f} "
                  f"pred={pred[:80]!r}", flush=True)
        else:
            full_layers = cache_layers(cache)
            for j, m in enumerate(methods):
                CONTROLLER.set_method(m, args.seed + 7919 * i + j)
                c = build_cache(model, full_layers, m.budget)
                toks = greedy(model, c, first, n, mx, stops)
                del c
                pred = tok.decode(toks, skip_special_tokens=True).strip()
                sc = scorer(pred, rec)
                scores[m.label].append(sc)
                pred_files[m.label].write(json.dumps({
                    "index": rec.get("index", rec.get("_id", i)), "prompt_tokens": n,
                    "pred": pred, "score": sc, "method": m.label,
                    "answers": rec.get("outputs", rec.get("answers")),
                }) + "\n")
                pred_files[m.label].flush()
                print(f"[{m.label} {i+1}/{len(rows)}] n={n} score={sc:.1f} pred={pred[:80]!r}", flush=True)
            del full_layers
        del cache
        print(f"  example {i+1} done; peak_gpu_gib={torch.cuda.max_memory_allocated()/2**30:.1f} "
              f"elapsed={time.time()-t_start:.0f}s", flush=True)

        # Rewrite the summary after each example so a preempted pod still leaves results.
        summary = []
        for m in methods:
            row = {"benchmark": args.benchmark, "task": args.task, "mode": args.mode,
                   "method": m.label, "budget": m.budget or "full", "decoder": m.decoder,
                   "s": m.s, "offset": m.offset, "examples": i + 1,
                   "mean_prompt_tokens": sum(lengths) / len(lengths),
                   "window": args.window, "kernel": args.kernel, "pool": args.pool,
                   "group_agg": args.group_agg, "model": args.model}
            if args.mode == "e2e":
                row["score"] = round(sum(scores[m.label]) / len(scores[m.label]), 3)
            row.update(CONTROLLER.stats[m.label].as_dict())
            summary.append(row)
        name = "summary.csv" if args.mode == "e2e" else "oracle_summary.csv"
        write_csv(out_dir / name, summary)
        if args.mode == "oracle":
            layer_rows = []
            for m in methods:
                for layer, (s, k) in sorted(CONTROLLER.stats[m.label].per_layer_err.items()):
                    layer_rows.append({"method": m.label, "layer": layer, "rel_err_mean": s / max(k, 1)})
            write_csv(out_dir / "oracle_per_layer.csv", layer_rows)

    for f in pred_files.values():
        f.close()
    print(f"\nWrote {out_dir}", flush=True)
    with open(out_dir / ("summary.csv" if args.mode == "e2e" else "oracle_summary.csv"), encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
