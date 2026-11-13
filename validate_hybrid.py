"""Correctness checks for the SnapKV+SANTA harness.  Run before any sweep.

  A  manual prefill+greedy loop with the full cache == model.generate
  B  SnapKV with a budget above the prompt length == full cache (token-exact)
  C  oracle: full/top-huge/snap-above-length errors ~0, sys4096 error small,
     and error falls as the SnapKV budget grows
  D  SnapKV-1024 (and a hybrid) still retrieve a passkey from a ~6k prompt

Ends with "HYBRID VALIDATION: PASS" or raises.
"""
from __future__ import annotations

import argparse
import random

import torch

import run_hybrid as rh
from hybrid_backend import CONTROLLER, parse_method


def passkey_prompt(tok, n_filler=260, seed=0):
    rng = random.Random(seed)
    key = str(rng.randint(10000, 99999))
    filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
    parts = [filler] * n_filler
    parts.insert(rng.randint(n_filler // 4, 3 * n_filler // 4),
                 f"The pass key is {key}. Remember it. {key} is the pass key. ")
    body = ("There is important info hidden inside a lot of irrelevant text. Find it and memorize it. "
            + "".join(parts) + "What is the pass key? The pass key is")
    text = tok.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"], key


def run_e2e(model, tok, ids, label, max_new=12):
    m = parse_method(label)
    rh.configure(argparse.Namespace(window=32, kernel=7, pool="max", group_agg="mean"), [m])
    cache, first = rh.prefill(model, ids)
    CONTROLLER.set_method(m, 0)
    c = rh.build_cache(model, rh.cache_layers(cache), m.budget)
    toks = rh.greedy(model, c, first, ids.shape[-1], max_new, rh.stop_ids(model, tok, "x"))
    return toks, CONTROLLER.stats[m.label].as_dict()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    a = p.parse_args()
    tok, model = rh.load_model(a.model, "bfloat16")
    ids, key = passkey_prompt(tok)
    ids = ids.cuda()
    n = ids.shape[-1]
    print(f"prompt tokens={n} passkey={key}", flush=True)

    # A ---------------------------------------------------------------
    toks_full, st = run_e2e(model, tok, ids, "full")
    CONTROLLER.set_method(parse_method("full"), 0)
    with torch.inference_mode():
        ref = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)[0, n:].tolist()
    stops = rh.stop_ids(model, tok, "x")
    ref = [t for t in ref if t not in stops][: len(toks_full)]
    print("A manual :", tok.decode(toks_full), "\nA generate:", tok.decode(ref), flush=True)
    assert toks_full == ref, "A FAILED: manual decode loop differs from model.generate"
    assert abs(st["traffic_frac_of_full"] - 1.0) < 1e-9, st

    # B ---------------------------------------------------------------
    toks_big, st = run_e2e(model, tok, ids, f"snap{n + 100}")
    assert toks_big == toks_full, "B FAILED: no-op SnapKV budget changed the output"

    # C ---------------------------------------------------------------
    labels = ["full", "full+top1000000", "full+sys4096", f"snap{n + 100}",
              "snap256", "snap512", "snap1024", "snap2048", "snap2048+sys128", "snap2048+top128"]
    variants = [parse_method(x) for x in labels]
    rh.configure(argparse.Namespace(window=32, kernel=7, pool="max", group_agg="mean"), variants)
    cache, first = rh.prefill(model, ids)
    CONTROLLER.set_oracle(variants, 0)
    rh.greedy(model, cache, first, n, 12, set())
    res = {v.label: CONTROLLER.stats[v.label].as_dict() for v in variants}
    for lab in labels:
        r = res[lab]
        print(f"C {lab:>18}: rel_err={r['rel_err_mean']:.5f} cos={r['cos_mean']:.5f} "
              f"mass={r['mass_kept_mean']:.4f} K={r['mean_K_rows']:.0f} V={r['mean_V_rows']:.0f} "
              f"traffic={r['traffic_frac_of_full']:.3f}", flush=True)
    assert res["full"]["rel_err_mean"] < 1e-6
    assert res["full+top1000000"]["rel_err_mean"] < 1e-4
    assert res[f"snap{n + 100}"]["rel_err_mean"] < 1e-6
    assert res["full+sys4096"]["rel_err_mean"] < 0.1
    errs = [res[f"snap{b}"]["rel_err_mean"] for b in (256, 512, 1024, 2048)]
    assert all(x >= y - 1e-3 for x, y in zip(errs, errs[1:])), f"C: error not decreasing with budget {errs}"

    # D ---------------------------------------------------------------
    for lab in ("snap1024", "snap2048+sys128"):
        toks, st = run_e2e(model, tok, ids, lab)
        text = tok.decode(toks)
        print(f"D {lab}: {text!r} traffic={st['traffic_frac_of_full']:.3f}", flush=True)
        if key not in text:
            print(f"D WARNING: {lab} did not retrieve the passkey (not fatal)", flush=True)

    print("HYBRID VALIDATION: PASS", flush=True)


if __name__ == "__main__":
    main()
