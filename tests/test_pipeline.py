import argparse
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import run_hybrid as rh
from hybrid_backend import BACKEND_NAME, CONTROLLER, parse_method, register_backend
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
register_backend()
cfg = LlamaConfig(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=3,
                  num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=4096)
cfg._attn_implementation = BACKEND_NAME
model = LlamaForCausalLM(cfg).eval()
model.config._attn_implementation = BACKEND_NAME
model.generation_config.eos_token_id = None
NS = argparse.Namespace(window=16, kernel=7, pool="max", group_agg="mean")
ids = torch.randint(0, 500, (1, 600))
n = 600


def e2e(label, max_new=20):
    m = parse_method(label)
    rh.configure(NS, [m])
    cache, first = rh.prefill(model, ids)
    CONTROLLER.set_method(m, 0)
    c = rh.build_cache(model, rh.cache_layers(cache), m.budget)
    assert c.get_seq_length() == (n if m.budget is None or m.budget >= n else m.budget), c.get_seq_length()
    return rh.greedy(model, c, first, n, max_new, set()), CONTROLLER.stats[m.label].as_dict()


full, st = e2e("full")
CONTROLLER.set_method(parse_method("full"), 0)
with torch.inference_mode():
    ref = model.generate(ids, max_new_tokens=20, do_sample=False)[0, n:].tolist()
print("A", full == ref, full[:8], ref[:8])
assert full == ref
big, _ = e2e("snap700")
assert big == full
print("B ok")
for lab in ("snap64", "snap128", "snap256+sys32", "snap256+top32", "full+sys64-mid"):
    toks, st = e2e(lab)
    print(lab, "match_full_frac", sum(a == b for a, b in zip(toks, full)) / len(full),
          {k: round(v, 3) for k, v in st.items()})

labels = ["full", "full+top100000", "full+sys2048", "snap700", "snap64", "snap128", "snap256", "snap512",
          "snap256+sys32", "snap256+top32"]
vs = [parse_method(x) for x in labels]
rh.configure(NS, vs)
cache, first = rh.prefill(model, ids)
CONTROLLER.set_oracle(vs, 0)
rh.greedy(model, cache, first, n, 10, set())
for lab in labels:
    r = CONTROLLER.stats[lab].as_dict()
    print(f"{lab:>16} err={r['rel_err_mean']:.5f} mass={r['mass_kept_mean']:.3f} "
          f"K={r['mean_K_rows']:.0f} V={r['mean_V_rows']:.1f} traffic={r['traffic_frac_of_full']:.3f}")
r = {lab: CONTROLLER.stats[lab].as_dict()["rel_err_mean"] for lab in labels}
assert r["full"] < 1e-6 and r["snap700"] < 1e-6 and r["full+top100000"] < 1e-5
print("PIPELINE TEST PASS")
