import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import hybrid_backend as hb

torch.manual_seed(0)
B, G, R, C, D = 1, 2, 4, 300, 16
scores = torch.randn(B, G, R, C) * 3
value = torch.randn(B, G, C, D)
p = torch.softmax(scores, -1)
exact = torch.einsum("bgrc,bgcd->bgrd", p, value)
gen = torch.Generator().manual_seed(1)

out, v = hb.decode_attend(scores, value, hb.parse_method("full"), gen)
assert torch.allclose(out, exact, atol=1e-5) and (v == C).all()

out, v = hb.decode_attend(scores, value, hb.parse_method(f"full+top{C}"), gen)
assert torch.allclose(out, exact, atol=1e-5), (out - exact).abs().max()

# unbiasedness of systematic sampling: average of many draws -> exact
m = hb.parse_method("full+sys8")
acc = torch.zeros_like(exact)
N = 4000
for _ in range(N):
    o, v = hb.decode_attend(scores, value, m, gen)
    acc += o
bias = ((acc / N - exact).norm(dim=-1) / exact.norm(dim=-1)).mean()
single = ((o - exact).norm(dim=-1) / exact.norm(dim=-1)).mean()
print(f"sys8: single-draw rel err {single:.3f}, mean-of-{N} rel err {bias:.4f}")
assert bias < 0.05

# error shrinks with S
errs = []
for s in (8, 32, 128, 512):
    o, _ = hb.decode_attend(scores, value, hb.parse_method(f"full+sys{s}"), gen)
    errs.append(float(((o - exact).norm(dim=-1) / exact.norm(dim=-1)).mean()))
print("sys err vs S", errs)
assert errs[-1] < errs[0]

# midpoint deterministic
a, _ = hb.decode_attend(scores, value, hb.parse_method("full+sys16-mid"), gen)
b, _ = hb.decode_attend(scores, value, hb.parse_method("full+sys16-mid"), gen)
assert torch.equal(a, b)

# union count
idx = torch.randint(0, 50, (1, 2, 4, 10))
u = hb._union_count(idx)
for g in range(2):
    assert int(u[0, g]) == len(set(idx[0, g].flatten().tolist()))

# SnapKV ranking vs naive reference
H, Gk, L, w = 8, 2, 200, 16
q = torch.randn(1, H, L, D)
k = torch.randn(1, Gk, L, D)
ctl = hb.CONTROLLER
ctl.window, ctl.kernel, ctl.pool, ctl.group_agg, ctl.max_budget = w, 7, "max", "mean", 120
order = hb._snapkv_order(q, k, 0.25)
r = H // Gk
ref = []
for g in range(Gk):
    votes = []
    for h in range(g * r, (g + 1) * r):
        att = (q[0, h, -w:] @ k[0, g].T) * 0.25
        mask = torch.ones(w, L, dtype=torch.bool)
        for i in range(w):
            mask[i, : L - w + i + 1] = False
        att = torch.softmax(att.masked_fill(mask, float("-inf")), -1)
        votes.append(att[:, : L - w].sum(0))
    vote = torch.stack(votes).mean(0)
    vote = F.max_pool1d(vote.view(1, 1, -1), 7, 1, 3).view(-1)
    ref.append(vote.topk(120 - w).indices)
ref = torch.stack(ref).unsqueeze(0)
assert torch.equal(order.sort(-1).values, ref.sort(-1).values), "snapkv order mismatch"

# keep_indices: budget subset + window + generated tail
ctl.begin_example(L)
ctl.orders[0] = order
idx = ctl.keep_indices(0, 64, L + 3, "cpu")
assert idx.shape == (1, Gk, 64 - w + w + 3)
assert set(range(L - w, L + 3)) <= set(idx[0, 0].tolist())
assert ctl.keep_indices(0, L, L, "cpu") is None
# nested budgets
small = set(ctl.keep_indices(0, 40, L, "cpu")[0, 1].tolist())
big = set(ctl.keep_indices(0, 100, L, "cpu")[0, 1].tolist())
assert small <= big
print("ALL BACKEND UNIT TESTS PASS")
