"""SnapKV (K/V eviction after prefill) + systematic SANTA (V sampling at decode).

Pipeline per example (batch size 1):

  1. Prefill runs exact SDPA.  While it runs, every layer also scores the
     prompt positions SnapKV-style: the last ``window`` prompt queries attend
     to the prefix, softmax weights are summed over those queries, averaged
     over the query heads of each GQA group, max-pooled along the sequence,
     and ranked.  The ranking for the largest budget in the sweep is stored;
     a smaller budget is a prefix of it (topk is sorted), so one prefill
     serves every budget.
  2. The runner builds a compressed cache per method: the top ``B - window``
     prefix positions plus the observation window, per KV head.
  3. Decode (q_len == 1) attends over the compressed cache with one of:
        dense     exact softmax(qK^T) V                       reads C K-rows, C V-rows
        sys{S}    exact scores, systematic SANTA on V          reads C K-rows, |union idx| V-rows
        top{m}    exact scores, renormalised top-m on V        reads C K-rows, |union idx| V-rows
     ``-mid`` on sys uses the constant 0.5 offset (deterministic control).

Row accounting is per KV head per decode step, because under GQA a K/V row is
fetched once for the whole group.  For sampled decoders the V count is the
union of indices over the group's query heads.  K and V rows have the same
width (head_dim), so "rows" is proportional to bytes.

Oracle mode (``CONTROLLER.oracle_variants`` set) keeps the full cache and
returns the exact attention output -- so the generated trajectory is the
dense one -- while measuring, at every decode step and layer, the error each
variant *would* have made.  This isolates operator error from compounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Method specs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Method:
    label: str
    budget: int | None          # None -> full cache (no SnapKV)
    decoder: str                # dense | sys | top
    s: int | None = None        # samples (sys) or kept rows (top)
    offset: str = "random"      # random | midpoint (sys only)


def parse_method(text: str) -> Method:
    """`full`, `snap1024`, `snap2048+sys128`, `snap2048+sys128-mid`, `full+top256`."""
    label = text.strip().lower()
    if label == "dense":
        label = "full"
    kv, _, dec = label.partition("+")
    if kv == "full":
        budget = None
    elif kv.startswith("snap"):
        budget = int(kv[4:])
    else:
        raise ValueError(f"bad KV part in {text!r}")
    if not dec or dec == "dense":
        return Method(label, budget, "dense")
    offset = "random"
    if dec.endswith("-mid"):
        dec, offset = dec[:-4], "midpoint"
    if dec.startswith("sys"):
        return Method(label, budget, "sys", int(dec[3:]), offset)
    if dec.startswith("top"):
        if offset != "random":
            raise ValueError("-mid only applies to sys")
        return Method(label, budget, "top", int(dec[3:]))
    raise ValueError(f"bad decoder part in {text!r}")


# --------------------------------------------------------------------------
# Controller (global, like santa_backend.CONTROLLER)
# --------------------------------------------------------------------------

@dataclass
class Stats:
    events: int = 0                 # (decode step, layer, kv head) triples
    k_rows: int = 0
    v_rows: int = 0
    full_rows: int = 0              # K-rows a full-cache dense decode would read
    # oracle-only
    err_sum: float = 0.0            # sum over (step, layer, q-head) of rel L2 error
    err_sq_sum: float = 0.0
    cos_sum: float = 0.0
    mass_sum: float = 0.0           # softmax mass retained by the KV subset
    err_n: int = 0
    per_layer_err: dict = field(default_factory=dict)   # layer -> [sum, n]

    def as_dict(self):
        e = max(self.events, 1)
        out = {
            "events": self.events,
            "mean_K_rows": self.k_rows / e,
            "mean_V_rows": self.v_rows / e,
            "mean_KV_rows": (self.k_rows + self.v_rows) / e,
            "mean_full_KV_rows": 2 * self.full_rows / e,
            "traffic_frac_of_full": (self.k_rows + self.v_rows) / max(2 * self.full_rows, 1),
        }
        if self.err_n:
            m = self.err_sum / self.err_n
            out.update({
                "rel_err_mean": m,
                "rel_err_std": max(self.err_sq_sum / self.err_n - m * m, 0.0) ** 0.5,
                "cos_mean": self.cos_sum / self.err_n,
                "mass_kept_mean": self.mass_sum / self.err_n,
            })
        return out


class HybridController:
    def __init__(self):
        self.window = 32
        self.kernel = 7
        self.pool = "max"          # max | avg
        self.group_agg = "mean"    # mean | max  (how query heads of a GQA group vote)
        self.max_budget = 0        # largest SnapKV budget in the sweep
        self.seed = 0
        self.method: Method = parse_method("full")
        self.oracle_variants: list[Method] | None = None
        self.prompt_len = 0
        self.full_context_len = 0  # set by the runner each decode step
        self.orders: dict[int, torch.Tensor] = {}
        self.stats: dict[str, Stats] = {}
        self._gen: dict[str, torch.Generator] = {}

    # -- lifecycle -------------------------------------------------------
    def begin_example(self, prompt_len: int):
        self.prompt_len = prompt_len
        self.orders = {}

    def init_stats(self, labels):
        """Stats accumulate across examples until init_stats is called again."""
        self.stats = {lab: Stats() for lab in labels}

    def set_method(self, m: Method, seed: int):
        self.method = m
        self.oracle_variants = None
        self.stats.setdefault(m.label, Stats())
        self._reseed(seed)

    def set_oracle(self, variants: list[Method], seed: int):
        self.oracle_variants = list(variants)
        for v in variants:
            self.stats.setdefault(v.label, Stats())
        self._reseed(seed)

    def _reseed(self, seed):
        self.seed = seed
        self._gen = {}

    def generator(self, device):
        key = str(device)
        if key not in self._gen:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            self._gen[key] = g
        return self._gen[key]

    # -- SnapKV index sets -----------------------------------------------
    def keep_indices(self, layer: int, budget: int | None, cache_len: int, device):
        """Per-KV-head indices [1, G, K] into a cache of length cache_len, or None for 'all'.

        Positions >= prompt_len (generated tokens) are always kept.
        """
        L = self.prompt_len
        if budget is None or L <= budget:
            return None
        w = min(self.window, L)
        order = self.orders[layer]                      # [1, G, max_budget - w] sorted by score
        sel = order[..., : budget - w]
        sel, _ = sel.sort(dim=-1)
        g = sel.shape[1]
        tail = torch.arange(L - w, cache_len, device=device).view(1, 1, -1).expand(1, g, -1)
        return torch.cat([sel, tail], dim=-1)


CONTROLLER = HybridController()


# --------------------------------------------------------------------------
# Decoders over an (already restricted) score/value set
# --------------------------------------------------------------------------

def _union_count(idx: torch.Tensor) -> torch.Tensor:
    """idx [B, G, R, S] -> number of distinct indices per (B, G) over R*S."""
    b, g = idx.shape[:2]
    flat, _ = idx.reshape(b, g, -1).sort(dim=-1)
    return 1 + (flat[..., 1:] != flat[..., :-1]).sum(dim=-1)


def decode_attend(scores: torch.Tensor, value: torch.Tensor, m: Method, gen):
    """scores fp32 [B,G,R,C] (masked, scaled); value [B,G,C,D].

    Returns (out fp32 [B,G,R,D], v_rows [B,G] long).
    """
    b, g, r, c = scores.shape
    d = value.shape[-1]
    p = torch.softmax(scores, dim=-1)
    vf = value.float()
    if m.decoder == "dense":
        out = torch.einsum("bgrc,bgcd->bgrd", p, vf)
        return out, torch.full((b, g), c, device=scores.device, dtype=torch.long)

    if m.decoder == "top":
        k = min(m.s, c)
        vals, idx = p.topk(k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        gathered = torch.gather(vf.unsqueeze(2).expand(b, g, r, c, d), 3,
                                idx.unsqueeze(-1).expand(b, g, r, k, d))
        out = (vals.unsqueeze(-1) * gathered).sum(dim=3)
        return out, _union_count(idx)

    if m.decoder == "sys":
        s = m.s
        cdf = p.cumsum(dim=-1)
        if m.offset == "midpoint":
            u = torch.full((b, g, r, 1), 0.5, device=p.device)
        else:
            u = torch.rand((b, g, r, 1), device=p.device, generator=gen)
        j = torch.arange(s, device=p.device, dtype=torch.float32).view(1, 1, 1, s)
        # Normalise by the last CDF value so fp32 round-off never pushes a
        # threshold past the end of the CDF.
        thresholds = (j + u) / s * cdf[..., -1:]
        idx = torch.searchsorted(cdf.contiguous(), thresholds.contiguous(), right=True).clamp_max(c - 1)
        gathered = torch.gather(vf.unsqueeze(2).expand(b, g, r, c, d), 3,
                                idx.unsqueeze(-1).expand(b, g, r, s, d))
        return gathered.mean(dim=3), _union_count(idx)

    raise ValueError(m.decoder)


# --------------------------------------------------------------------------
# Attention function
# --------------------------------------------------------------------------

def _grouped_scores(module, query, key, scaling, attention_mask):
    """query [B,H,1,D], key [B,G,C,D] -> fp32 scores [B,G,R,C]."""
    b, h, q, d = query.shape
    g = key.shape[1]
    r = h // g
    qg = query.reshape(b, g, r, d).float()
    scores = torch.einsum("bgrd,bgcd->bgrc", qg, key.float()) * float(scaling)
    if attention_mask is not None:
        m = attention_mask[..., -1:, :]                       # [B,1|H,1,C]
        if m.shape[1] != 1:
            raise ValueError("per-head masks not supported")
        m = m.reshape(b, 1, 1, -1)
        if m.dtype == torch.bool:
            scores = scores.masked_fill(~m, float("-inf"))
        else:
            scores = scores + m.float()
    return scores


def _snapkv_order(query, key, scaling):
    """Rank prefix positions per KV head; returns [1, G, max_budget - w] (desc score)."""
    ctl = CONTROLLER
    b, h, L, d = query.shape
    g = key.shape[1]
    r = h // g
    w = min(ctl.window, L)
    keep = min(ctl.max_budget, L) - w
    if keep <= 0:
        return None
    qw = query[:, :, -w:, :].reshape(b, g, r, w, d).float()
    att = torch.einsum("bgrwd,bgcd->bgrwc", qw, key.float()) * float(scaling)   # [B,G,R,w,L]
    # causal inside the window: window query i sits at absolute L-w+i
    causal = torch.ones(w, w, dtype=torch.bool, device=att.device).triu(1)
    att[..., L - w:] = att[..., L - w:].masked_fill(causal, float("-inf"))
    att = torch.softmax(att, dim=-1)[..., : L - w].sum(dim=3)                  # [B,G,R,L-w]
    vote = att.mean(dim=2) if ctl.group_agg == "mean" else att.amax(dim=2)     # [B,G,L-w]
    pad = ctl.kernel // 2
    if ctl.pool == "max":
        vote = F.max_pool1d(vote, kernel_size=ctl.kernel, stride=1, padding=pad)
    else:
        vote = F.avg_pool1d(vote, kernel_size=ctl.kernel, stride=1, padding=pad, count_include_pad=False)
    vote = vote[..., : L - w]
    return vote.topk(keep, dim=-1, sorted=True).indices


def hybrid_attention_forward(module, query, key, value, attention_mask, scaling,
                             dropout=0.0, **kwargs):
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    ctl = CONTROLLER
    if query.shape[0] != 1:
        raise ValueError("hybrid backend supports batch size 1 only")
    layer = int(getattr(module, "layer_idx", -1))

    # ---- prefill: exact, plus SnapKV ranking --------------------------
    if query.shape[-2] != 1:
        order = _snapkv_order(query, key, scaling)
        if order is not None:
            ctl.orders[layer] = order
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)

    b, h, _, d = query.shape
    g = key.shape[1]
    c = key.shape[-2]

    # ---- oracle: exact output returned, variants measured -------------
    if ctl.oracle_variants is not None:
        out, _ = sdpa_attention_forward(module, query, key, value, attention_mask,
                                        scaling=scaling, dropout=dropout, **kwargs)
        scores = _grouped_scores(module, query, key, scaling, attention_mask)   # [1,G,R,C]
        p_full = torch.softmax(scores, dim=-1)
        exact = torch.einsum("bgrc,bgcd->bgrd", p_full, value.float())
        exact_norm = exact.norm(dim=-1).clamp_min(1e-12)
        gen = ctl.generator(query.device)
        cache_by_budget = {}
        for m in ctl.oracle_variants:
            if m.budget not in cache_by_budget:
                idx = ctl.keep_indices(layer, m.budget, c, query.device)
                if idx is None:
                    cache_by_budget[m.budget] = (scores, value, p_full.new_ones(p_full.shape[:-1]), c)
                else:
                    r = h // g
                    sc = torch.gather(scores, 3, idx.unsqueeze(2).expand(b, g, r, -1))
                    vv = torch.gather(value, 2, idx.unsqueeze(-1).expand(b, g, -1, d))
                    mass = torch.gather(p_full, 3, idx.unsqueeze(2).expand(b, g, r, -1)).sum(-1)
                    cache_by_budget[m.budget] = (sc, vv, mass, idx.shape[-1])
            sc, vv, mass, k_rows = cache_by_budget[m.budget]
            approx, v_rows = decode_attend(sc, vv, m, gen)
            err = (approx - exact).norm(dim=-1) / exact_norm                     # [1,G,R]
            cos = F.cosine_similarity(approx, exact, dim=-1)
            st = ctl.stats[m.label]
            st.events += g
            st.k_rows += k_rows * g
            st.v_rows += int(v_rows.sum())
            st.full_rows += c * g
            st.err_sum += float(err.sum())
            st.err_sq_sum += float((err * err).sum())
            st.cos_sum += float(cos.sum())
            st.mass_sum += float(mass.sum())
            st.err_n += err.numel()
            pl = st.per_layer_err.setdefault(layer, [0.0, 0])
            pl[0] += float(err.sum())
            pl[1] += err.numel()
        return out, None

    # ---- end-to-end decode over the (compressed) cache ----------------
    m = ctl.method
    st = ctl.stats[m.label]
    st.events += g
    st.k_rows += c * g
    st.full_rows += ctl.full_context_len * g
    if m.decoder == "dense":
        st.v_rows += c * g
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kwargs)
    scores = _grouped_scores(module, query, key, scaling, attention_mask)
    out, v_rows = decode_attend(scores, value, m, ctl.generator(query.device))
    st.v_rows += int(v_rows.sum())
    out = out.reshape(b, h, 1, d).transpose(1, 2).contiguous().to(value.dtype)
    return out, None


BACKEND_NAME = "hybridkv"


def register_backend():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register(BACKEND_NAME, hybrid_attention_forward)
    AttentionMaskInterface.register(BACKEND_NAME, sdpa_mask)
