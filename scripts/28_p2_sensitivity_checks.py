"""Valuation study: sensitivity checks (2026-11-19).

1. Exact-rank conformal quantile. 13 calibrates the C1q intervals with np.quantile(E, 0.8 (1 + 1/n)), a linearly
   interpolated quantile; split-conformal coverage is guaranteed for the order statistic of rank ceil((n + 1) 0.8).
   The intervals are recalibrated with that rank and their coverage and width (upper / lower - 1) reported on the
   calibration months, out of time, in the hold-out states and in the hold-out states out of time.
2. Unseen-state embedding. The attention network (14) maps every hold-out state to one 'unknown' state index whose
   embedding never receives a training example, so its value is whatever the initialisation (and weight decay) left.
   The trained network is re-scored on the hold-out states with that embedding replaced by zeros, by the mean of the
   trained state embeddings, and by three fresh draws from the initial N(0, 1); no retraining.
Writes reports/p2/r6_conformal_exact_rank.csv and reports/p2/r6_unseen_state_embedding.csv.
"""
import sys, math, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd, torch, torch.nn as nn
print = functools.partial(print, flush=True)
DW = WORK / "p2"; OUT = REPORTS / "p2"
S = pd.read_parquet(DW / "features.parquet")
G = pd.read_parquet(DW / "preds_gbm.parquet")
A = pd.read_parquet(DW / "preds_attn.parquet")
assert (G.addr_id.to_numpy() == S.addr_id.to_numpy()).all() and (A.addr_id.to_numpy() == S.addr_id.to_numpy()).all()
OOT0 = int(S[S.split == "oot"].t_sale.min())

# ---------------------------------------------------------------- 1. exact-rank conformal quantile
y = S.y.to_numpy(); lo, hi = G.c1_lo.to_numpy(), G.c1_hi.to_numpy()
cal = (S.split == "calib").to_numpy(); n = int(cal.sum())
E = np.maximum(lo[cal] - y[cal], y[cal] - hi[cal])               # residual score of the published (calibrated) intervals
k = math.ceil((n + 1) * 0.8)
delta = float(np.sort(E)[k - 1] - np.quantile(E, min(1, 0.8 * (1 + 1 / n))))
windows = {"calibration 2024-04..06": cal, "out of time": (S.split == "oot").to_numpy(),
           "hold-out states, all months": (S.split == "spatial").to_numpy(),
           "hold-out states, out-of-time months": ((S.split == "spatial") & (S.t_sale >= OOT0)).to_numpy()}
rows = []
for name, m in windows.items():
    for version, (l, h) in (("published (interpolated quantile)", (lo, hi)), ("exact rank ceil((n+1) x 0.8)", (lo - delta, hi + delta))):
        rows.append({"window": name, "version": version, "sales": int(m.sum()), "coverage": float(((y[m] >= l[m]) & (y[m] <= h[m])).mean()),
                     "median_width_upper_over_lower_minus_1": float(np.median(np.exp(h[m] - l[m]) - 1)), "qhat_shift": delta})
cf = pd.DataFrame(rows); cf.to_csv(OUT / "r6_conformal_exact_rank.csv", index=False); print(cf.round(5).to_string(index=False))

# ---------------------------------------------------------------- 2. unseen-state embedding (same preprocessing as 14)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
C = np.load(DW / "comp_tensor.npy")
NUM = ["prev_idx_logp", "months_since_prev", "log_mv0", "log_av0", "log_cv0", "log_tax0", "age", "log_sqft", "beds",
       "baths", "log_lot", "pool", "condo", "stories0", "units0", "comp_dist1", "comp_dist20", "comp_n_1km",
       "mkt_idx", "lat", "lon", "log_density", "t_owned", "t_medage"]
tr = (S.split == "train").to_numpy()
base = (S.comp_med20.fillna(S.comp_med20[tr].median()) + S.mkt_idx).to_numpy(np.float32)
Xn = S[NUM].to_numpy(np.float32); mu = np.nanmean(Xn[tr], 0); sd = np.nanstd(Xn[tr], 0) + 1e-6
Xs = np.concatenate([np.nan_to_num((Xn - mu) / sd), np.isnan(Xn).astype(np.float32)], 1).astype(np.float32)
Cc = C.copy(); Cc[:, :, 0] = C[:, :, 0] - S.comp_med20.fillna(0).to_numpy(np.float32)[:, None]
rel_sz = C[:, :, 1] - S.log_sqft.to_numpy(np.float32)[:, None]; rel_age = C[:, :, 2] - S.age.to_numpy(np.float32)[:, None]
Cc = np.concatenate([Cc, rel_sz[..., None], rel_age[..., None]], 2)
cmu = np.nanmean(Cc[tr].reshape(-1, Cc.shape[2]), 0); csd = np.nanstd(Cc[tr].reshape(-1, Cc.shape[2]), 0) + 1e-6
Ct = np.concatenate([np.nan_to_num((Cc - cmu) / csd), np.isnan(Cc).astype(np.float32)], 2).astype(np.float32)
pad = np.isnan(C[:, :, 0])
st_code = S.state.astype("category").cat.codes.to_numpy().astype(np.int64)
st_code = np.where((S.split == "spatial").to_numpy(), st_code.max() + 1, st_code)
moy = (S.t_sale % 12).to_numpy().astype(np.int64); n_states = int(st_code.max()) + 1

class CompAttnAVM(nn.Module):
    def __init__(self, d_sub, d_tok, d=64, heads=4):
        super().__init__()
        self.st = nn.Embedding(n_states, 8); self.mo = nn.Embedding(12, 4)
        self.sub = nn.Sequential(nn.Linear(d_sub + 12, 128), nn.GELU(), nn.Linear(128, d))
        self.tok = nn.Sequential(nn.Linear(d_tok, 64), nn.GELU(), nn.Linear(64, d))
        layer = nn.TransformerEncoderLayer(d, heads, 128, dropout=0.05, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, xs, st, mo, ct, pad):
        q = self.sub(torch.cat([xs, self.st(st), self.mo(mo)], 1)).unsqueeze(1)
        h = self.enc(self.tok(ct), src_key_padding_mask=pad)
        ctx, _ = self.cross(q, h, h, key_padding_mask=pad, need_weights=False)
        return self.head(torch.cat([q.squeeze(1), ctx.squeeze(1)], 1)).squeeze(1)

model = CompAttnAVM(Xs.shape[1], Ct.shape[2]).to(dev)
model.load_state_dict(torch.load(DW / "c2_attention.pt", map_location=dev)); model.eval()
sp = np.where((S.split == "spatial").to_numpy())[0]
def predict(idx, bs=8192):
    out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; p = pad[b].copy(); p[p.all(1), 0] = False
            t = [torch.from_numpy(a).to(dev) for a in (Xs[b], st_code[b], moy[b], Ct[b], p)]
            out[i:i + len(b)] = model(*t).float().cpu().numpy()
    return out
W0 = model.st.weight.detach().clone(); unk = n_states - 1
variants = {"as trained": W0[unk].clone(), "zeros": torch.zeros(8, device=dev), "mean of trained states": W0[:unk].mean(0)}
for s in (1, 2, 3):
    g = torch.Generator().manual_seed(s); variants[f"fresh N(0,1) draw, seed {s}"] = torch.randn(8, generator=g).to(dev)
yy = S.y.to_numpy()[sp]; c1 = G.c1_gbm.to_numpy()[sp]; late = (S.t_sale.to_numpy()[sp] >= OOT0)
mdape = lambda pr, m: float(np.median(np.abs(np.exp(pr[m] - yy[m]) - 1)))
er = []
for name, vec in variants.items():
    with torch.no_grad(): model.st.weight[unk] = vec
    c2 = predict(sp) + base[sp]; c3 = 0.5 * c1 + 0.5 * c2
    if name == "as trained":
        assert np.allclose(c2, A.c2_attn.to_numpy()[sp], atol=1e-3), "re-scoring does not reproduce the published C2"
    for win, m in (("all months", np.ones(len(sp), bool)), ("out-of-time months", late)):
        er.append({"embedding": name, "window": win, "sales": int(m.sum()), "c2_mdape": mdape(c2, m), "c3_blend_mdape": mdape(c3, m),
                   "c2_median_ratio": float(np.median(np.exp(c2[m] - yy[m])))})
er = pd.DataFrame(er); er.to_csv(OUT / "r6_unseen_state_embedding.csv", index=False); print(er.round(4).to_string(index=False)); print("done")
