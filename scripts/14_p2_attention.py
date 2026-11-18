"""Valuation study, part 3: neighbour-attention AVM (PyTorch, Apple MPS).

Tokens = the 32 nearest pre-2022-05 comparable sales (indexed log price, size, age, beds, baths, condo, dx, dy,
distance, years since comp sale) plus relative size/age vs. the subject. A 2-layer transformer encoder mixes
the comps; the subject (its own attributes + state/month embeddings) cross-attends to them. The network
predicts the residual from the baseline 'median comp price x market index'. Same splits as the GBM.
"""
import sys, time, json, math, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd, torch, torch.nn as nn
print = functools.partial(print, flush=True)
torch.manual_seed(0); np.random.seed(0)
t0 = time.time(); DW = WORK / "p2"; OUT = REPORTS / "p2"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
S = pd.read_parquet(DW / "features.parquet"); C = np.load(DW / "comp_tensor.npy")       # [n, 32, 10]
assert len(S) == len(C)
NUM = ["prev_idx_logp", "months_since_prev", "log_mv0", "log_av0", "log_cv0", "log_tax0", "age", "log_sqft", "beds",
       "baths", "log_lot", "pool", "condo", "stories0", "units0", "comp_dist1", "comp_dist20", "comp_n_1km",
       "mkt_idx", "lat", "lon", "log_density", "t_owned", "t_medage"]
tr = (S.split == "train").to_numpy(); ca = (S.split == "calib").to_numpy()
base = (S.comp_med20.fillna(S.comp_med20[tr].median()) + S.mkt_idx).to_numpy(np.float32)
y_res = (S.y.to_numpy(np.float32) - base)
# subject features: standardise on train, NaN -> 0 plus a missing flag per column
Xn = S[NUM].to_numpy(np.float32); mu = np.nanmean(Xn[tr], 0); sd = np.nanstd(Xn[tr], 0) + 1e-6
Xf = np.nan_to_num((Xn - mu) / sd); Xm = np.isnan(Xn).astype(np.float32)
Xs = np.concatenate([Xf, Xm], 1).astype(np.float32)
# comp tokens: centre price on the subject's baseline comp level; relative size/age; standardise; mask missing comps
Cc = C.copy()
Cc[:, :, 0] = C[:, :, 0] - S.comp_med20.fillna(0).to_numpy(np.float32)[:, None]
rel_sz = C[:, :, 1] - S.log_sqft.to_numpy(np.float32)[:, None]; rel_age = C[:, :, 2] - S.age.to_numpy(np.float32)[:, None]
Cc = np.concatenate([Cc, rel_sz[..., None], rel_age[..., None]], 2)
cmu = np.nanmean(Cc[tr].reshape(-1, Cc.shape[2]), 0); csd = np.nanstd(Cc[tr].reshape(-1, Cc.shape[2]), 0) + 1e-6
Cmiss = np.isnan(Cc).astype(np.float32)
Ct = np.concatenate([np.nan_to_num((Cc - cmu) / csd), Cmiss], 2).astype(np.float32)
pad = np.isnan(C[:, :, 0])                                                   # comp absent entirely
st_code = S.state.astype("category").cat.codes.to_numpy().astype(np.int64)
st_code = np.where((S.split == "spatial").to_numpy(), st_code.max() + 1, st_code)   # hold-out states -> 'unknown'
moy = (S.t_sale % 12).to_numpy().astype(np.int64)
n_states = int(st_code.max()) + 1

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
    def forward(self, xs, st, mo, ct, pad, need_w=False):
        q = self.sub(torch.cat([xs, self.st(st), self.mo(mo)], 1)).unsqueeze(1)
        h = self.enc(self.tok(ct), src_key_padding_mask=pad)
        ctx, w = self.cross(q, h, h, key_padding_mask=pad, need_weights=need_w, average_attn_weights=True)
        out = self.head(torch.cat([q.squeeze(1), ctx.squeeze(1)], 1)).squeeze(1)
        return (out, w) if need_w else out

def batches(idx, bs, shuffle):
    idx = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        p = pad[b].copy(); p[p.all(1), 0] = False                             # keep at least one key
        yield b, [torch.from_numpy(a).to(dev) for a in (Xs[b], st_code[b], moy[b], Ct[b], p)]

model = CompAttnAVM(Xs.shape[1], Ct.shape[2]).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
tr_idx, ca_idx = np.where(tr)[0], np.where(ca)[0]
EPOCHS, BS = 10, 2048
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=EPOCHS * math.ceil(len(tr_idx) / BS))
loss_fn = nn.HuberLoss(delta=0.3)
def predict(idx, bs=8192):
    model.eval(); out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for k, (b, t) in enumerate(batches(idx, bs, False)):
            out[k * bs:k * bs + len(b)] = model(*t).float().cpu().numpy()
    return out
best, hist = 1e9, []
for ep in range(EPOCHS):
    model.train(); tl = 0.0; nb = 0
    for b, t in batches(tr_idx, BS, True):
        yb = torch.from_numpy(y_res[b]).to(dev)
        loss = loss_fn(model(*t), yb)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        tl += loss.item(); nb += 1
    pc = predict(ca_idx)
    mdape = float(np.median(np.abs(np.exp(pc + base[ca_idx] - S.y.to_numpy()[ca_idx]) - 1)))
    hist.append({"epoch": ep + 1, "train_huber": tl / nb, "calib_MdAPE": mdape, "sec": time.time() - t0}); print(hist[-1])
    if mdape < best:
        best = mdape; torch.save(model.state_dict(), DW / "c2_attention.pt")
model.load_state_dict(torch.load(DW / "c2_attention.pt"))
allp = predict(np.arange(len(S)))
P = S[["addr_id", "split", "state", "t_sale", "price", "y"]].copy()
P["c2_attn"] = allp + base; P["baseline_comp_index"] = base
P.to_parquet(DW / "preds_attn.parquet", index=False)
pd.DataFrame(hist).to_csv(OUT / "c2_training_history.csv", index=False)
# attention diagnostics on 20k OOT rows: weight vs comp distance rank / distance / similarity
oi = np.random.default_rng(1).choice(np.where((S.split == "oot").to_numpy())[0], 20000, replace=False)
model.eval(); W = []
with torch.no_grad():
    for b, t in batches(oi, 4096, False):
        W.append(model(*t, need_w=True)[1].squeeze(1).float().cpu().numpy())
W = np.concatenate(W)
rank_w = W.mean(0)
dist = C[oi, :, 8]; rel = np.abs(C[oi, :, 1] - S.log_sqft.to_numpy()[oi, None])
diag = {"mean_weight_by_neighbour_rank": rank_w.round(4).tolist(),
        "spearman_weight_vs_distance": float(pd.Series(W.ravel()).corr(pd.Series(dist.ravel()), method="spearman")),
        "spearman_weight_vs_size_gap": float(pd.Series(W.ravel()).corr(pd.Series(rel.ravel()), method="spearman"))}
json.dump(diag, open(OUT / "c2_attention_diagnostics.json", "w"), indent=2)
print({k: v for k, v in diag.items() if k != "mean_weight_by_neighbour_rank"}, f"done {time.time()-t0:.0f}s")
