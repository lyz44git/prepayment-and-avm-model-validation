"""Valuation study extension: start the scoring service locally (uvicorn, one worker) and measure latency over HTTP
with real out-of-time records. Writes reports/p2/service_latency.json. Local machine only; not a production
load test."""
import sys, os, json, time, subprocess, platform, functools; sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from _paths import WORK, REPORTS
import numpy as np, pandas as pd, httpx
from service import app as svc
from service.tests.conftest import records_from_frame
print = functools.partial(print, flush=True)
PORT = 8765; URL = f"http://127.0.0.1:{PORT}"

S = pd.read_parquet(WORK / "p2" / "features.parquet")
oot = S[S.split == "oot"].sample(6000, random_state=0)
months = ((oot.t_sale // 12 + 1990).astype(str) + "-" + (oot.t_sale % 12 + 1).astype(str).str.zfill(2)).tolist()
recs = records_from_frame(oot, oot.state.tolist(), months)

env = {**os.environ, "MODEL_PATH": str(WORK / "p2" / "c1_gbm.txt")}
proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "service.app:app", "--port", str(PORT), "--log-level", "warning"],
                        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
try:
    c = httpx.Client(base_url=URL, timeout=60)
    for _ in range(100):
        try:
            if c.get("/health").status_code == 200: break
        except httpx.TransportError: time.sleep(0.2)
    health = c.get("/health").json(); print(health)
    c.post("/score", json={"records": recs[:10]})                       # warm-up
    def timed(path, payload, n):
        out = []
        for _ in range(n):
            t = time.perf_counter(); r = c.post(path, json=payload); out.append((time.perf_counter() - t) * 1000)
            assert r.status_code == 200, r.text[:300]
        return np.array(out)
    single = np.concatenate([timed("/score", {"records": [recs[i]]}, 1) for i in range(300)])
    batch = timed("/score", {"records": recs[:1000]}, 20)
    psi_ = timed("/monitor/psi", {"records": recs[:5000]}, 5)
    res = {"machine": f"{platform.machine()}, {os.cpu_count()} cores, {platform.system()} {platform.release()}",
           "server": "uvicorn, 1 worker, localhost HTTP", "model_trees": health["trees"],
           "single_record_ms": {"p50": float(np.percentile(single, 50)), "p95": float(np.percentile(single, 95)), "n": len(single)},
           "batch_1000_ms": {"p50": float(np.percentile(batch, 50)), "p95": float(np.percentile(batch, 95)), "n": len(batch)},
           "batch_1000_records_per_s": float(1000 / (np.percentile(batch, 50) / 1000)),
           "psi_5000_ms": {"p50": float(np.percentile(psi_, 50)), "n": len(psi_)}}
    json.dump(res, open(REPORTS / "p2" / "service_latency.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
finally:
    proc.terminate(); proc.wait(10)
