# -*- coding: utf-8 -*-
"""
BAN HOP NHAT - MOT BASELINE CHO CA BA THI NGHIEM.
Voi moi (M,D) va moi seed:
  1) Huan luyen mo hinh A (gia + khoi luong). Gio cua no cho ra:
       goc    = chia deu von                      -> BASELINE DUY NHAT
       loc    = loai ma ROEA < 10%  (TN2)
       phanbo = von ti le max(ROEA,0)  (TN3)
  2) Huan luyen mo hinh TC (them 4 chi so tho, dien trung vi) tren CUNG fold:
       dauvao = gio cua mo hinh TC, chia deu  (TN1) — so voi goc CUNG NGAY.
Moi so sanh ghep cap voi cung mot day goc.

Chay:
  python stage2_v5.py --data prepared_2018_2026.csv --start 2018-04-02 \
      --md "100,50;63,20" --epochs 30 --batch 128 --seed 42 --out rg3_all_s42
"""
import os, json, time, argparse
import numpy as np
import pandas as pd
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import experiment as E

NGUONG_LOC = 10.0
MIN_GIO = 2

def load_roea_raw(data_path, start, tickers):
    df = pd.read_csv(data_path, parse_dates=["date"])
    base = ["ticker","date","close","volume","PB","PE","ROAA","ROEA"]
    df.columns = base + list(df.columns[len(base):])
    df = df[(df.date >= start) & (df.date.dt.dayofweek <= 4)]
    piv = df.pivot_table(index="date", columns="ticker")
    x = piv["ROEA"].ffill().reindex(columns=tickers)
    have = x.notna().values
    return np.nan_to_num(x.values.astype(np.float32), nan=0.0), have

def sharpe_w(w, y):
    eps = 1e-6
    s = w.sum()
    if s < eps:
        return 0.0
    port = (w / s).dot(y).squeeze()
    return float(np.sqrt(y.shape[1]) * port.mean() / max(port.std(), eps))

def train_and_masks(scen, M, D, a, chans, y_ret):
    """Huan luyen 1 mo hinh theo kich ban, tra ve danh sach (sel, mask) cua cac fold danh gia va y."""
    tf.keras.utils.set_random_seed(a.seed)
    X, y = E.build_xy(chans, y_ret, E.SCENARIOS[scen], M, D)
    model = E.build_model(a.model, X.shape[1:])
    tscv = TimeSeriesSplit(n_splits=a.n_fold)
    out = []
    for k, (tr, te) in enumerate(tscv.split(X)):
        model.fit(X[tr], y[tr], batch_size=a.batch, epochs=a.epochs, verbose=0)
        sel = te[range(D - 1, len(te), a.stride)]
        pred = model.predict(X[sel], verbose=0)
        if k >= a.keep_folds:
            out.append((sel, (pred > 0.5).astype(np.float32)))
    return out, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--start", default="2018-04-02")
    ap.add_argument("--md", default="100,50")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_fold", type=int, default=5)
    ap.add_argument("--keep_folds", type=int, default=2)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--model", default="GRU", help="GRU / SA_GRU / SA_BiGRU / BiGRU")
    ap.add_argument("--out", default="rg3_all")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    detail = open(os.path.join(a.out, "all_detail.jsonl"), "a")
    chans, y_ret, tickers = E.load_channels(a.data, a.start)
    roea, have = load_roea_raw(a.data, a.start, tickers)

    for part in a.md.split(";"):
        M, D = (int(v) for v in part.split(","))
        t0 = time.time()
        evA, y = train_and_masks("A_gia_kl", M, D, a, chans, y_ret)     # mo hinh A
        evC, _ = train_and_masks("C_TC_med", M, D, a, chans, y_ret)    # mo hinh co TC
        res = {v: [] for v in ["goc", "loc", "phanbo", "dauvao"]}
        n_loc = 0
        for (selA, mA), (selC, mC) in zip(evA, evC):
            assert (selA == selC).all(), "lech ngay lap gio giua hai mo hinh"
            idx = E.EMA_WARM + selA + M - 1
            for i in range(len(selA)):
                m = mA[i]
                q, hv = roea[idx[i]], have[idx[i]]
                ml = m * np.where(hv, (q >= NGUONG_LOC).astype(np.float32), 1.0)
                if ml.sum() < MIN_GIO:
                    ml = m
                elif ml.sum() != m.sum():
                    n_loc += 1
                mp = m * np.maximum(q, 0.0)
                res["goc"].append(sharpe_w(m, y[selA][i]))
                res["loc"].append(sharpe_w(ml, y[selA][i]))
                res["phanbo"].append(sharpe_w(mp, y[selA][i]))
                res["dauvao"].append(sharpe_w(mC[i], y[selA][i]))
        g = np.array(res["goc"])
        detail.write(json.dumps({"M": M, "D": D, "seed": a.seed, "model": a.model, "n_loc": n_loc, **res}) + "\n")
        detail.flush()
        print("[M=%d D=%d seed=%d] goc=%.4f | dauvao=%+.4f loc=%+.4f phanbo=%+.4f | n=%d (%ds)"
              % (M, D, a.seed, g.mean(),
                 np.mean(res["dauvao"]) - g.mean(), np.mean(res["loc"]) - g.mean(),
                 np.mean(res["phanbo"]) - g.mean(), len(g), time.time() - t0), flush=True)
    detail.close()
    print("Xong. Ket qua:", a.out)

if __name__ == "__main__":
    main()
