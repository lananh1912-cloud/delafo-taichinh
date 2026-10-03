# -*- coding: utf-8 -*-
"""
Chay toan bo thuc nghiem: 3 mo hinh x 6 kich ban x cac cap (M, D).
Vi du:
    python run_experiments.py --data combined_dataset_2012_2026.csv
    python run_experiments.py --data ... --models SA_BiGRU --scenarios D_EMA_TC_rank --md 63,20
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
import experiment as E

def run_one(model_name, feats, M, D, chans, y_ret, cfg_seed,
            epochs, batch, n_fold, keep_folds, stride):
    tf.keras.utils.set_random_seed(cfg_seed)
    X, y = E.build_xy(chans, y_ret, feats, M, D)
    model = E.build_model(model_name, X.shape[1:])
    ratios = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_fold).split(X)):
        model.fit(X[tr], y[tr], batch_size=batch, epochs=epochs, verbose=0)
        sel = te[range(D - 1, len(te), stride)]
        pred = model.predict(X[sel], verbose=0)
        mask = pred > 0.5
        sr = [E.calc_sharpe(mask[i], y[sel][i]) for i in range(len(sel))]
        if k >= keep_folds:
            ratios.extend(sr)
    return np.array(ratios)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="duong dan CSV: ticker,date,close,volume,P/B,P/E,ROAA,ROEA")
    ap.add_argument("--models", default="GRU,BiGRU,SA_BiGRU")
    ap.add_argument("--scenarios", default=",".join(E.SCENARIOS))
    ap.add_argument("--md", default="63,20;42,14", help="cac cap M,D phan cach boi ';'")
    ap.add_argument("--start", default="2024-12-31", help="ngay bat dau (ngay dau co du lieu tai chinh)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n_fold", type=int, default=5)
    ap.add_argument("--keep_folds", type=int, default=2, help="chi danh gia cac fold k >= gia tri nay")
    ap.add_argument("--stride", type=int, default=5, help="buoc lay danh muc danh gia (ngay)")
    ap.add_argument("--seed", type=int, default=42, help="seed goc (chay nhieu seed de kiem tra do vung)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "results.csv")
    det_path = os.path.join(args.out, "ratios_detail.jsonl")
    if not os.path.exists(res_path):
        pd.DataFrame(columns=["model","scenario","M","D","mean","std","n_port"]).to_csv(res_path, index=False)

    chans, y_ret, tickers = E.load_channels(args.data, start=args.start)
    print("tickers=%d, days=%d" % (len(tickers), chans["close"].shape[0]))

    mds = [tuple(int(v) for v in p.split(",")) for p in args.md.split(";")]
    configs = [(m, s, M, D) for (M, D) in mds
               for m in args.models.split(",") for s in args.scenarios.split(",")]
    for i, (m, s, M, D) in enumerate(configs):
        t0 = time.time()
        r = run_one(m, E.SCENARIOS[s], M, D, chans, y_ret, args.seed + i,
                    args.epochs, args.batch, args.n_fold, args.keep_folds, args.stride)
        pd.DataFrame([[m, s, M, D, round(r.mean(), 4), round(r.std(), 4), len(r)]],
                     columns=["model","scenario","M","D","mean","std","n_port"]
                     ).to_csv(res_path, mode="a", header=False, index=False)
        with open(det_path, "a") as f:
            f.write(json.dumps({"model": m, "scenario": s, "M": M, "D": D,
                                "ratios": [float(v) for v in r]}) + "\n")
        print("[%d/%d] %s %s M=%d D=%d: mean=%.4f std=%.4f n=%d (%.0fs)"
              % (i + 1, len(configs), m, s, M, D, r.mean(), r.std(), len(r), time.time() - t0))
    print("Xong. Ket qua:", res_path)

if __name__ == "__main__":
    main()
