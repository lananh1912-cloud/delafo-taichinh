# -*- coding: utf-8 -*-
"""
Thu vien: Toi uu hoa danh muc dau tu bang hoc sau ket hop du lieu tai chinh.
Ke thua DELAFO (Cao et al. 2020) va luan van Phan Thi Thuy An (2023).

Du lieu dau vao: CSV voi cac cot [ticker, date, close, volume, P/B, P/E, ROAA, ROEA]
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

SEED = 42
EMA_WARM = 20   # bo 20 ngay dau o MOI kich ban de so sanh cong bang

# ---------------- 6 kich ban dac trung ----------------
SCENARIOS = {
    "A_gia_kl":    ["close", "vol"],
    "B_them_EMA":  ["close", "vol", "ema10", "ema20"],
    "C_TC_tho":    ["close", "vol", "PB", "PE", "ROAA", "ROEA"],
    "C_TC_rank":   ["close", "vol", "PBr", "PEr", "ROAAr", "ROEAr"],
    "C_TC_z":      ["close", "vol", "PBz", "PEz", "ROAAz", "ROEAz"],  # winsorize + z-score cheo theo ngay
    "C_TC_med":    ["close", "vol", "PBm", "PEm", "ROAAm", "ROEAm"],  # tho + dien khuyet bang trung vi ngay (khong con fill 0)
    "D_EMA_TC_tho":  ["close", "vol", "ema10", "ema20", "PB", "PE", "ROAA", "ROEA"],
    "D_EMA_TC_rank": ["close", "vol", "ema10", "ema20", "PBr", "PEr", "ROAAr", "ROEAr"],
    # bo tinh gon (can CSV co them cot dROE - xem fetch/prepare):
    "E2_bundle":     ["close", "vol", "PBr", "ROEAr", "dROEr"],
    "E3_bundle_EMA": ["close", "vol", "ema10", "ema20", "PBr", "ROEAr", "dROEr"],
    # bo chi so MOI (can CSV morong co cot CFQ, RevG, DE):
    "E4_tho":  ["close", "vol", "ROEAm", "CFQm", "RevGm", "DEm"],   # gia tri goc, dien khuyet trung vi ngay
    "E4_rank": ["close", "vol", "ROEAr", "CFQr", "RevGr", "DEr"],   # thu hang cheo theo ngay
}

# ---------------- tien xu ly ----------------
def calculate_ema(mat, period):
    """mat: (days, tickers). EMA giu nguyen do dai, period-1 dong dau = 0."""
    out = np.zeros_like(mat, dtype=np.float32)
    sma = mat[:period].mean(axis=0)
    mult = 2.0 / (period + 1)
    out[period - 1] = sma
    for i in range(period, len(mat)):
        out[i] = (mat[i] - out[i - 1]) * mult + out[i - 1]
    return out

def rolling(a, window):
    return np.stack([a[i:i + window] for i in range(a.shape[0] - window + 1)], axis=0)

def load_channels(data_path, start="2024-12-31", drop=("VPL",), pe_clip=100):
    """Doc CSV, tra ve dict cac kenh dac trung (days x tickers), loi nhuan y va danh sach ma."""
    df = pd.read_csv(data_path, parse_dates=["date"])
    base_cols = ["ticker", "date", "close", "volume", "PB", "PE", "ROAA", "ROEA"]
    has_droe = len(df.columns) >= 9
    df.columns = base_cols + (["dROE"] if has_droe else []) + list(df.columns[9:])
    df = df[~df.ticker.isin(list(drop))]
    df = df[(df.date >= start) & (df.date.dt.dayofweek <= 4)]
    piv = df.pivot_table(index="date", columns="ticker")
    tickers = list(piv["close"].columns)

    close = piv["close"].ffill()
    daily_ret = close.pct_change()
    close_v = np.nan_to_num(close.values.astype(np.float32), nan=0.0)
    vol_v = piv["volume"].fillna(0).values.astype(np.float32)

    chans = {"close": close_v, "vol": vol_v,
             "ema10": calculate_ema(close_v, 10), "ema20": calculate_ema(close_v, 20)}

    fin_cols = ["PB", "PE", "ROAA", "ROEA"] + (["dROE"] if has_droe else [])
    for f in fin_cols:
        x = piv[f].ffill().reindex(columns=tickers)   # chi dung qua khu -> khong lookahead
        if f == "PE":
            x = x.clip(0, pe_clip)                    # chan outlier P/E TRUOC khi tinh moi kenh dan xuat
        v = x.values.astype(np.float32)
        chans[f] = np.nan_to_num(v, nan=0.0)          # gia tri tho
        r = x.rank(axis=1, pct=True)                  # cross-sectional rank 0-1 theo ngay
        chans[f + "r"] = np.nan_to_num(r.values.astype(np.float32), nan=0.5)
        # z-score cheo theo ngay, sau khi winsorize 2%/98% (goi y GVHD: xu ly outlier truoc)
        xw = x.clip(x.quantile(0.02, axis=1), x.quantile(0.98, axis=1), axis=0)
        mu, sd = xw.mean(axis=1), xw.std(axis=1)
        z = xw.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
        chans[f + "z"] = np.nan_to_num(z.values.astype(np.float32), nan=0.0)  # thieu du lieu -> 0 = trung binh (trung lap)
        # gia tri tho nhung dien khuyet TRUNG LAP: NaN -> trung vi cheo cua ngay do (thay vi 0 = tin hieu cuc doan gia)
        med = x.T.fillna(x.median(axis=1)).T          # vector hoa: dien theo cot ngay
        chans[f + "m"] = np.nan_to_num(med.values.astype(np.float32), nan=0.0)

    # cac cot mo rong (sau dROE): dung kenh tho-dien-trung-vi (m), z va hang (r) nhu fin_cols
    extra_cols = [c for c in df.columns if c not in base_cols + ["dROE"]]
    for f in extra_cols:
        try:
            x = piv[f].ffill().reindex(columns=tickers)
        except KeyError:
            continue
        chans[f] = np.nan_to_num(x.values.astype(np.float32), nan=0.0)
        r = x.rank(axis=1, pct=True)
        chans[f + "r"] = np.nan_to_num(r.values.astype(np.float32), nan=0.5)
        xw = x.clip(x.quantile(0.02, axis=1), x.quantile(0.98, axis=1), axis=0)
        z = xw.sub(xw.mean(axis=1), axis=0).div(xw.std(axis=1).replace(0, np.nan), axis=0)
        chans[f + "z"] = np.nan_to_num(z.values.astype(np.float32), nan=0.0)
        med = x.T.fillna(x.median(axis=1)).T
        chans[f + "m"] = np.nan_to_num(med.values.astype(np.float32), nan=0.0)
    y = np.nan_to_num(daily_ret.values.astype(np.float32), nan=-1e2)
    return chans, y, tickers

def build_xy(chans, y_ret, feats, M, D):
    """Tao tensor X (samples, N, M, c) va y (samples, N, D)."""
    X = np.stack([chans[f] for f in feats], axis=1)   # (days, c, tickers)
    X = X[EMA_WARM:]
    y = y_ret[EMA_WARM:]
    Xr = rolling(X[:-D], M)
    yr = rolling(y[M:], D)
    yr = np.swapaxes(yr, 1, 2)
    Xr = np.moveaxis(Xr, -1, 1)
    return Xr.astype(np.float32), yr.astype(np.float32)

# ---------------- ham mat mat & do do (giu nguyen DELAFO) ----------------
def sharpe_ratio_loss(y_true, y_pred):
    eps = 1e-6
    n = tf.cast(tf.shape(y_true)[1], tf.float32)
    w = tf.expand_dims(y_pred, -1)
    z = tf.reduce_sum(y_true * w, axis=1)
    sum_w = tf.clip_by_value(tf.reduce_sum(w, axis=1), eps, n * (1 - eps))
    rate = z / sum_w
    sharpe = tf.reduce_mean(rate, axis=1) / tf.maximum(tf.math.reduce_std(rate, axis=1), eps)
    constraint = tf.reduce_sum((1.6 - y_pred) * y_pred, axis=1)
    return tf.reduce_mean(3e-3 * constraint - sharpe)

def calc_sharpe(weight, y):
    """weight: (tickers,) nhi phan; y: (tickers, days) loi nhuan hang ngay."""
    eps = 1e-6
    w = np.round(weight)
    sum_w = np.clip(w.sum(), eps, y.shape[0])
    port = (w / sum_w).dot(y).squeeze()
    return np.sqrt(y.shape[1]) * port.mean() / max(port.std(), eps)

# ---------------- mo hinh ----------------
class SelfAttentionLayer(layers.Layer):
    def __init__(self, latent_dim=32, reg=1e-4, **kw):
        super().__init__(**kw)
        self.latent_dim = latent_dim
        self.reg = reg
    def build(self, input_shape):
        h = input_shape[-1]
        self.WQ = self.add_weight(name="WQ", shape=(h, self.latent_dim), initializer="uniform",
                                  regularizer=regularizers.l2(self.reg), trainable=True)
        self.WK = self.add_weight(name="WK", shape=(h, self.latent_dim), initializer="uniform",
                                  regularizer=regularizers.l2(self.reg), trainable=True)
    def call(self, x):
        q = tf.matmul(x, self.WQ); k = tf.matmul(x, self.WK)
        energy = tf.matmul(q, k, transpose_b=True) / self.latent_dim
        return tf.nn.softmax(energy, axis=-1)

def build_model(name, input_shape):
    """name: GRU | BiGRU | SA_BiGRU. input_shape: (N, M, c)."""
    N, M, c = input_shape
    inp = layers.Input(shape=input_shape)
    x = layers.Permute((2, 1, 3))(inp)
    x = layers.Reshape((M, N * c))(x)
    x = layers.BatchNormalization()(x)
    if name == "SA_BiGRU":
        prob = SelfAttentionLayer(32)(x)
        x = layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([prob, x])
        x = layers.Bidirectional(layers.GRU(32, activation="sigmoid",
                kernel_regularizer=regularizers.l2(1e-4)))(x)
    elif name == "SA_GRU":
        prob = SelfAttentionLayer(32)(x)
        x = layers.Lambda(lambda t: tf.matmul(t[0], t[1]))([prob, x])
        x = layers.GRU(32, activation="sigmoid",
                kernel_regularizer=regularizers.l2(1e-4))(x)
    elif name == "BiGRU":
        x = layers.Bidirectional(layers.GRU(32, activation="sigmoid",
                kernel_regularizer=regularizers.l2(1e-3)))(x)
    elif name == "GRU":
        x = layers.GRU(32, activation="sigmoid", kernel_regularizer=regularizers.l2(1e-3))(x)
    else:
        raise ValueError("model phai la GRU | BiGRU | SA_GRU | SA_BiGRU")
    x = layers.BatchNormalization()(x)
    out = layers.Dense(N, kernel_regularizer=regularizers.l2(1e-2))(x)
    out = layers.Activation("sigmoid")(out)
    m = keras.Model(inp, out)
    m.compile(loss=sharpe_ratio_loss, optimizer=keras.optimizers.Adam(1e-3))
    return m
