# -*- coding: utf-8 -*-
"""
Tai chi so tai chinh theo quy (P/B, P/E, ROA, ROE) bang thu vien vnstock,
roi ghep voi file gia de tao combined dataset dung dinh dang cho run_experiments.py.

CHAY TREN GOOGLE COLAB HOAC MAY CA NHAN:
    pip install -U vnstock pandas
    python fetch_financial_data.py --price data_2016_2022.csv --out combined_2016_2022.csv

Ghi chu:
- ROA/ROE tu nguon du lieu duoc dung lam xap xi cho ROAA/ROEA (tinh tren binh quan).
- Ngay hieu luc cua so lieu quy = ngay ket thuc quy + 45 ngay (han cong bo BCTC)
  de tranh nhin trom tuong lai (lookahead bias).
"""
import argparse, sys, time
import pandas as pd

VN30 = ("ACB,BID,BSR,CTG,FPT,GAS,GVR,HDB,HPG,LPB,MBB,MSN,MWG,PLX,SAB,"
        "SHB,SSB,SSI,STB,TCB,TPB,VCB,VHM,VIB,VIC,VJC,VNM,VPB,VRE")

def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df

def find_col(df, *keys):
    """Tim cot co ten chua 1 trong cac tu khoa (khong phan biet hoa thuong)."""
    for k in keys:
        for c in df.columns:
            if k.lower() == str(c).lower():
                return c
    for k in keys:
        for c in df.columns:
            if k.lower() in str(c).lower():
                return c
    return None

def fetch_ticker(symbol, sources=("VCI", "TCBS")):
    from vnstock import Vnstock
    last_err = None
    for src in sources:
        try:
            stock = Vnstock().stock(symbol=symbol, source=src)
            df = stock.finance.ratio(period="quarter", lang="en", dropna=False)
            df = flatten_cols(df.reset_index())
            c_year = find_col(df, "yearReport", "year")
            c_quar = find_col(df, "lengthReport", "quarter")
            c_pe   = find_col(df, "P/E", "priceToEarning", "pe")
            c_pb   = find_col(df, "P/B", "priceToBook", "pb")
            c_roe  = find_col(df, "ROE (%)", "ROE", "roe")
            c_roa  = find_col(df, "ROA (%)", "ROA", "roa")
            if not all([c_year, c_quar, c_pe, c_pb, c_roe, c_roa]):
                raise ValueError("thieu cot; cac cot hien co: %s" % list(df.columns)[:25])
            out = df[[c_year, c_quar, c_pe, c_pb, c_roe, c_roa]].copy()
            out.columns = ["year", "quarter", "PE", "PB", "ROEA", "ROAA"]
            out["ticker"] = symbol
            return out, src
        except Exception as e:
            last_err = e
    raise RuntimeError("%s: %s" % (symbol, last_err))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", required=True, help="CSV gia: ticker,date,close,volume")
    ap.add_argument("--tickers", default=VN30)
    ap.add_argument("--out", default="combined_fetched.csv")
    ap.add_argument("--quarterly_out", default="financial_quarterly.csv")
    ap.add_argument("--publish_lag_days", type=int, default=45)
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    frames, fails = [], []
    for i, t in enumerate(tickers):
        try:
            df, src = fetch_ticker(t)
            frames.append(df)
            print("[%d/%d] %s: %d quy (nguon %s)" % (i + 1, len(tickers), t, len(df), src))
        except Exception as e:
            fails.append(t)
            print("[%d/%d] %s LOI: %s" % (i + 1, len(tickers), t, str(e)[:150]))
        time.sleep(1)  # tranh rate limit
    if not frames:
        sys.exit("Khong tai duoc ma nao - cần doi nguon.")
    fin = pd.concat(frames, ignore_index=True)
    fin = fin.dropna(subset=["PB", "PE", "ROAA", "ROEA"], how="all")
    fin = fin[fin.quarter.isin([1, 2, 3, 4])]
    fin["year"] = fin["year"].astype(int)

    # ROE/ROA dang % -> chuyen ve ti le neu can (de khop dinh dang du lieu cu: 0.21 thay vi 21)
    for c in ["ROEA", "ROAA"]:
        if fin[c].abs().median() > 1.5:
            fin[c] = fin[c] / 100.0

    fin["quarter_end"] = fin.apply(
        lambda r: pd.Timestamp(year=int(r.year), month=int(r.quarter) * 3, day=1) + pd.offsets.MonthEnd(0), axis=1)
    fin["avail_date"] = fin["quarter_end"] + pd.Timedelta(days=args.publish_lag_days)
    fin = fin.sort_values(["ticker", "avail_date"])
    fin.to_csv(args.quarterly_out, index=False)
    print("Da luu du lieu quy:", args.quarterly_out, "(%d dong)" % len(fin))
    print("Pham vi:", fin.quarter_end.min().date(), "->", fin.quarter_end.max().date())
    if fails:
        print("Cac ma loi (bo qua):", ",".join(fails))

    # ---- ghep voi gia thanh dataset ngay ----
    px = pd.read_csv(args.price, parse_dates=["date"])
    px.columns = [str(c).strip().lstrip("﻿") for c in px.columns]
    px = px[px.ticker.isin(tickers)].sort_values(["ticker", "date"])
    out = []
    for t, g in px.groupby("ticker"):
        f = fin[fin.ticker == t][["avail_date", "PB", "PE", "ROAA", "ROEA"]].rename(
            columns={"avail_date": "date"}).sort_values("date")
        if f.empty:
            g = g.assign(PB=None, PE=None, ROAA=None, ROEA=None)
            out.append(g)
            continue
        m = pd.merge_asof(g.sort_values("date"), f, on="date")  # ban ghi quy gan nhat DA cong bo
        out.append(m)
    combined = pd.concat(out).sort_values(["date", "ticker"])
    combined = combined[["ticker", "date", "close", "volume", "PB", "PE", "ROAA", "ROEA"]]
    combined.columns = ["ticker", "date", "close", "volume", "P/B", "P/E", "ROAA", "ROEA"]
    combined.to_csv(args.out, index=False)
    cov = combined["P/B"].notna().mean() * 100
    first = combined.dropna(subset=["P/B"]).date.min()
    print("Da luu:", args.out, "(%d dong, %.0f%% dong co chi so TC)" % (len(combined), cov))
    print("Goi y: chay run_experiments.py voi --start", str(first.date()))

if __name__ == "__main__":
    main()
