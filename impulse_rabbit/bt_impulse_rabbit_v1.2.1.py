#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest — Impulse Grid Rabbit v1.2
-----------------------------------
- 以簡化的 Impulse Grid 策略做回測，重點在「正確輸出每筆交易的 entry_time / exit_time」。
- 需要輸入一份包含至少欄位：time,open,high,low,close 的 K 線 CSV（時間升冪）。

輸出：
- grid_trades_nav_v1.2.csv  : 每筆交易
- grid_baskets_nav_v1.2.csv : 每個籃子彙總

用法：
  python bt_impulse_rabbit_v1.2.py --csv XAUUSD_5min.csv --grid-step 0.5 --levels 6 --tp-mult 1.0 --sl-mult 1.5

備註：
- 本版的目標是正確紀錄交易時間，價格撮合模型採用「區間觸價判定」，未做路徑重建（OHLC 路徑假設）。
- 若用 Excel 開啟 CSV 看不到時間，請改用文字匯入或用 pandas 讀取；輸出已固定 ISO8601 含時區。
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import pandas as pd
import numpy as np


# =========================
# Utilities
# =========================
def to_utc_datetime(s: pd.Series) -> pd.Series:
    """Ensure time column is tz-aware (UTC)."""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt


def within_range(price: float, low_: float, high_: float) -> bool:
    return (price >= low_) and (price <= high_)


# =========================
# Data Structures
# =========================
@dataclass
class Order:
    level: int
    side: str        # "BUY" or "SELL"
    price: float     # limit entry
    tp: float
    sl: float
    units: float
    filled: bool = False
    closed: bool = False
    entry: Optional[float] = None
    exit: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None
    exit_time: Optional[pd.Timestamp] = None


@dataclass
class Basket:
    basket_id: int
    created_time: pd.Timestamp
    grid_levels: List[Order] = field(default_factory=list)
    cost: float = 0.0         # 手續費/點差累計
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    closed_time: Optional[pd.Timestamp] = None
    is_closed: bool = False


# =========================
# Grid Builder
# =========================
def build_grid(center_price: float,
               grid_step: float,
               levels: int,
               tp_mult: float,
               sl_mult: float,
               units_each: float) -> List[Order]:
    """
    以 center_price 為中心，向上（SELL）和向下（BUY）各鋪 levels 層。
    - SELL 格：以更高的價位限價賣出，TP 在下方，SL 在上方
    - BUY  格：以更低的價位限價買入，TP 在上方，SL 在下方
    """
    orders: List[Order] = []

    # 上方 SELL 階梯
    for k in range(1, levels + 1):
        px = center_price + k * grid_step
        tp = px - tp_mult * grid_step
        sl = px + sl_mult * grid_step
        orders.append(Order(
            level=k, side="SELL", price=px, tp=tp, sl=sl, units=units_each
        ))

    # 下方 BUY 階梯
    for k in range(1, levels + 1):
        px = center_price - k * grid_step
        tp = px + tp_mult * grid_step
        sl = px - sl_mult * grid_step
        orders.append(Order(
            level=-k, side="BUY", price=px, tp=tp, sl=sl, units=units_each
        ))

    # 依價格排序（非必要，只是讓 log 好看）
    orders.sort(key=lambda o: o.price)
    return orders


# =========================
# Backtest Engine
# =========================
def run_backtest(df: pd.DataFrame,
                 grid_step: float = 0.5,
                 levels: int = 6,
                 tp_mult: float = 1.0,
                 sl_mult: float = 1.5,
                 units_each: float = 10.0,
                 spread: float = 0.0,
                 commission_per_trade: float = 0.0,
                 new_basket_on_flat: bool = True,
                 force_exit_minutes: Optional[int] = None) -> (pd.DataFrame, pd.DataFrame):

    """
    簡化撮合規則：
    1) 一根 K 內，先檢查限價是否被觸發（low<=buy_px 或 high>=sell_px）。
    2) 成交後，同一根 K 內再檢查是否命中 TP 或 SL（若兩者都在區間內，保守地以 TP/SL 皆可能命中 -> 先以 TP 優先，若你的策略希望先 SL 可自行調換）。
    3) 若設定 force_exit_minutes，超過該持倉時間（以籃子建立時間為基準）則強制平倉（用該 K 的 mid 價）。
    4) 當所有單都關閉，籃子結束；若 new_basket_on_flat=True，下一根 K 建立新的籃子。
    """

    # 準備輸出容器
    trades_log: List[Dict[str, Any]] = []
    basket_log: List[Dict[str, Any]] = []

    active_basket: Optional[Basket] = None
    basket_id_seq = 0

    df = df.copy()
    df["time"] = to_utc_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    for i in range(len(df)):
        t = df.at[i, "time"]
        o_ = float(df.at[i, "open"])
        h_ = float(df.at[i, "high"])
        l_ = float(df.at[i, "low"])
        c_ = float(df.at[i, "close"])

        # 若目前沒有籃子且允許新建，則以當根 close 為中心建新 grid
        if active_basket is None and new_basket_on_flat:
            basket_id_seq += 1
            active_basket = Basket(
                basket_id=basket_id_seq,
                created_time=t,
                grid_levels=build_grid(c_, grid_step, levels, tp_mult, sl_mult, units_each),
                cost=0.0, gross_pnl=0.0, net_pnl=0.0
            )

        if active_basket is None:
            continue  # 沒有可交易的籃子

        # 1) 嘗試撮合所有未成交的限價單
        for od in active_basket.grid_levels:
            if od.filled:
                continue

            # BUY 限價：價格區間觸及 od.price（l_ <= price）
            if od.side == "BUY":
                if l_ <= od.price:
                    od.filled = True
                    od.entry = od.price
                    od.entry_time = t
                    active_basket.cost += commission_per_trade
            else:  # SELL 限價：價格區間觸及 od.price（h_ >= price）
                if h_ >= od.price:
                    od.filled = True
                    od.entry = od.price
                    od.entry_time = t
                    active_basket.cost += commission_per_trade

        # 2) 對已成交且未關閉的單，檢查 TP / SL
        for od in active_basket.grid_levels:
            if not od.filled or od.closed:
                continue

            # 命中條件（以本根 OHLC 區間觸價）
            hit_tp = within_range(od.tp, l_, h_)
            hit_sl = within_range(od.sl, l_, h_)

            # 假設若同根同時命中 TP & SL，TP 優先（可依需求調換）
            exit_px = None
            exit_reason = None
            if hit_tp:
                exit_px = od.tp
                exit_reason = "TP"
            elif hit_sl:
                exit_px = od.sl
                exit_reason = "SL"

            if exit_px is not None:
                # 粗略計算 PnL：BUY 為 (exit - entry)*units；SELL 為 (entry - exit)*units
                raw = (exit_px - od.entry) * od.units if od.side == "BUY" else (od.entry - exit_px) * od.units
                pnl = raw - spread - commission_per_trade
                od.closed = True
                od.exit = exit_px
                od.exit_time = t
                active_basket.gross_pnl += raw
                active_basket.net_pnl += pnl
                active_basket.cost += commission_per_trade

                trades_log.append(dict(
                    basket_id=active_basket.basket_id,
                    entry_time=od.entry_time,
                    exit_time=od.exit_time,
                    side=od.side,
                    entry=od.entry,
                    exit=od.exit,
                    units=od.units,
                    pnl=pnl,
                    reason=exit_reason
                ))

        # 3) 強制平倉（依籃子建立時間）
        if force_exit_minutes is not None:
            elapsed = (t - active_basket.created_time).total_seconds() / 60.0
            if elapsed >= force_exit_minutes:
                mid = (h_ + l_) / 2.0
                for od in active_basket.grid_levels:
                    if od.filled and not od.closed:
                        raw = (mid - od.entry) * od.units if od.side == "BUY" else (od.entry - mid) * od.units
                        pnl = raw - spread - commission_per_trade
                        od.closed = True
                        od.exit = mid
                        od.exit_time = t
                        active_basket.gross_pnl += raw
                        active_basket.net_pnl += pnl
                        active_basket.cost += commission_per_trade

                        trades_log.append(dict(
                            basket_id=active_basket.basket_id,
                            entry_time=od.entry_time,
                            exit_time=od.exit_time,
                            side=od.side,
                            entry=od.entry,
                            exit=od.exit,
                            units=od.units,
                            pnl=pnl,
                            reason="FORCE"
                        ))

        # 4) 若此籃子的所有已成交單皆關閉，則結束籃子
        #    條件：沒有任何 (filled 且未 closed) 的單存在；若所有單都未成交則跳過，直到有成交並結束。
        filled_exist = any(od.filled for od in active_basket.grid_levels)
        open_exist = any(od.filled and not od.closed for od in active_basket.grid_levels)

        if filled_exist and (not open_exist):
            active_basket.closed_time = t
            active_basket.is_closed = True

            basket_log.append(dict(
                basket_id=active_basket.basket_id,
                created_time=active_basket.created_time,
                closed_time=active_basket.closed_time,
                cost=active_basket.cost,
                gross_pnl=active_basket.gross_pnl,
                net_pnl=active_basket.net_pnl
            ))
            # 清空，等待下一根建立新籃子
            active_basket = None

    # 若最後仍有未結束的籃子，寫出至 log（但不強平）
    if active_basket is not None:
        basket_log.append(dict(
            basket_id=active_basket.basket_id,
            created_time=active_basket.created_time,
            closed_time=None,
            cost=active_basket.cost,
            gross_pnl=active_basket.gross_pnl,
            net_pnl=active_basket.net_pnl
        ))

    trades_df = pd.DataFrame(trades_log)
    baskets_df = pd.DataFrame(basket_log)

    # 時間欄位統一為 tz-aware datetime
    if not trades_df.empty:
        for col in ["entry_time", "exit_time"]:
            trades_df[col] = pd.to_datetime(trades_df[col], utc=True, errors="coerce")

    if not baskets_df.empty:
        for col in ["created_time", "closed_time"]:
            baskets_df[col] = pd.to_datetime(baskets_df[col], utc=True, errors="coerce")

    return trades_df, baskets_df


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Backtest — Impulse Grid Rabbit v1.2")
    ap.add_argument("--csv", required=True, help="Input OHLC CSV with columns: time,open,high,low,close")
    ap.add_argument("--grid-step", type=float, default=0.5, help="Grid step in price units")
    ap.add_argument("--levels", type=int, default=6, help="Levels up and down from center")
    ap.add_argument("--tp-mult", type=float, default=1.0, help="TP multiple of grid-step")
    ap.add_argument("--sl-mult", type=float, default=1.5, help="SL multiple of grid-step")
    ap.add_argument("--units-each", type=float, default=10.0, help="Units per grid order")
    ap.add_argument("--spread", type=float, default=0.0, help="Spread cost per order (absolute)")
    ap.add_argument("--commission", type=float, default=0.0, help="Commission per order (absolute)")
    ap.add_argument("--force-exit-minutes", type=int, default=None, help="Force close basket after N minutes (optional)")
    ap.add_argument("--out-trades", default="grid_trades_nav_v1.2.csv", help="Output trades csv")
    ap.add_argument("--out-baskets", default="grid_baskets_nav_v1.2.csv", help="Output baskets csv")
    args = ap.parse_args()

    # 讀檔與初步檢查
    df = pd.read_csv(args.csv)
    req_cols = {"time", "open", "high", "low", "close"}
    missing = req_cols - set(c.lower() for c in df.columns)
    # 寬鬆處理：做一個小寫 mapping
    lower_map = {c.lower(): c for c in df.columns}
    if missing:
        raise ValueError(f"CSV 缺少欄位：{missing}. 需要欄位：{req_cols}")

    # 標準化欄位名稱（保留原欄位，但新增標準欄）
    df["time"] = df[lower_map["time"]]
    df["open"] = pd.to_numeric(df[lower_map["open"]], errors="coerce")
    df["high"] = pd.to_numeric(df[lower_map["high"]], errors="coerce")
    df["low"]  = pd.to_numeric(df[lower_map["low"]], errors="coerce")
    df["close"]= pd.to_numeric(df[lower_map["close"]], errors="coerce")

    # 丟棄缺失列
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).copy()

    trades_df, baskets_df = run_backtest(
        df,
        grid_step=args.grid_step,
        levels=args.levels,
        tp_mult=args.tp_mult,
        sl_mult=args.sl_mult,
        units_each=args.units_each,
        spread=args.spread,
        commission_per_trade=args.commission,
        new_basket_on_flat=True,
        force_exit_minutes=args.force_exit_minutes
    )

    # === 輸出：固定時間格式（含時區） ===
    date_fmt = "%Y-%m-%dT%H:%M:%S%z"  # ISO8601 with timezone

    # 為了避免 Excel 誤判，明確輸出為文字的日期時間格式
    trades_df.to_csv(args.out_trades, index=False, date_format=date_fmt)
    baskets_df.to_csv(args.out_baskets, index=False, date_format=date_fmt)

    # 簡短總結
    if not baskets_df.empty:
        total_net = baskets_df["net_pnl"].sum()
        n_baskets = len(baskets_df)
        print(f"[Done] baskets={n_baskets} | total_net_pnl={total_net:.2f} | out: {args.out_trades}, {args.out_baskets}")
    else:
        print(f"[Done] No baskets closed. out: {args.out_trades}, {args.out_baskets}")


if __name__ == "__main__":
    main()
