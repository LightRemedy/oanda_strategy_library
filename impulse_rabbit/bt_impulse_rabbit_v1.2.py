#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# ========= 基本參數 =========
PAIR = "XAUUSD"
TIMEFRAME = "M5"

# 初始資金與風險
initial_balance = 10_000.0   # 初始 NAV
risk_pct = 0.01              # 每個「籃子」可承受的最大風險比例（相對 NAV 或 base_nav）

# 複利與上限
compound = False             # True=隨 NAV 複利, False=固定 base_nav 風險基準
base_nav = 10_000.0          # compound=False 時使用的 NAV 基準
expected_fills_for_risk = 3  # 預期會成交的格數（用於分攤風險）
max_units_each   = 500       # 每筆單位上限
max_total_units  = 1_500     # 單一籃子「已成交」總單位上限

# 交易成本（簡化）
spread = 0.05                # round-trip spread 模擬（USD），會在計算 PnL 時一次扣除
commission_per_trade = 0.0   # 每筆成交的固定手續費（USD）

# 策略參數
n_atr = 14
impulse_lookback = 6
impulse_k_atr = 1.8

grid_k_atr = 0.6
levels = 6
tp_steps = 1.0
sl_steps = 2.0

basket_be_atr = 0.7          # 可用於保本移 SL（此版先不移，保留參數）
max_hold_bars = 72           # 時間止損
cooldown_bars = 24

mode = "mean_revert"         # "mean_revert" 或 "trend_follow"

# ========= 工具 =========
def atr(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

class GridState:
    def __init__(self):
        self.active = False
        self.dir = None
        self.center_price = None
        self.grid_levels = []     # 每格：{level, side, price, tp, sl, filled, entry, exit, units, closed}
        self.open_time_idx = None
        self.net_pnl = 0.0        # USD 合計（含交易成本）
        self.cost = 0.0
        self.units_each = 0
        self.max_total_units = max_total_units

    def reset(self):
        self.__init__()

def build_grid(center, step, direction, units_each):
    orders = []
    if direction == "DOWN":
        # 急跌 → 走均值回歸：向下掛 Buy Limit
        for k in range(1, levels + 1):
            px = center - k * step
            if mode == "mean_revert":
                tp, sl = px + tp_steps*step, px - sl_steps*step
            else:
                tp, sl = px - tp_steps*step, px + sl_steps*step
            orders.append(dict(level=k, side="BUY", price=px, tp=tp, sl=sl,
                               filled=False, entry=None, exit=None,
                               units=units_each, closed=False))
    else:
        # 急漲 → 向上掛 Sell Limit
        for k in range(1, levels + 1):
            px = center + k * step
            if mode == "mean_revert":
                tp, sl = px - tp_steps*step, px + sl_steps*step
            else:
                tp, sl = px + tp_steps*step, px - sl_steps*step
            orders.append(dict(level=k, side="SELL", price=px, tp=tp, sl=sl,
                               filled=False, entry=None, exit=None,
                               units=units_each, closed=False))
    return orders

def filled_units_sum(grid):
    return sum(o['units'] for o in grid.grid_levels if o.get('filled') and not o.get('closed'))

def current_basket_max_loss_if_sl_hit(grid):
    """計算尚未平倉部位若全打到 SL 的最大可能損失（USD）"""
    loss = 0.0
    for o in grid.grid_levels:
        if o.get('filled') and not o.get('closed'):
            entry, sl, u = o['entry'], o['sl'], o['units']
            loss += abs(sl - entry) * u
    return loss

def backtest(df):
    df = df.copy()
    df['ATR'] = atr(df, n_atr)

    grid = GridState()
    cooldown = 0
    nav = initial_balance

    trades_log = []
    basket_log = []

    for i in range(len(df)):
        if i < max(n_atr + 2, impulse_lookback + 2):
            continue

        atr_now = df.iloc[i].ATR
        if pd.isna(atr_now) or atr_now <= 0:
            continue

        # === 1) 觸發新籃子 ===
        if not grid.active and cooldown == 0:
            delta = df['close'].iloc[i] - df['close'].iloc[i - impulse_lookback]
            if abs(delta) >= impulse_k_atr * atr_now:
                # 依 NAV 風險決定每筆 units（把風險分到預期會成交的格數）
                nav_for_risk = nav if compound else base_nav
                per_unit_risk = atr_now * sl_steps * grid_k_atr
                if per_unit_risk <= 0:
                    continue
                units_each = int((nav_for_risk * risk_pct) / (per_unit_risk * expected_fills_for_risk))
                units_each = max(1, min(units_each, max_units_each))
                if units_each <= 0:
                    continue

                grid.active = True
                grid.dir = "UP" if delta > 0 else "DOWN"
                grid.center_price = df['close'].iloc[i]
                step = grid_k_atr * atr_now
                grid.grid_levels = build_grid(grid.center_price, step, grid.dir, units_each)
                grid.open_time_idx = i
                grid.net_pnl = 0.0
                grid.cost = 0.0
                grid.units_each = units_each

        # === 2) 籃子運行 ===
        if grid.active:
            high_, low_ = df['high'].iloc[i], df['low'].iloc[i]

            # 控制總曝險上限：動態壓縮未成交格子的 units
            total_filled = filled_units_sum(grid)
            remaining_capacity = max(0, grid.max_total_units - total_filled)
            # 依尚未成交格數平均分配可用容量（簡化）
            not_filled_count = sum(1 for o in grid.grid_levels if not o['filled'])
            if not_filled_count > 0:
                cap_each = max(0, remaining_capacity // not_filled_count)
                for o in grid.grid_levels:
                    if not o['filled']:
                        o['units'] = min(o['units'], cap_each)

            # 嘗試成交未成交的限價單
            for o in grid.grid_levels:
                if o['filled'] or o['units'] <= 0:
                    continue
                px = o['price']
                if o['side'] == "BUY" and low_ <= px:
                    o['filled'] = True
                    o['entry'] = px
                    grid.cost += commission_per_trade
                elif o['side'] == "SELL" and high_ >= px:
                    o['filled'] = True
                    o['entry'] = px
                    grid.cost += commission_per_trade

            # 檢查 TP/SL（成交後才檢查）
            for o in grid.grid_levels:
                if not o['filled'] or o['closed']:
                    continue
                entry, tp, sl, u = o['entry'], o['tp'], o['sl'], o['units']
                hit_tp = (o['side'] == "BUY" and high_ >= tp) or (o['side'] == "SELL" and low_ <= tp)
                hit_sl = (o['side'] == "BUY" and low_ <= sl) or (o['side'] == "SELL" and high_ >= sl)

                if hit_tp or hit_sl:
                    exit_px = tp if hit_tp else sl
                    # 扣除簡化的 round-trip spread
                    raw_pnl = (exit_px - entry) * u if o['side'] == "BUY" else (entry - exit_px) * u
                    pnl = raw_pnl - spread
                    o['closed'], o['exit'] = True, exit_px
                    grid.net_pnl += pnl

            # ===== 籃子強平 / 時間止損 =====
            nav_for_risk = nav if compound else base_nav
            allow_loss = nav_for_risk * risk_pct * 1.2
            max_loss_now = current_basket_max_loss_if_sl_hit(grid)
            time_stop = (i - grid.open_time_idx) >= max_hold_bars
            force_exit = (grid.net_pnl <= -allow_loss) or (max_loss_now >= allow_loss * 1.5) or time_stop

            if force_exit:
                mid = (high_ + low_) / 2.0
                for o in grid.grid_levels:
                    if o.get('closed') or not o.get('filled'):
                        continue
                    u = o['units']
                    raw_pnl = (mid - o['entry']) * u if o['side'] == "BUY" else (o['entry'] - mid) * u
                    pnl = raw_pnl - spread
                    o['closed'], o['exit'] = True, mid
                    grid.net_pnl += pnl

                # 更新 NAV + 記錄
                nav += (grid.net_pnl - grid.cost)
                basket_log.append(dict(
                    open_idx=grid.open_time_idx,
                    close_idx=i,
                    pnl=grid.net_pnl - grid.cost,
                    nav=nav
                ))
                # 記錄單筆
                for o in grid.grid_levels:
                    if o.get('entry') is not None:
                        if o.get('exit') is None:
                            # 未平（理論上不會進來），略過或以 mid 結算
                            continue
                        raw = (o['exit'] - o['entry']) * o['units'] if o['side']=="BUY" else (o['entry'] - o['exit']) * o['units']
                        trades_log.append(dict(
                            time=df['time'].iloc[i],
                            side=o['side'],
                            entry=o['entry'],
                            exit=o['exit'],
                            units=o['units'],
                            pnl=raw - spread - commission_per_trade
                        ))
                grid.reset()
                cooldown = cooldown_bars
                continue

        if cooldown > 0 and not grid.active:
            cooldown -= 1

    trades_df = pd.DataFrame(trades_log)
    baskets_df = pd.DataFrame(basket_log)
    return trades_df, baskets_df

# ========= 直接執行 =========
if __name__ == "__main__":
    csv_path = "XAUUSD_5m.csv"
    df = pd.read_csv(csv_path)
    # 標準化欄位
    cols = {c.lower(): c for c in df.columns}
    df.rename(columns={
        cols.get('time','time'): 'time',
        cols.get('open','open'): 'open',
        cols.get('high','high'): 'high',
        cols.get('low','low'): 'low',
        cols.get('close','close'): 'close',
    }, inplace=True)
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')

    trades, baskets = backtest(df)

    # ===== 統計與輸出 =====
    start_nav = initial_balance
    end_nav = baskets['nav'].iloc[-1] if len(baskets) else start_nav

    # 籃子 / 單筆 勝率
    basket_winrate = (baskets['pnl'] > 0).mean() if len(baskets) else np.nan
    trade_winrate  = (trades['pnl']  > 0).mean() if len(trades)  else np.nan

    # NAV 曲線（以籃子為節點）
    nav_curve = pd.Series([start_nav] + baskets['nav'].tolist()) if len(baskets) else pd.Series([start_nav])
    # 最大回撤（基於籃子 NAV）
    roll_max = nav_curve.cummax()
    drawdown = nav_curve - roll_max
    max_dd = drawdown.min()

    print(f"[{PAIR} {TIMEFRAME}] Start NAV={start_nav:.2f}, End NAV={end_nav:.2f}")
    if len(baskets):
        print(f"Baskets={len(baskets)}, Basket WinRate={basket_winrate:.2%}, "
              f"Avg Basket PnL={baskets['pnl'].mean():.2f}")
    if len(trades):
        print(f"Trades={len(trades)}, Trade WinRate={trade_winrate:.2%}, "
              f"Avg Trade PnL={trades['pnl'].mean():.2f}")
    print(f"Max Drawdown (basket-level) = {max_dd:.2f}")

    # 保存
    trades.to_csv("grid_trades_nav_v1.2.csv", index=False)
    baskets.to_csv("grid_baskets_nav_v1.2.csv", index=False)
    pd.DataFrame({"nav": nav_curve, "drawdown": drawdown}).to_csv("nav_curve_v1.2.csv", index=False)
    print("Saved: grid_trades_nav_v1.2.csv, grid_baskets_nav_v1.2.csv, nav_curve_v1.2.csv")
