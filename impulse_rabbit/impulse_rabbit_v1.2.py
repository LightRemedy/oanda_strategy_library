#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime Impulse-Grid Strategy for XAU_USD (OANDA v20)
JumpingLoopRabbit — v1.2 (safer sizing & risk controls)

- Impulse detector on M5 close
- ATR-adaptive grid (mean-reversion by default)
- Position sizing by NAV with risk splitting across expected fills
- Basket-level exposure caps and force-exit on loss/time
"""

import os, time, uuid
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.pricing as pricing

# ====== 基本設定 ======
INSTRUMENT  = "XAU_USD"
GRANULARITY = "M5"
ENV         = os.getenv("OANDA_ENV", "practice")  # practice / live
API_KEY = os.getenv("OANDA_API_KEY", "b63ac389c9a98c2b06c066f1fc1592ee-845908de52c2081905d4b4b6e9a486f9")
ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-003-30564326-002")
assert API_KEY and ACCOUNT_ID, "請先設定 OANDA_API_KEY / OANDA_ACCOUNT_ID"

# ====== 策略參數（與 v1.2 對齊）======
risk_pct = 0.01              # 每個籃子允許的最大風險 (相對 NAV 或 base_nav)
compound = True             # True=使用當前NAV複利；False=固定 base_nav 做 sizing
base_nav = 10_000.0          # compound=False 時使用
expected_fills_for_risk = 3  # 風險分攤到預期成交的格數（保守可用 levels）
max_units_each  = 500        # 單筆上限（避免複利爆掉）
max_total_units = 1_500      # 單籃「已成交」總上限

n_atr = 14
impulse_lookback = 6
impulse_k_atr = 1.8

grid_k_atr = 0.6
levels = 6
tp_steps = 1.0
sl_steps = 2.0
mode = "mean_revert"         # "mean_revert" or "trend_follow"

max_hold_minutes = 6 * 60    # 籃子最長持倉
POLL_SEC = 10
PRINT_EVERY = 6              # 心跳輸出頻率

# ====== 工具 ======
def now_utc():
    return datetime.now(timezone.utc)

def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def atr_from_df(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def fetch_candles(api, count=200) -> pd.DataFrame:
    r = instruments.InstrumentsCandles(
        instrument=INSTRUMENT,
        params={"granularity": GRANULARITY, "price": "M", "count": count}
    )
    data = api.request(r)
    rows = []
    for c in data["candles"]:
        mid = c["mid"]
        rows.append({
            "time":  pd.to_datetime(c["time"], utc=True),
            "open":  float(mid["o"]),
            "high":  float(mid["h"]),
            "low":   float(mid["l"]),
            "close": float(mid["c"]),
            "complete": c["complete"]
        })
    return pd.DataFrame(rows)

def last_complete_bar(df: pd.DataFrame):
    cdf = df[df["complete"]]
    return cdf.iloc[-1] if len(cdf) else None

def get_nav(api) -> float:
    r = accounts.AccountDetails(ACCOUNT_ID)
    acc = api.request(r)
    return float(acc["account"]["NAV"])

def list_open_trades_by_tag(api, tag: str):
    """回傳該 tag 的開倉 trades（含 units, current price, unrealizedPL）。"""
    lt = trades.TradesList(ACCOUNT_ID, params={"state": "OPEN"})
    data = api.request(lt)
    result = []
    for t in data.get("trades", []):
        ce = t.get("clientExtensions") or {}
        if ce.get("tag") == tag:
            result.append(t)
    return result

def list_pending_orders_by_tag(api, tag: str):
    import oandapyV20.endpoints.orders as ORD
    lo = ORD.OrdersPending(ACCOUNT_ID)
    data = api.request(lo)
    result = []
    for od in data.get("orders", []):
        if (od.get("clientExtensions") or {}).get("tag") == tag:
            result.append(od)
    return result

def cancel_orders_by_tag(api, tag: str):
    import oandapyV20.endpoints.orders as ORD
    for od in list_pending_orders_by_tag(api, tag):
        try:
            api.request(ORD.OrderCancel(ACCOUNT_ID, orderID=od["id"]))
        except Exception as e:
            print("[WARN] cancel fail:", e)

def close_trades_by_tag(api, tag: str):
    """市價平掉該 tag 的所有開倉。"""
    lst = list_open_trades_by_tag(api, tag)
    for t in lst:
        trid = t["id"]
        units = -int(t["currentUnits"])
        if units == 0: 
            continue
        try:
            api.request(trades.TradeClose(ACCOUNT_ID, tradeID=trid, data={"units": str(abs(units))}))
        except Exception as e:
            print("[WARN] close trade fail:", e)

def basket_unrealized_pl(api, tag: str) -> float:
    """彙總該籃子目前未實現損益（USD）。"""
    s = 0.0
    for t in list_open_trades_by_tag(api, tag):
        s += float(t.get("unrealizedPL", "0"))
    return s

def place_limit(api, side: str, px: float, tp: float, sl: float, units: int, tag: str):
    units_signed = units if side == "buy" else -units
    payload = {
        "order": {
            "type": "LIMIT", "instrument": INSTRUMENT,
            "units": str(units_signed), "price": f"{px:.3f}",
            "timeInForce": "GTC", "positionFill": "DEFAULT",
            "takeProfitOnFill": {"price": f"{tp:.3f}"},
            "stopLossOnFill":   {"price": f"{sl:.3f}"},
            "clientExtensions": {"tag": tag, "id": str(uuid.uuid4())}
        }
    }
    r = orders.OrderCreate(ACCOUNT_ID, data=payload)
    return api.request(r)

def build_grid_units_each(atr_now: float, nav: float) -> int:
    """依 v1.2：把 per-basket risk 分攤到預期成交格數，並套用單筆上限。"""
    nav_for_risk = nav if compound else base_nav
    per_unit_risk = atr_now * sl_steps * grid_k_atr
    if per_unit_risk <= 0:
        return 0
    u = int((nav_for_risk * risk_pct) / (per_unit_risk * max(1, expected_fills_for_risk)))
    return max(1, min(u, max_units_each))

def planned_grid(center: float, step: float, direction: str, units_each: int, dp=3):
    """產生網格（僅價格與每筆 units，每筆帶自己的 TP/SL）。"""
    def rp(x): return round(x, dp)
    out = []
    if direction == "DOWN":
        for k in range(1, levels + 1):
            px = rp(center - k * step)
            if mode == "mean_revert":
                tp, sl = rp(px + tp_steps*step), rp(px - sl_steps*step)
            else:
                tp, sl = rp(px - tp_steps*step), rp(px + sl_steps*step)
            out.append(dict(side="buy", price=px, tp=tp, sl=sl, units=units_each))
    else:
        for k in range(1, levels + 1):
            px = rp(center + k * step)
            if mode == "mean_revert":
                tp, sl = rp(px - tp_steps*step), rp(px + sl_steps*step)
            else:
                tp, sl = rp(px + tp_steps*step), rp(px - sl_steps*step)
            out.append(dict(side="sell", price=px, tp=tp, sl=sl, units=units_each))
    return out

# ====== 主流程 ======
def main():
    api = API(access_token=API_KEY, environment=ENV)
    last_bar_time = None
    active_tag = None
    basket_open = None

    poll_i = 0

    while True:
        try:
            df = fetch_candles(api, count=200)
            bar = last_complete_bar(df)
            if bar is None:
                time.sleep(POLL_SEC); continue

            # 每根收盤只判斷一次
            if (last_bar_time is None) or (bar["time"] > last_bar_time):
                last_bar_time = bar["time"]

                df["ATR"] = atr_from_df(df, n_atr)
                atr_now = df.iloc[-1]["ATR"]
                if pd.isna(atr_now) or atr_now <= 0:
                    time.sleep(POLL_SEC); continue

                close_now = df.iloc[-1]["close"]
                close_prev = df.iloc[-1 - impulse_lookback]["close"]
                delta = close_now - close_prev

                # 1) 現有籃子：到期或強平？
                if active_tag and basket_open:
                    # 時間到期
                    if now_utc() - basket_open >= timedelta(minutes=max_hold_minutes):
                        print(f"[Basket Expire] tag={active_tag} -> cancel & close")
                        cancel_orders_by_tag(api, active_tag)
                        close_trades_by_tag(api, active_tag)
                        active_tag = None
                        basket_open = None
                    else:
                        # 強平：未實現損益低於允許值
                        nav_now = get_nav(api)
                        nav_for_risk = nav_now if compound else base_nav
                        allow_loss = nav_for_risk * risk_pct * 1.2
                        uPL = basket_unrealized_pl(api, active_tag)  # 負值代表虧損
                        if uPL <= -allow_loss:
                            print(f"[Basket Stop] tag={active_tag} uPL={uPL:.2f} <= -{allow_loss:.2f}")
                            cancel_orders_by_tag(api, active_tag)
                            close_trades_by_tag(api, active_tag)
                            active_tag = None
                            basket_open = None

                # 2) 沒有籃子時才考慮新觸發
                if not active_tag:
                    if abs(delta) >= impulse_k_atr * atr_now:
                        direction = "UP" if delta > 0 else "DOWN"
                        nav_now = get_nav(api)
                        units_each = build_grid_units_each(atr_now, nav_now)
                        if units_each <= 0:
                            time.sleep(POLL_SEC); continue

                        step = grid_k_atr * atr_now
                        grid = planned_grid(center=close_now, step=step, direction=direction, units_each=units_each)

                        # 曝險上限：規劃的總 units 不能超過 max_total_units（未來成交的保險）
                        planned_total = sum(g["units"] for g in grid)
                        if planned_total > max_total_units:
                            # 平均縮水
                            scale = max_total_units / planned_total
                            for g in grid:
                                g["units"] = max(1, int(g["units"] * scale))

                        tag = f"ImpulseRabbit:{uuid.uuid4().hex[:8]}"
                        print(f"[Trigger] {iso_z(last_bar_time)} dir={direction} NAV={nav_now:.2f} "
                              f"units_each={grid[0]['units']} tag={tag}")

                        # 下單
                        for g in grid:
                            try:
                                place_limit(api, g["side"], g["price"], g["tp"], g["sl"], g["units"], tag)
                            except Exception as e:
                                print("[WARN] order fail:", e)

                        active_tag = tag
                        basket_open = now_utc()

            # 心跳輸出
            poll_i += 1
            if poll_i % PRINT_EVERY == 0:
                hb = f"[Heartbeat] {iso_z(now_utc())} lastbar={iso_z(bar['time'])} close={bar['close']:.3f} active={active_tag}"
                if active_tag:
                    uPL = basket_unrealized_pl(api, active_tag)
                    hb += f" uPL={uPL:.2f}"
                print(hb)

            time.sleep(POLL_SEC)

        except KeyboardInterrupt:
            print("Interrupted. Cleaning up active basket...")
            if active_tag:
                cancel_orders_by_tag(api, active_tag)
                close_trades_by_tag(api, active_tag)
            break
        except Exception as e:
            print("[ERROR]", e)
            time.sleep(3)

if __name__ == "__main__":
    main()
