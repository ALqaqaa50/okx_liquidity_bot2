import json
from pathlib import Path
from dataclasses import dataclass


DATA = Path(__file__).resolve().parents[1] / "data" / "market_snapshots.jsonl"


@dataclass
class Trade:
    direction: str
    entry_price: float
    exit_price: float
    entry_ts: int
    exit_ts: int
    pnl: float


def run_backtest():
    if not DATA.exists():
        print("No market_snapshots.jsonl found in data/")
        return

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--slippage_pct", type=float, default=0.05, help="Slippage percent per trade (e.g. 0.05 => 0.05%)")
    args = parser.parse_args()

    balance = 1000.0
    risk_pct = 0.5
    slippage_pct = args.slippage_pct / 100.0
    open_pos = None
    trades = []

    with open(DATA, "r", encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh if l.strip()]

    for row in lines:
        price = row.get("last_price")
        decision = row.get("decision")
        ts = row.get("ts")

        # if we have open pos, check SL/TP
        if open_pos:
            if open_pos["direction"] == "LONG":
                if price <= open_pos["sl"] or price >= open_pos["tp"]:
                    exit_price = price
                    pnl = (exit_price - open_pos["entry_price"]) * open_pos["contracts"]
                    balance += pnl
                    trades.append(
                        Trade(
                            direction="LONG",
                            entry_price=open_pos["entry_price"],
                            exit_price=exit_price,
                            entry_ts=open_pos["entry_ts"],
                            exit_ts=ts,
                            pnl=pnl,
                        )
                    )
                    open_pos = None
            else:
                # SHORT
                if price >= open_pos["sl"] or price <= open_pos["tp"]:
                    exit_price = price
                    pnl = (open_pos["entry_price"] - exit_price) * open_pos["contracts"]
                    balance += pnl
                    trades.append(
                        Trade(
                            direction="SHORT",
                            entry_price=open_pos["entry_price"],
                            exit_price=exit_price,
                            entry_ts=open_pos["entry_ts"],
                            exit_ts=ts,
                            pnl=pnl,
                        )
                    )
                    open_pos = None

        # open new if signalled and no open
        if not open_pos and decision in ("LONG", "SHORT"):
            # simplified sizing
            notional = balance * (risk_pct / 100.0)
            contracts = round(notional / price, 3) if price and price > 0 else 0
            if contracts <= 0:
                continue
            if decision == "LONG":
                sl = price * (1.0 - 0.005)
                tp = price * (1.0 + 0.01)
            else:
                sl = price * (1.0 + 0.005)
                tp = price * (1.0 - 0.01)

            open_pos = {
                "direction": decision,
                "entry_price": price,
                "contracts": contracts,
                "sl": sl,
                "tp": tp,
                "entry_ts": ts,
            }
            # simulate immediate fill with slippage
            if slippage_pct > 0:
                if decision == "LONG":
                    open_pos["entry_price"] = price * (1.0 + slippage_pct)
                else:
                    open_pos["entry_price"] = price * (1.0 - slippage_pct)

    # summary
    pnl_total = sum(t.pnl for t in trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)
    print(f"Trades: {len(trades)}, Wins: {wins}, Losses: {losses}, PnL: {pnl_total:.2f}, Final balance: {balance:.2f}")


if __name__ == "__main__":
    run_backtest()
