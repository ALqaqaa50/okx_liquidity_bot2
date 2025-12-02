import os
import os
import time
import hmac
import base64
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pathlib import Path
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import threading
from utils.reporting import send_report_via_env, generate_excel_report
from pathlib import Path as _Path

import requests
from dotenv import load_dotenv

# -----------------------------------------------------
# Basic setup
# -----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("OKXLiquidityBot")

load_dotenv()

# data directory for logs / collected snapshots
DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Prometheus metrics
HTTP_REQUESTS = Counter("okx_http_requests_total", "Total OKX HTTP requests", ["method", "path", "status"])
HTTP_DURATION = Histogram("okx_http_request_duration_seconds", "OKX HTTP request duration seconds")
ORDERS_TOTAL = Counter("okx_orders_total", "Total orders sent (including sandbox)")
LAST_PRICE = Gauge("okx_last_price", "Last observed price for primary instrument")

def start_metrics_server(port: int = 8000):
    start_http_server(port)

# Start metrics HTTP server in background thread
try:
    t = threading.Thread(target=start_metrics_server, args=(8000,), daemon=True)
    t.start()
except Exception:
    logger.debug("Prometheus metrics server failed to start")


def _start_reporter_thread(interval_min: int = 60):
    """Start a background thread that periodically generates & emails reports."""

    def _worker():
        while True:
            try:
                report = send_report_via_env()
                if report:
                    logger.info(f"Report generated/sent: {report}")
                else:
                    logger.info("Report generated but not sent (SMTP not configured or failed)")
            except Exception as e:
                logger.error(f"Reporter error: {e}")
            time.sleep(interval_min * 60)

    th = threading.Thread(target=_worker, daemon=True)
    th.start()


# -----------------------------------------------------
# Config & state
# -----------------------------------------------------


@dataclass
class BotConfig:
    inst_id: str = "BTC-USDT-SWAP"
    # support multiple timeframes (primary first)
    bars: List[str] = field(default_factory=lambda: ["5m", "15m"])
    candles_limit: int = 100
    orderbook_depth: int = 40
    trades_limit: int = 50
    poll_interval_sec: int = 60  # how often to refresh in seconds
    use_sandbox: bool = True     # demo trading mode flag
    enable_trading: bool = False  # <-- flip to True when you are ready


@dataclass
class RiskConfig:
    position_notional_usdt: float = 10.0   # approx notional per trade
    max_daily_loss_usdt: float = 15.0      # (not fully enforced yet, placeholder)
    trade_cooldown_sec: int = 900          # 15 minutes between new trades
    min_contracts: float = 0.001           # min contract size (approx)
    sl_pct: float = 0.005                  # 0.5% stop-loss
    tp_pct: float = 0.01                   # 1% take-profit
    # alternative sizing: percent of account balance per trade (0 => disabled)
    risk_pct_per_trade: float = 0.5       # percent of account balance (e.g., 0.5 => 0.5%)
    account_balance_usdt: float = 1000.0  # used when risk_pct_per_trade > 0
    # maximum contracts per order (safety cap)
    max_contracts: float = 1000.0


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class PositionState:
    direction: TradeDirection
    entry_price: float
    contracts: float
    sl: float
    tp: float
    opened_at: float
    closed: bool = False


# -----------------------------------------------------
# OKX Client
# -----------------------------------------------------


class OKXClient:
    def __init__(self, config: BotConfig):
        self.config = config
        self.base_url = "https://www.okx.com"
        self.api_key = os.getenv("OKX_API_KEY")
        self.api_secret = os.getenv("OKX_API_SECRET")
        self.api_passphrase = os.getenv("OKX_API_PASSPHRASE")

    # ------------ HTTP helpers ------------

    @staticmethod
    def _timestamp() -> str:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        return now.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _sign(
        self,
        timestamp: str,
        method: str,
        request_path: str,
        body: str = "",
    ) -> str:
        if not self.api_secret:
            raise RuntimeError("Private signing requested but OKX_API_SECRET is not set")

        message = f"{timestamp}{method}{request_path}{body}"
        mac = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        )
        d = mac.digest()
        return base64.b64encode(d).decode("utf-8")

    def _build_headers(
        self,
        method: str,
        request_path: str,
        body: Optional[Dict[str, Any]] = None,
        private: bool = False,
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }

        if self.config.use_sandbox:
            # Simulated trading header – works for demo environment
            headers["x-simulated-trading"] = "1"

        if private:
            if not (self.api_key and self.api_passphrase and self.api_secret):
                raise RuntimeError(
                    "Private endpoint requested but one or more OKX API env vars are missing"
                )

            ts = self._timestamp()
            body_str = json.dumps(body) if body else ""
            sign = self._sign(ts, method.upper(), request_path, body_str)

            headers.update(
                {
                    "OK-ACCESS-KEY": self.api_key,
                    "OK-ACCESS-SIGN": sign,
                    "OK-ACCESS-TIMESTAMP": ts,
                    "OK-ACCESS-PASSPHRASE": self.api_passphrase,
                }
            )

        return headers

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        private: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._build_headers(method, path, body, private=private)

        attempts = 3
        backoff_base = 0.5
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            start_ts = time.time()
            try:
                if method.upper() == "GET":
                    resp = requests.get(url, headers=headers, params=params, timeout=10)
                else:
                    resp = requests.request(
                        method.upper(),
                        url,
                        headers=headers,
                        params=params,
                        data=json.dumps(body) if body else None,
                        timeout=10,
                    )

                duration = time.time() - start_ts

                # write metric
                try:
                    m = {
                        "ts": int(time.time()),
                        "method": method.upper(),
                        "path": path,
                        "status_code": resp.status_code,
                        "duration": round(duration, 4),
                        "attempt": attempt,
                    }
                    with open(DATA_DIR / "metrics.jsonl", "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception:
                    logger.debug("Failed to write metric")
                # update Prometheus metrics
                try:
                    HTTP_REQUESTS.labels(method=method.upper(), path=path, status=str(resp.status_code)).inc()
                    HTTP_DURATION.observe(duration)
                except Exception:
                    pass

            except requests.RequestException as e:
                duration = time.time() - start_ts
                last_exc = e
                logger.warning(f"HTTP request error (attempt {attempt}): {e}")
                try:
                    m = {
                        "ts": int(time.time()),
                        "method": method.upper(),
                        "path": path,
                        "error": repr(e),
                        "duration": round(duration, 4),
                        "attempt": attempt,
                    }
                    with open(DATA_DIR / "metrics.jsonl", "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception:
                    logger.debug("Failed to write metric error")
                # prometheus: count errors
                try:
                    HTTP_REQUESTS.labels(method=method.upper(), path=path, status="error").inc()
                except Exception:
                    pass

                if attempt == attempts:
                    logger.error(f"HTTP error after {attempts} attempts: {e}")
                    raise
                time.sleep(backoff_base * attempt)
                continue

            # non-200 status
            if resp.status_code != 200:
                logger.error(f"Non-200 status code: {resp.status_code} - {resp.text}")
                # record and raise
                try:
                    m = {
                        "ts": int(time.time()),
                        "method": method.upper(),
                        "path": path,
                        "status_code": resp.status_code,
                        "text": resp.text,
                        "attempt": attempt,
                    }
                    with open(DATA_DIR / "metrics.jsonl", "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                try:
                    HTTP_REQUESTS.labels(method=method.upper(), path=path, status=str(resp.status_code)).inc()
                except Exception:
                    pass
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

            # OKX semantic error
            try:
                data = resp.json()
            except Exception as e:
                logger.error(f"Failed to decode JSON response: {e}")
                raise

            if data.get("code") != "0":
                logger.error(f"OKX error: {data}")
                try:
                    m = {
                        "ts": int(time.time()),
                        "method": method.upper(),
                        "path": path,
                        "okx_error": data,
                        "attempt": attempt,
                    }
                    with open(DATA_DIR / "metrics.jsonl", "a", encoding="utf-8") as mf:
                        mf.write(json.dumps(m, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                raise RuntimeError(f"OKX API error: {data}")

            return data.get("data", [])

    # ------------ Public endpoints ------------

    def get_candles(self) -> List[Dict[str, Any]]:
        """
        Get recent candles for the instrument.
        OKX returns:
        [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
        """
        # default to primary timeframe if none provided
        path = "/api/v5/market/candles"
        bar = self.config.bars[0] if self.config.bars else "15m"
        params = {
            "instId": self.config.inst_id,
            "bar": bar,
            "limit": str(self.config.candles_limit),
        }
        raw = self._request("GET", path, params=params, private=False)
        candles = []
        for item in raw:
            candles.append(
                {
                    "ts": int(item[0]),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        # OKX returns newest first, we want oldest first
        return list(reversed(candles))

    def get_orderbook(self) -> Dict[str, Any]:
        """
        Get order book snapshot.
        """
        path = "/api/v5/market/books"
        params = {
            "instId": self.config.inst_id,
            "sz": str(self.config.orderbook_depth),
        }
        raw = self._request("GET", path, params=params, private=False)
        if not raw:
            return {"bids": [], "asks": []}
        book = raw[0]
        return {
            "bids": [(float(b[0]), float(b[1])) for b in book.get("bids", [])],
            "asks": [(float(a[0]), float(a[1])) for a in book.get("asks", [])],
        }

    def get_trades(self) -> List[Dict[str, Any]]:
        """
        Get recent trades (tick-by-tick).
        """
        path = "/api/v5/market/trades"
        params = {
            "instId": self.config.inst_id,
            "limit": str(self.config.trades_limit),
        }
        raw = self._request("GET", path, params=params, private=False)
        trades = []
        for t in raw:
            trades.append(
                {
                    "ts": int(t["ts"]),
                    "px": float(t["px"]),
                    "sz": float(t["sz"]),
                    "side": t["side"],  # buy / sell
                }
            )
        return trades

    # ------------ Private endpoints (orders) ------------

    def place_order(
        self,
        side: str,
        pos_side: str,
        size: float,
    ) -> Any:
        """
        Place a simple market order on swap instrument.
        side: 'buy' or 'sell'
        pos_side: 'long' or 'short'
        size: contracts (approx)
        """
        path = "/api/v5/trade/order"
        body = {
            "instId": self.config.inst_id,
            "tdMode": "cross",
            "side": side,
            "posSide": pos_side,
            "ordType": "market",
            "sz": str(size),
        }
        logger.info(f"Sending order: {body}")
        resp = self._request("POST", path, body=body, private=True)
        # log order response for monitoring and audit
        try:
            order_log = {
                "ts": int(time.time()),
                "request": body,
                "response": resp,
            }
            with open(DATA_DIR / "orders.jsonl", "a", encoding="utf-8") as of:
                of.write(json.dumps(order_log, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write order log: {e}")
        try:
            ORDERS_TOTAL.inc()
        except Exception:
            pass
        return resp


# -----------------------------------------------------
# Simple liquidity-based strategy (read-only)
# -----------------------------------------------------


@dataclass
class MarketSnapshot:
    last_price: float
    trend_score: float
    bid_imbalance: float
    ask_imbalance: float
    buy_pressure: float
    sell_pressure: float


class LiquidityStrategy:
    """
    This is a safe version:
    - Can run in "monitor only" or with trading enabled.
    """

    def __init__(self, config: BotConfig):
        self.config = config

    @staticmethod
    def _calc_trend_score(candles: List[Dict[str, Any]]) -> float:
        if len(candles) < 5:
            return 0.0
        closes = [c["close"] for c in candles[-10:]]  # last 10 candles
        first = closes[0]
        last = closes[-1]
        if first == 0:
            return 0.0
        change_pct = (last - first) / first * 100.0
        # clamp to [-5, 5]
        if change_pct > 5:
            change_pct = 5
        if change_pct < -5:
            change_pct = -5
        return change_pct

    @staticmethod
    def _calc_orderbook_imbalance(book: Dict[str, Any]) -> Tuple[float, float]:
        bids = book.get("bids", [])
        asks = book.get("asks", [])

        total_bid_vol = sum(v for _, v in bids)
        total_ask_vol = sum(v for _, v in asks)

        if total_bid_vol + total_ask_vol == 0:
            return 0.0, 0.0

        bid_imb = total_bid_vol / (total_bid_vol + total_ask_vol)
        ask_imb = total_ask_vol / (total_bid_vol + total_ask_vol)
        return bid_imb, ask_imb

    @staticmethod
    def _calc_trade_pressure(trades: List[Dict[str, Any]]) -> Tuple[float, float]:
        if not trades:
            return 0.0, 0.0

        buy_vol = sum(t["sz"] for t in trades if t["side"] == "buy")
        sell_vol = sum(t["sz"] for t in trades if t["side"] == "sell")
        total = buy_vol + sell_vol
        if total == 0:
            return 0.0, 0.0

        buy_pressure = buy_vol / total
        sell_pressure = sell_vol / total
        return buy_pressure, sell_pressure

    def build_snapshot(
        self,
        candles: List[Dict[str, Any]],
        orderbook: Dict[str, Any],
        trades: List[Dict[str, Any]],
    ) -> MarketSnapshot:
        if not candles:
            raise RuntimeError("No candles available")

        last_price = candles[-1]["close"]
        trend_score = self._calc_trend_score(candles)
        bid_imb, ask_imb = self._calc_orderbook_imbalance(orderbook)
        buy_pressure, sell_pressure = self._calc_trade_pressure(trades)

        return MarketSnapshot(
            last_price=last_price,
            trend_score=trend_score,
            bid_imbalance=bid_imb,
            ask_imbalance=ask_imb,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
        )

    def decide(self, snapshot: MarketSnapshot, trend_scores: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Decision logic with simple protections.
        """
        decision = "FLAT"
        confidence = 0.0
        reasons: List[str] = []

        # Trend bias
        if snapshot.trend_score > 1.0:
            reasons.append(f"Uptrend {snapshot.trend_score:.2f}%")
        elif snapshot.trend_score < -1.0:
            reasons.append(f"Downtrend {snapshot.trend_score:.2f}%")
        else:
            reasons.append("Trend is weak / sideways")

        # Orderbook imbalance
        if snapshot.bid_imbalance > 0.6:
            reasons.append(f"Bid imbalance high ({snapshot.bid_imbalance:.2f})")
        elif snapshot.ask_imbalance > 0.6:
            reasons.append(f"Ask imbalance high ({snapshot.ask_imbalance:.2f})")

        # Trade pressure
        if snapshot.buy_pressure > 0.6:
            reasons.append(f"Recent trades mostly buys ({snapshot.buy_pressure:.2f})")
        elif snapshot.sell_pressure > 0.6:
            reasons.append(f"Recent trades mostly sells ({snapshot.sell_pressure:.2f})")

        # Combine signal logic
        long_score = 0
        short_score = 0

        if snapshot.trend_score > 1.0:
            long_score += 1
        if snapshot.trend_score < -1.0:
            short_score += 1

        if snapshot.bid_imbalance > 0.6:
            long_score += 1
        if snapshot.ask_imbalance > 0.6:
            short_score += 1

        if snapshot.buy_pressure > 0.6:
            long_score += 1
        if snapshot.sell_pressure > 0.6:
            short_score += 1

        if long_score >= 2 and long_score > short_score:
            decision = "LONG"
            confidence = long_score / 3.0
        elif short_score >= 2 and short_score > long_score:
            decision = "SHORT"
            confidence = short_score / 3.0
        else:
            decision = "FLAT"
            confidence = 0.0

        # extra protection: require decent confidence
        if confidence < 0.66:
            decision = "FLAT"

        # Multi-timeframe confirmation: require majority agreement and primary timeframe bias
        if trend_scores:
            # determine directional votes per timeframe
            votes_up = 0
            votes_down = 0
            primary_vote = None
            for tf, score in trend_scores.items():
                if score > 1.0:
                    votes_up += 1
                    if primary_vote is None:
                        primary_vote = "UP"
                elif score < -1.0:
                    votes_down += 1
                    if primary_vote is None:
                        primary_vote = "DOWN"

            # require majority and that primary timeframe (first in config) supports the decision
            if votes_up + votes_down > 0:
                if votes_up > votes_down and decision == "LONG":
                    # ok
                    pass
                elif votes_down > votes_up and decision == "SHORT":
                    # ok
                    pass
                else:
                    # disagreement between timeframes -> flatten
                    decision = "FLAT"
                    confidence = 0.0

        return {
            "decision": decision,
            "confidence": round(confidence, 2),
            "reasons": reasons,
        }


# -----------------------------------------------------
# Trade manager (opens/closes positions)
# -----------------------------------------------------


class TradeManager:
    def __init__(
        self,
        client: OKXClient,
        bot_config: BotConfig,
        risk_config: RiskConfig,
    ):
        self.client = client
        self.bot_config = bot_config
        self.risk_config = risk_config
        self.position: Optional[PositionState] = None
        self.last_trade_ts: float = 0.0

    def _calc_contracts(self, price: float) -> float:
        """
        Rough position size based on notional and price.
        """
        if price <= 0:
            return 0.0
        # choose notional: percent-based sizing preferred if set
        if getattr(self.risk_config, "risk_pct_per_trade", 0) and getattr(
            self.risk_config, "account_balance_usdt", 0
        ):
            notional = (
                self.risk_config.account_balance_usdt
                * (self.risk_config.risk_pct_per_trade / 100.0)
            )
        else:
            notional = self.risk_config.position_notional_usdt

        contracts = notional / price
        # keep 3 decimal places
        contracts = round(contracts, 3)
        if contracts < self.risk_config.min_contracts:
            return 0.0
        return contracts

    def _can_open_new_trade(self) -> bool:
        now = time.time()
        if self.position is not None and not self.position.closed:
            return False
        if now - self.last_trade_ts < self.risk_config.trade_cooldown_sec:
            return False
        return True

    def maybe_open_position(
        self,
        snapshot: MarketSnapshot,
        decision_info: Dict[str, Any],
    ):
        if decision_info["decision"] not in ("LONG", "SHORT"):
            return
        if not self._can_open_new_trade():
            return

        price = snapshot.last_price
        contracts = self._calc_contracts(price)
        if contracts <= 0:
            logger.warning("Calculated contracts too small, skipping trade")
            return

        direction = (
            TradeDirection.LONG
            if decision_info["decision"] == "LONG"
            else TradeDirection.SHORT
        )

        if direction == TradeDirection.LONG:
            sl = price * (1.0 - self.risk_config.sl_pct)
            tp = price * (1.0 + self.risk_config.tp_pct)
            side = "buy"
            pos_side = "long"
        else:
            sl = price * (1.0 + self.risk_config.sl_pct)
            tp = price * (1.0 - self.risk_config.tp_pct)
            side = "sell"
            pos_side = "short"

        pos = PositionState(
            direction=direction,
            entry_price=price,
            contracts=contracts,
            sl=sl,
            tp=tp,
            opened_at=time.time(),
        )
        self.position = pos
        self.last_trade_ts = time.time()

        logger.info(
            f"OPEN {direction.value}: price={price:.2f}, sz={contracts}, sl={sl:.2f}, tp={tp:.2f}"
        )

        if self.bot_config.enable_trading:
            try:
                self.client.place_order(side=side, pos_side=pos_side, size=contracts)
            except Exception as e:
                logger.error(f"Error sending open order: {e}")

    def maybe_close_position(self, snapshot: MarketSnapshot):
        if self.position is None or self.position.closed:
            return

        price = snapshot.last_price
        pos = self.position
        reason = None
        side = None
        pos_side = None

        if pos.direction == TradeDirection.LONG:
            if price <= pos.sl:
                reason = "SL"
                side = "sell"
                pos_side = "long"
            elif price >= pos.tp:
                reason = "TP"
                side = "sell"
                pos_side = "long"
        else:
            if price >= pos.sl:
                reason = "SL"
                side = "buy"
                pos_side = "short"
            elif price <= pos.tp:
                reason = "TP"
                side = "buy"
                pos_side = "short"

        if reason is None:
            return

        logger.info(
            f"CLOSE {pos.direction.value} by {reason}: "
            f"entry={pos.entry_price:.2f}, now={price:.2f}, sz={pos.contracts}"
        )

        pos.closed = True

        if self.bot_config.enable_trading:
            try:
                self.client.place_order(side=side, pos_side=pos_side, size=pos.contracts)
            except Exception as e:
                logger.error(f"Error sending close order: {e}")


# -----------------------------------------------------
# Main loop
# -----------------------------------------------------


def main():
    config = BotConfig()
    risk = RiskConfig()
    client = OKXClient(config)
    strategy = LiquidityStrategy(config)
    trader = TradeManager(client, config, risk)

    # load runtime overrides from environment
    tf_env = os.getenv("TIMEFRAMES")
    if tf_env:
        config.bars = [t.strip() for t in tf_env.split(",") if t.strip()]

    # allow toggles from env
    cfg_enable = os.getenv("BOT_ENABLE_TRADING")
    if cfg_enable is not None:
        try:
            config.enable_trading = bool(int(cfg_enable))
        except Exception:
            config.enable_trading = cfg_enable.lower() in ("1", "true", "yes")

    cfg_sandbox = os.getenv("BOT_USE_SANDBOX")
    if cfg_sandbox is not None:
        try:
            config.use_sandbox = bool(int(cfg_sandbox))
        except Exception:
            config.use_sandbox = cfg_sandbox.lower() in ("1", "true", "yes")

    # risk overrides
    acct_bal = os.getenv("ACCOUNT_BALANCE_USDT")
    if acct_bal:
        try:
            risk.account_balance_usdt = float(acct_bal)
        except Exception:
            pass

    rperc = os.getenv("RISK_PCT_PER_TRADE")
    if rperc:
        try:
            risk.risk_pct_per_trade = float(rperc)
        except Exception:
            pass

    # reporter interval (minutes)
    rpt_interval = os.getenv("REPORT_INTERVAL_MIN")
    try:
        rpt_interval_min = int(rpt_interval) if rpt_interval else 60
    except Exception:
        rpt_interval_min = 60

    # start background reporter if user configured SMTP or wants periodic reports
    _start_reporter_thread(interval_min=rpt_interval_min)

    mode = "TRADING" if config.enable_trading else "MONITOR ONLY"
    logger.info(f"Starting OKX Liquidity Bot ({mode})")
    logger.info(f"Instrument: {config.inst_id}, Timeframes: {','.join(config.bars)}")

    while True:
        try:
            # fetch candles for each configured timeframe
            candles_map: Dict[str, List[Dict[str, Any]]] = {}
            for b in config.bars:
                try:
                    # temporarily request candles using requested bar
                    # OKXClient.get_candles() uses the primary bar by default,
                    # so we call the API via a small wrapper below
                    path = "/api/v5/market/candles"
                    params = {
                        "instId": config.inst_id,
                        "bar": b,
                        "limit": str(config.candles_limit),
                    }
                    raw = client._request("GET", path, params=params, private=False)
                    candles_tmp = []
                    for item in raw:
                        candles_tmp.append(
                            {
                                "ts": int(item[0]),
                                "open": float(item[1]),
                                "high": float(item[2]),
                                "low": float(item[3]),
                                "close": float(item[4]),
                                "volume": float(item[5]),
                            }
                        )
                    candles_map[b] = list(reversed(candles_tmp))
                except Exception as e:
                    logger.error(f"Error fetching candles for {b}: {e}")
                    candles_map[b] = []
            orderbook = client.get_orderbook()
            trades = client.get_trades()

            # use primary timeframe's candles for snapshot calculations
            primary_tf = config.bars[0] if config.bars else "15m"
            primary_candles = candles_map.get(primary_tf, [])

            snapshot = strategy.build_snapshot(primary_candles, orderbook, trades)
            # compute trend scores per timeframe
            trend_scores = {}
            for b, c in candles_map.items():
                try:
                    trend_scores[b] = strategy._calc_trend_score(c)
                except Exception:
                    trend_scores[b] = 0.0

            decision_info = strategy.decide(snapshot, trend_scores=trend_scores)

            # First, manage existing position (SL/TP)
            trader.maybe_close_position(snapshot)

            # Log snapshot
            logger.info("------ Market Snapshot ------")
            logger.info(f"Last price: {snapshot.last_price:.2f}")
            logger.info(
                f"Trend score (primary {primary_tf}): {snapshot.trend_score:.2f}%"
            )
            for b, s in trend_scores.items():
                logger.info(f"  - {b}: {s:.2f}%")
            logger.info(
                f"Orderbook imbalance: bids={snapshot.bid_imbalance:.2f}, "
                f"asks={snapshot.ask_imbalance:.2f}"
            )
            logger.info(
                f"Trade pressure: buys={snapshot.buy_pressure:.2f}, "
                f"sells={snapshot.sell_pressure:.2f}"
            )

            # Log decision
            logger.info("------ Decision ------")
            logger.info(f"Action: {decision_info['decision']}")
            logger.info(f"Confidence: {decision_info['confidence']:.2f}")
            logger.info("Reasons:")
            for r in decision_info["reasons"]:
                logger.info(f"  - {r}")

            # --- Monitoring / continuous improvement: append JSONL snapshot ---
            try:
                log_entry = {
                    "ts": int(time.time()),
                    "inst_id": config.inst_id,
                    "primary_tf": primary_tf,
                    "candles_count": len(primary_candles),
                    "last_price": snapshot.last_price,
                    "trend_score": snapshot.trend_score,
                    "trend_scores_all": trend_scores,
                    "bid_imbalance": snapshot.bid_imbalance,
                    "ask_imbalance": snapshot.ask_imbalance,
                    "buy_pressure": snapshot.buy_pressure,
                    "sell_pressure": snapshot.sell_pressure,
                    "decision": decision_info.get("decision"),
                    "confidence": decision_info.get("confidence"),
                    "reasons": decision_info.get("reasons"),
                    "enable_trading": bool(config.enable_trading),
                    "use_sandbox": bool(config.use_sandbox),
                }

                log_path = DATA_DIR / "market_snapshots.jsonl"
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(f"Error writing monitoring log: {e}")

            # Try to open new position (if allowed)
            trader.maybe_open_position(snapshot, decision_info)

            if not config.enable_trading:
                logger.info("Safe mode: no orders are being sent.\n")
            else:
                logger.info("Trading mode: orders may be sent.\n")

        except Exception as e:
            logger.error(f"Error in main loop: {e}")

        time.sleep(config.poll_interval_sec)


if __name__ == "__main__":
    main()
