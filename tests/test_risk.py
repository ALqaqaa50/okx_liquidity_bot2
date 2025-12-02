from main import RiskConfig, TradeManager, BotConfig


def test_risk_percent_sizing():
    bot_conf = BotConfig()
    risk = RiskConfig()
    risk.account_balance_usdt = 2000.0
    risk.risk_pct_per_trade = 1.0
    tm = TradeManager(None, bot_conf, risk)
    contracts = tm._calc_contracts(100.0)
    # 1% of 2000 = 20 notional / price 100 => 0.2 contracts
    assert contracts >= 0
