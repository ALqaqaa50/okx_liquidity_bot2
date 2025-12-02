**Blueprint — منظومة تداول 4 طبقات (MVP → Production)**

ملخص سريع
- الهدف: منظومة تداول متعددة الطبقات (Price Action, Order Flow, Algorithms, Risk) قابلة للتوسع، تبدأ كـ MVP تعمل على `BTC-USDT` مع تايمفريمات `5m` و`15m`.

مكونات النظام (عالي المستوى)
- Data Layer: مصادر السوق (OKX REST for candles, orderbook, trades). لاحقاً: مزوّد Heatmap/CVD.
- Layer 1 — Price Action: تحليل Market Structure، Order Blocks، FVG، Support/Resistance.
- Layer 2 — Order Flow: Orderbook imbalance, trade ticks aggregation, simple CVD.
- Layer 3 — Algorithms: rule-based signal combiner (MVP)، لاحقاً ML models وexecution engine.
- Layer 4 — Risk Management: position sizing (%-based أو fixed notional)، daily loss limits، cooldowns.

قواعد MVP (واضحة وقابلة للقياس)
- Data: استخدم تايمفريمين: Primary=`5m`, Secondary=`15m`.
- Signal (Entry):
  - Layer1: primary trend_score > +1% => bullish; < -1% => bearish.
  - Layer2: orderbook imbalance > 0.6 on bids => bias buy; >0.6 on asks => bias sell.
  - Layer3 combine: at least 2/3 internal scores (trend, orderbook, trade pressure) نشانة.
  - Multi-timeframe confirmation: direction must be supported by majority of configured timeframes (primary must agree).
- Execution: market order on swap instrument, simple TP/SL (configurable percentages).

Position sizing (MVP)
- Default: risk_pct_per_trade (مثال 0.5 => 0.5% من الحساب). إذا =0 يستخدم fixed notional.
- حساب العقود = notional / price (مع مراعاة min_contracts).

Safety & Deployment
- Always start sandbox/paper until backtest & live-paper validated.
- Keep credentials in secrets manager for production.
- Observe latency & order rejections; implement retries and idempotency.

Roadmap (المراحل)
1) MVP (2-4 أسابيع):
   - Implement data ingestion (candles, orderbook, trades).
   - Implement 2-layer signal (PA + simple OF) وmulti-tf confirmation.
   - Implement risk sizing %-based + cooldowns + logging.
   - Paper-run for 2-4 أسابيع، جمع بيانات.
2) Backtest & Analytics (2 أسابيع):
   - Replay historical ticks, metrics: P&L, drawdown, win-rate.
3) Execution & Reliability (1-2 أسابيع):
   - Improve execution engine، handle partial fills، retries، order monitoring.
4) Advanced (4+ أسابيع):
   - Add Heatmap/CVD data sources.
   - Build ML signals (CNN/LSTM) على بيانات التاريخية المجموعة.
   - Implement feature store + model training pipeline.

Deliverables الآن
- ملف مشروع أساس (موجود) مع multi-timeframe، percent sizing.
- وثيقة Blueprint (هذا الملف).
- إمكانية توسيع: مثال endpoints، format logs، وكيفية إضافة Heatmap.

هل تريد أن أضع الآن: (أ) ملف `design_diagram.png` تقريبي، (ب) جدول زمني مفصّل بالمهام، أو (ج) أبدأ في كتابة اختبارات وحدات وبيانات backtest؟

CI/ML Pipeline (مُقترح)
- جمع الميزات: `data/market_snapshots.jsonl` هو مصدر الحقائق الأولي—نحو التحويل لملف CSV أو Feature Store لكل نافذة زمنية.
- Feature engineering: تحويل `trend_scores`, `imbalance`, `trade_pressure`, TF differences إلى صفات قابلة للنمذجة.
- تدريب/اختبار: حفظ مجموعات train/val/test زمنيًا، استخدام طرق بسيطة أولًا (XGBoost, RandomForest) ثم الانتقال لـ CNN/LSTM على نوافذ الشموع.
- تقييم: طُرُز scoring على Sharpe, max drawdown, hit-rate, avg PnL per trade.
- نشر: حفظ النموذج كملف، تشغيله داخل `Layer 3` عبر واجهة بسيطة `predict(features) -> score`.

التالى (مقترح تنفيذي فوري)
- تحسين المراقبة: مقاييس latency, HTTP errors, order fills (تم إضافتها).
- إضافة backtest محلي يُعيد تشغيل لقطات السوق ويحسب PnL (تمت إضافة `scripts/backtest.py`).
- هيكلة Heatmap Provider كـ scaffold (ملف `data_providers/heatmap.py`) لاستخدام مزود لاحقًا.

*** End Patch
