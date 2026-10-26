# OxHack: Stock Order Execution

A Python study of stock-order timing, started at OxHack and extended afterwards. It compares equal release (TWAP), a historical volume schedule, fixed early release, and feedback based on past intraday volume.

The current sample uses **AAPL, MSFT, AMZN and INTC, April-June 2021**: 63 sessions per stock, 20 for warm-up and 43 for evaluation. Every forecast uses only the previous 20 valid sessions. The 252 source files contain 98,280 selected minute rows.

At USD 1m, all four rules completed in the model. The historical schedule tracked the selected-window VWAP more closely than TWAP for each stock; feedback never changed that schedule. At a 1% cap, the historical schedule's median capacity limit was **30-46% lower with exchange-only volume** than with total reported volume. These limits use realised future volume, so they are retrospective diagnostics, not pre-trade size forecasts.

![Capacity limits and benchmark tracking](results/equities/equity_capacity.png)

Browse the [2,752 comparison rows](results/equities/equity_daily.csv), [metadata](results/equities/equity_results.json) and [methods](docs/STUDY.md). These are price and capacity proxies, not actual fills or P&L. Spreads, fees, queues and own impact are omitted.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

Tests need no market data. To recompute, follow the [data setup](data/README.md); raw files are not redistributed. Read [stock evaluation](study/equity.py), [release and replay](study/adaptive.py), then [stock tests](tests/test_equity.py).

The [previous pilot](https://github.com/Wanji-Jimmy/execution-timing-lab/tree/archive/before-2026-10-26-independent-data) used EBAY/LULU. This replacement shares no stocks, dates or raw files with the market-making study, while retaining AlgoSeek as provider. The [protocol](docs/protocol.md#independent-equity-sample---26-october-2026) records the selection before new outcomes were calculated.
