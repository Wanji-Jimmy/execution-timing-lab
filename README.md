# OxHack: Stock Order Execution

A Python study of stock-order timing, started at OxHack and extended afterwards. It compares equal release (TWAP), a historical volume schedule, fixed early release, and feedback based on past intraday volume. An order can fail because total capacity is too small or because it was released too late.

The sample covers EBAY and LULU from 4 January to 2 March 2021: 40 sessions per stock, with 20 for history and 20 evaluated using a strictly prior rolling window.

Every USD 1m order completed in the model. Feedback never changed the historical schedule. At the 1% participation cap, the historical schedule's median capacity limit was **29-31% lower with exchange-only volume** than with total reported volume. These limits use realised future volume. They are retrospective diagnostics added after the first run, not forecasts available before trading.

![Capacity limits and benchmark tracking](results/equities/equity_capacity.png)

Browse the [640 comparison rows](results/equities/equity_daily.csv), [metadata](results/equities/equity_results.json) and [methods](docs/STUDY.md) without installing anything. These are historical price and capacity proxies, not actual fills or PnL. Spreads, fees, queue position and own market impact are omitted.

To run the tests, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
```

Tests need no market data. To recompute, first follow [data setup](data/README.md) to obtain the 80 stock files; that page gives the run command. Raw data is not redistributed.

Read the code in this order: [stock inputs and evaluation](study/equity.py), [release and replay](study/adaptive.py), then [stock tests](tests/test_equity.py). [analysis.py](study/analysis.py) contains optional baseline and diagnostic helpers. The [protocol](docs/protocol.md) records design decisions; the [archive branch](https://github.com/Wanji-Jimmy/execution-timing-lab/tree/archive/before-2026-10-26-simplification) retains the earlier BTC/ETH experiment.
