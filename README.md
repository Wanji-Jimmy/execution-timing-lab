# OxHack: Stock Order Execution

A Python study of when to release a large stock order, started at OxHack and extended afterwards. It compares equal release (TWAP), a historical volume schedule, fixed early release, and feedback based on past intraday volume. The question is whether an order fails because total capacity is too small or because it was released too late.

The sample is EBAY and LULU minute data from 4 January to 2 March 2021: 40 sessions per stock, with the first 20 for history and the next 20 evaluated using a strictly prior rolling window.

Every USD 1m order completed in the model. Feedback never changed the historical schedule. At the 1% participation cap, the historical schedule's median capacity limit was **29-31% lower with exchange-only volume** than with total reported volume. These limits use realised future volume: they are retrospective diagnostics added after the first run, not forecasts available before trading.

![Capacity limits and benchmark tracking](results/equities/equity_capacity.png)

These are historical price and capacity proxies, not actual fills or PnL. The model omits spreads, fees, queue position and own market impact. Two stocks cannot establish a reliable trading advantage.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -q
python -m study.equity
```

The 38 tests need no downloads. The final command needs the 80 raw stock files described in [data setup](data/README.md); it replaces the saved outputs. Missing usable evaluation data stops the run before any output is written. Raw data is not redistributed.

Browse the [640 comparison rows](results/equities/equity_daily.csv), [metadata](results/equities/equity_results.json), [methods and limits](docs/STUDY.md), or [dated protocol](docs/protocol.md). Earlier BTC/ETH outputs and the original layout remain on the [archive branch](https://github.com/Wanji-Jimmy/execution-timing-lab/tree/archive/before-2026-10-26-simplification). Shared calculations and their tests remain here; the stock experiment is the main entry point.
