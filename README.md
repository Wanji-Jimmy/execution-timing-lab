# OxHack: Stock Order Execution

**How much of an order can a trading schedule finish before the deadline?**

A Python project on order timing and capacity, started at OxHack and extended afterwards. It compares equal release (TWAP), a historical volume schedule, a fixed early release, and a rule that reacts to past intraday volume. The aim is to separate insufficient total capacity from capacity lost by releasing an order too late.

## Main finding

**At USD 1 million, every method completed on every evaluated stock-day in the model. Volume feedback never changed the historical schedule.** The extra rule added no value in this sample.

The stock experiment uses EBAY and LULU minute bars from 4 January to 2 March 2021: 40 sessions each, with the first 20 for history and the remaining 20 evaluated using a strictly prior rolling window. The sample and settings were fixed before the comparisons.

Each release path is replayed at 1% and 5% participation caps, using either total reported volume or exchange-only volume as capacity. Prices and the benchmark stay fixed across those two definitions.

All USD 1m orders fit, but the capacity definitions leave different headroom. For the historical schedule at the 1% cap:

| Stock | Median limit: total volume | Median limit: exchange only |
|---|---:|---:|
| EBAY | USD 2.86m | USD 1.96m |
| LULU | USD 2.87m | USD 2.03m |

These limits are calculated from realised future volume. They are a **retrospective diagnostic added after the first run**, not a size recommendation available before trading. Fixed early release increased the limit but tracked the price benchmark less closely. That is a trade-off, not a universal strategy ranking.

![Capacity limits and benchmark tracking](results/equities/equity_capacity.png)

## View the saved results

No download or Python setup is needed to inspect the [640 comparison rows](results/equities/equity_daily.csv), [coverage, hashes and summaries](results/equities/equity_results.json), or chart above. The rows cover 40 evaluated stock-days, four methods, two caps and two volume definitions.

The [full study](docs/STUDY.md) explains the schedules, capacity formula, field checks and earlier BTC/ETH experiment. The [protocol](protocol.md) records design decisions and amendments. Saved source hashes identify the original evaluation; later input checks and documentation changes are recorded there.

## Run the tests

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest -q
```

The 38 tests run without market-data downloads. They check quantity conservation, capacity limits, causal decisions, missing and zero-volume data, and the analytical completion boundary. A regression test also checks that missing input cannot overwrite existing results.

## Recompute the stock experiment

First obtain the minute files from [AlgoSeek's book companion data](https://algoseek.com/ml-for-trading/) under the applicable terms. Put all 80 files in `data/equities/`, named `EBAY_20210104.csv`, `LULU_20210104.csv`, etc. The exact dates are in `sessions()` in [equity.py](equity.py). Raw stock data is excluded and is not redistributed.

Then, with the environment above active:

```bash
python equity.py --data-dir data/equities --output-dir results/equities
```

This replaces the saved stock outputs. If no usable evaluation records exist, the command stops before writing any output.

## Model limits

Fills use historical five-minute reference prices and assumed participation caps. Bars do not reveal actual fills, queue position, spreads, fees or own market impact. Reported volume does not establish access to a venue's liquidity.

The session is 09:30 inclusive to 16:00 exclusive, America/New_York; records assigned to 16:00 are excluded. The benchmark is therefore a selected-window reference, not an auction-complete daily VWAP. Shares are continuous, and provider adjustment flags remain uncertified. Two stocks and 20 evaluation dates each cannot establish future superiority, profit or net execution savings.
