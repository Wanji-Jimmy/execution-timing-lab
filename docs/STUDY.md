# Stock Order Execution

How much of an order can a trading schedule finish before the deadline?

This study separates too little total capacity from releasing the order too late to use earlier capacity. The current sample is AAPL, MSFT, AMZN and INTC from 1 April to 30 June 2021. Each stock has 63 sessions: 20 for warm-up and 43 for evaluation, using a rolling history of the preceding 20 valid sessions.

The names and full quarter were selected before calculating these results. They share no stocks, dates or raw-file hashes with Market Making Lab's EBAY/LULU sample. Both use AlgoSeek; four selected Nasdaq stocks are not a representative universe or statistically independent markets.

## Four release rules

- **TWAP:** release equal quantities every five minutes.
- **Historical:** follow the average intraday volume shares of the previous 20 sessions.
- **Fixed early:** release 10% immediately and follow the historical plan for the rest.
- **Feedback:** advance the historical plan if past intraday volume suggests insufficient later capacity.

Each method buys the same share quantity, initially worth USD 1m. Eligible shares fill at a historical five-minute reference price, up to an assumed 1% or 5% of reported volume. Unfilled shares carry forward. This is a hypothetical replay, not observed execution.

## What changed with the new data

All four methods completed on all 43 evaluation dates per stock under both caps and volume definitions. Feedback made no change to the historical schedule. The more complex rule did not add value at this size in this sample.

At the 1% cap, mean absolute distance from the common selected-window VWAP was:

| Capacity definition | Stock | TWAP (bp) | Historical (bp) | Fixed early (bp) | Feedback (bp) |
|---|---|---:|---:|---:|---:|
| Total | AAPL | 7.12 | 3.73 | 7.51 | 3.73 |
| Exchange only | AAPL | 7.12 | 3.73 | 7.51 | 3.73 |
| Total | MSFT | 6.37 | 3.98 | 6.89 | 3.98 |
| Exchange only | MSFT | 6.37 | 3.98 | 6.89 | 3.98 |
| Total | AMZN | 7.28 | 4.24 | 9.00 | 4.24 |
| Exchange only | AMZN | 7.28 | 4.24 | 9.00 | 4.24 |
| Total | INTC | 8.15 | 5.37 | 8.72 | 5.37 |
| Exchange only | INTC | 8.15 | 5.37 | 8.72 | 5.37 |

One basis point is 0.01%. Historical weights tracked the benchmark more closely than equal release for every stock here. Fixed early release tracked it less closely than historical weights. Smaller absolute distance does not necessarily mean a cheaper buy price, and these averages do not establish future superiority or cost savings. Price comparisons include only dates where every rule completes under both scopes; unfinished orders remain in completion statistics.

## Why volume coverage matters

The source separates exchange-reported `Volume` from `FinraVolume`; `TotalVolume` is their sum. All 98,280 selected minute rows passed coverage and volume-identity checks. A zero-volume minute is retained; a missing minute is not silently replaced with zero.

The same release path is replayed with two capacity definitions: a percentage of Total volume, or a percentage of exchange-only volume. Both retain the same Total-field price reference and benchmark. Only the capacity assumption changes. This does not simulate routing or imply that reported off-exchange volume is available to our order.

At USD 1m, neither definition prevents completion, but they imply different headroom. For the historical schedule at the 1% cap:

| Stock | Median Total limit ($m) | Median exchange-only limit ($m) | Lowest exchange-only limit ($m) | Reduction between medians |
|---|---:|---:|---:|---:|
| AAPL | 74.59 | 39.94 | 27.45 | 46.4% |
| MSFT | 39.65 | 27.88 | 19.56 | 29.7% |
| AMZN | 72.69 | 48.15 | 30.89 | 33.8% |
| INTC | 8.55 | 5.89 | 3.03 | 31.1% |

The reduction is `1 - median(exchange limit) / median(Total limit)`, calculated separately for each stock. It is a difference in modelled capacity, not a reduction in trading costs or proof of accessible liquidity.

## A capacity check without a size search

For a fixed fractional schedule, every suffix of the day must have enough remaining capacity for shares still to be released. If `r[j-1]` is the fraction released before bucket j, with `r[-1] = 0`:

```
Q_max = min_j(cap * sum(volume[j:]) / (1 - r[j-1]))
```

Ignore zero denominators. The first suffix is the whole-day capacity bound; later suffixes detect capacity lost by waiting too long. This applies to TWAP, historical and fixed early. Feedback is excluded because its fractions change with order size.

For example, capacity of 50, 20 and 10 shares allows an equal release schedule to finish at most 30 shares, although total capacity is 80. At Q = 60 it releases 20 per bucket and fills 20, 20, 10, leaving 10 unfilled.

The formula uses realised future volume. It is a retrospective diagnostic, not a forecast. It was developed after the earlier EBAY/LULU pilot and fixed before this replacement sample was evaluated. A higher capacity limit and closer benchmark tracking are different objectives.

## Reproduce and read the code

Follow [data setup](../data/README.md) for the 252 source members and checksums. Raw data stays local. After installing the dependencies:

```bash
python -m unittest discover -s tests -q
python -m study.equity --data-dir /path/to/stock-files --output-dir /path/to/new-results
```

The tests use synthetic data. The equity command requires the original source files and refuses to write output if no usable evaluation records exist. Missing or invalid sessions are disclosed; insufficient prior history is skipped rather than supplemented with future data.

Read [equity.py](../study/equity.py) for loading and evaluation, [adaptive.py](../study/adaptive.py) for release and replay, then [stock tests](../tests/test_equity.py). [analysis.py](../study/analysis.py) holds optional baseline and diagnostic helpers, not another experiment entry point.

The selected window is 09:30 <= minute start < 16:00, America/New_York, grouped into 78 five-minute buckets. Each VWAP is weighted by its corresponding volume field. The share quantity uses the first reported trade in the opening minute, not that minute's later VWAP. Records assigned to 16:00 are excluded, so this is not an auction-complete full-day benchmark.

The CSV contains 2,752 comparisons across 172 evaluated stock-days. The JSON records coverage, exclusions, exact history dates, input hashes and current producer hashes. Tests cover quantity conservation, caps, causal decisions, missing/zero-volume data, separation of stock histories and the analytical boundary.

## Earlier work and limits

The [previous pilot archive](https://github.com/Wanji-Jimmy/execution-timing-lab/tree/archive/before-2026-10-26-independent-data) preserves EBAY/LULU results, including the old 29-31% capacity difference. Those numbers are not results of the new sample. The [earlier experiment archive](https://github.com/Wanji-Jimmy/execution-timing-lab/tree/archive/before-2026-10-26-simplification) also retains the original BTC/ETH work. The [protocol](protocol.md) preserves the research sequence and dated selection rules.

Historical bars do not reveal queues, spreads paid, fees, own market impact or actual fills. Participation caps are scenarios, and continuous share quantities are an approximation. Four stocks over one quarter do not establish profit, executable size guarantees or broad market robustness. Issuer histories show no split in the window, but cash dividends occurred; provider prices and share volumes are unchanged, with retrospective adjustment flags not independently certified.

- [AlgoSeek field definitions](https://algoseek.com/docs/rest-api/reference/endpoint-v-1-data-us-equity-taq-1-min-identifier-get) and [data licensing](https://algoseek.com/licensing-faq/).
- [CALM](https://github.com/Shivesh777/CALM) motivated comparing forecast quality with execution outcomes; its online algorithm is not implemented here.
- [Busseti-Boyd](https://stanford.edu/~boyd/papers/vwap_opt_exec.html) and [Markov-Mazur-Saltz](https://arxiv.org/html/1409.1441v1) inform the scheduling question; this study does not reproduce their optimizers.
- [hftbacktest fill assumptions](https://hftbacktest.readthedocs.io/en/latest/order_fill.html) inform the replay limitations; no hftbacktest code is used.
