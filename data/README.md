# Stock data

Use the [AlgoSeek Machine Learning for Trading companion data](https://algoseek.com/ml-for-trading/), under its applicable terms. Select **AAPL, MSFT, AMZN and INTC, 1 April-30 June 2021** from the 2021 daily archives. Raw prices are not redistributed here.

The [acquisition manifest](acquisition_manifest.json) identifies the 252 source members, public archive links, CRC32 values and SHA256 hashes. Extract only the four selected stocks from each daily ZIP and place them in `data/equities/` as `AAPL_20210401.csv`, `MSFT_20210401.csv`, etc. The original archive member is named, for example, `20210401/A/AAPL.csv`.

There are 63 sessions per stock. [sessions()](../study/equity.py) lists them: Good Friday (2 April) and Memorial Day (31 May) are excluded; 18 and 21 June were open in 2021. The first 20 dates end 29 April. The next 43 dates, 30 April-30 June, use a rolling history of the preceding 20 valid sessions.

Each CSV needs the original 90-column schema and all minute starts from 09:30 inclusive to 16:00 exclusive, America/New_York. `PAIRS`, `REQUIRED` and `load_session()` define the fields and checks. A zero-volume minute may have no VWAP; a missing minute is not treated as zero. Invalid sessions and insufficient history are reported rather than substituted.

After installing the dependencies in the [README](../README.md), run from the repository root:

```bash
python -m study.equity
```

To keep raw data elsewhere and avoid replacing the saved results:

```bash
python /path/to/execution-timing-lab/study/equity.py \
  --data-dir /path/to/stock-files \
  --output-dir /path/to/new-results
```

A successful run writes a CSV, JSON and chart. Existing files with those names in the output directory are replaced. No usable evaluation records means no output is written. Tests use synthetic data and need no downloads.

The saved result metadata records all input checksums and history dates. This sample has no common stocks, dates or raw-file hashes with the EBAY/LULU stock diagnostic in Market Making Lab. Both use AlgoSeek; different observations do not imply statistically independent markets.

Provider adjustment flags were not independently certified. Issuer split histories show no split in this window, but cash dividends occurred; no additional price or volume adjustment is applied. See the [dated protocol](../docs/protocol.md#independent-equity-sample---26-october-2026) for sources and the previous pilot's archive. Source availability and terms can change.
