# Stock data

Obtain the minute files from [AlgoSeek's Machine Learning for Trading companion data](https://algoseek.com/ml-for-trading/) under its applicable terms. Raw market data is not included in this repository.

Put 80 CSV files in `data/equities/`, named `EBAY_20210104.csv`, `LULU_20210104.csv`, etc. The exact 40 dates per stock are returned by `sessions()` in [study/equity.py](../study/equity.py): 4 January to 2 March 2021, excluding weekends, 18 January and 15 February.

Each file needs the original 90-column schema and every minute from 09:30 inclusive to 16:00 exclusive, America/New_York. Required fields and validation rules are in `PAIRS`, `REQUIRED` and `load_session()`. The loader reports missing or invalid sessions rather than substituting data. The first 20 valid prior sessions form each evaluated day's history; an evaluated day with insufficient history is skipped.

After setting up the Python environment in the [README](../README.md) and placing the files above, run this from the repository root:

```bash
python -m study.equity
```

The module command above requires the repository root as the working directory. To run from elsewhere, invoke the script directly with absolute paths:

```bash
python /path/to/execution-timing-lab/study/equity.py \
  --data-dir /path/to/stock-files \
  --output-dir /path/to/new-results
```

A successful run writes a CSV, JSON and chart. Existing files with those names are replaced. No usable evaluation records means no output is written. The [saved metadata](../results/equities/equity_results.json) records the original file checksums and coverage; the [study](../docs/STUDY.md) explains the session and price definitions.
