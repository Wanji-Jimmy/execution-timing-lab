"""Download, verify and aggregate the fixed public-data pilot sample."""

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
OUT = ROOT / "data"
ASSETS = ("BTCUSDT", "ETHUSDT")
MONTHS = ("2026-06", "2026-07", "2026-08", "2026-09")
VALIDATION_DATES = tuple(f"2026-10-{day:02d}" for day in range(1, 24))
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
MINUTE = 60_000_000  # Binance spot archive timestamps use microseconds from 2025.


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def period_days(period):
    if len(period) == 10:
        return [date.fromisoformat(period).isoformat()]
    year, month = map(int, period.split("-"))
    return [date(year, month, day).isoformat()
            for day in range(1, calendar.monthrange(year, month)[1] + 1)]


def source_record(asset, period, frequency):
    if frequency not in ("monthly", "daily"):
        raise ValueError("frequency must be monthly or daily")
    if len(period) != (7 if frequency == "monthly" else 10):
        raise ValueError("period does not match archive frequency")
    period_days(period)
    name = f"{asset}-1m-{period}.zip"
    base = BASE_URL.replace("/monthly/", f"/{frequency}/")
    url = f"{base}/{asset}/1m/{name}"
    return {"asset": asset, "month" if frequency == "monthly" else "date": period,
            "frequency": frequency, "url": url, "checksum_url": url + ".CHECKSUM"}


def acquire(asset, month, frequency="monthly"):
    source = source_record(asset, month, frequency)
    name = f"{asset}-1m-{month}.zip"
    url = source["url"]
    with urlopen(url + ".CHECKSUM", timeout=60) as response:
        checksum_text = response.read().decode("ascii").strip()
    checksum_time = now()
    parts = checksum_text.split()
    if len(parts) != 2 or parts[1].lstrip("*") != name:
        raise ValueError(f"Unexpected checksum record: {checksum_text}")
    expected = parts[0].lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Invalid SHA-256 for {name}")
    (CACHE / (name + ".CHECKSUM")).write_text(checksum_text + "\n")
    archive = CACHE / name
    downloaded = not archive.exists()
    if downloaded:
        temporary = archive.with_suffix(".zip.part")
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as target:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    target.write(chunk)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    actual = sha256(archive)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {name}: {actual} != {expected}")
    return archive, {
        **source, "sha256": actual,
        "checksum_verified": True, "bytes": archive.stat().st_size,
        "archive_retrieved_at": datetime.fromtimestamp(
            archive.stat().st_mtime, timezone.utc).isoformat(),
        "checksum_retrieved_at": checksum_time,
        "downloaded_this_run": downloaded,
    }


def validate(row, timestamp):
    if len(row) != 12:
        raise ValueError("wrong_column_count")
    o, h, low, close, volume = map(float, row[1:6])
    quote, taker_base, taker_quote = float(row[7]), float(row[9]), float(row[10])
    numbers = (o, h, low, close, volume, quote, taker_base, taker_quote)
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("nonfinite_value")
    if min(o, h, low, close) <= 0 or min(volume, quote, taker_base, taker_quote) < 0:
        raise ValueError("nonpositive_price_or_negative_volume")
    if not low <= min(o, close) <= max(o, close) <= h:
        raise ValueError("inconsistent_ohlc")
    if timestamp % MINUTE or int(row[6]) != timestamp + MINUTE - 1:
        raise ValueError("invalid_minute_boundary")
    if int(row[8]) < 0 or taker_base > volume * (1 + 1e-8) or taker_quote > quote * (1 + 1e-8):
        raise ValueError("invalid_trade_or_taker_volume")
    if (volume == 0) != (quote == 0):
        raise ValueError("inconsistent_zero_volume")
    if volume and not low * (1 - 1e-8) <= quote / volume <= h * (1 + 1e-8):
        raise ValueError("vwap_outside_price_range")
    return o, volume, quote


def aggregate(archive, asset, month):
    days = period_days(month)
    minutes = {day: {} for day in days}
    errors = {day: Counter() for day in days}
    counts = Counter()
    with ZipFile(archive) as zipped:
        names = zipped.namelist()
        if names != [archive.stem + ".csv"]:
            raise ValueError(f"Unexpected archive members: {names}")
        with zipped.open(names[0]) as binary, io.TextIOWrapper(binary) as stream:
            for row in csv.reader(stream):
                counts["archive_rows"] += 1
                timestamp = int(row[0])
                dt = datetime.fromtimestamp(timestamp / 1_000_000, timezone.utc)
                day = dt.date().isoformat()
                if day not in minutes:
                    raise ValueError(f"Out-of-period row: {day} in {archive.name}")
                if not 8 <= dt.hour < 16:
                    continue
                counts["window_rows"] += 1
                slot = (dt.hour - 8) * 60 + dt.minute
                if slot in minutes[day]:
                    errors[day]["duplicate_minute"] += 1
                try:
                    values = validate(row, timestamp)
                except (ValueError, OverflowError) as exc:
                    errors[day][str(exc)] += 1
                    counts["invalid_window_rows"] += 1
                    values = None
                minutes[day][slot] = values

    output, excluded = [], []
    for day in days:
        rows = minutes[day]
        missing = sorted(set(range(480)) - rows.keys())
        if missing:
            errors[day]["missing_minutes"] = len(missing)
        buckets = []
        if not errors[day]:
            for bucket in range(96):
                chunk = [rows[i] for i in range(bucket * 5, bucket * 5 + 5)]
                volume = math.fsum(x[1] for x in chunk)
                quote = math.fsum(x[2] for x in chunk)
                if volume <= 0 or quote <= 0:
                    errors[day]["nonpositive_bucket_volume"] += 1
                buckets.append((asset, day, bucket, chunk[0][0], volume, quote))
        if errors[day]:
            excluded.append({"asset": asset, "date": day,
                             "reasons": dict(errors[day]),
                             "missing_minute_slots": missing})
        else:
            assert len(buckets) == 96
            output.extend(buckets)
    return output, {**dict(counts), "expected_days": len(days),
                    "valid_days": len(output) // 96, "excluded_days": excluded,
                    "validation_scope": "08:00 inclusive to 16:00 exclusive UTC"}


def collect(asset, period, frequency):
    source = source_record(asset, period, frequency)
    try:
        archive, source = acquire(asset, period, frequency)
        rows, checks = aggregate(archive, asset, period)
        return rows, {**source, **checks, "status": "complete"}
    except Exception as exc:
        return [], {**source, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true",
                        help="Fetch the fixed 1-23 October daily validation sample.")
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    periods = VALIDATION_DATES if args.validation else MONTHS
    frequency = "daily" if args.validation else "monthly"
    prefix = "validation_" if args.validation else ""
    csv_path = OUT / f"{prefix}buckets.csv"
    manifest_path = OUT / f"{prefix}manifest.json"
    manifest = {
        "generated_at": now(), "status": "running", "sources": [], "errors": [],
        "sample": "October validation" if args.validation else "June-September pilot",
        "required_archives": len(ASSETS) * len(periods),
        "timestamp_unit": "microseconds", "window_utc": "08:00-16:00",
        "bucket_minutes": 5, "required_minutes_per_day": 480,
        "schema": ["asset", "date", "bucket", "open", "base_volume", "quote_volume"],
        "bucket_numbering": "0 through 95; bucket 0 starts at 08:00 UTC",
        "price_definition": "bucket VWAP = quote_volume / base_volume",
    }
    output = []
    requests = [(asset, period) for asset in ASSETS for period in periods]
    completed = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(collect, asset, period, frequency): (asset, period)
                   for asset, period in requests}
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            if len(completed) % 10 == 0 or len(completed) == len(requests):
                print(f"Checked {len(completed)}/{len(requests)} archives", flush=True)
    for request in requests:
        rows, source = completed[request]
        manifest["sources"].append(source)
        output.extend(rows)
        if source["status"] == "failed":
            manifest["errors"].append(source)
    manifest["status"] = "failed" if manifest["errors"] else "complete"
    manifest["bucket_rows"] = len(output)
    manifest["valid_asset_days"] = len(output) // 96
    manifest["verified_archives"] = sum(bool(s.get("checksum_verified"))
                                        for s in manifest["sources"])
    manifest["downloaded_archives"] = sum(bool(s.get("downloaded_this_run"))
                                          for s in manifest["sources"])
    manifest["archive_rows"] = sum(s.get("archive_rows", 0) for s in manifest["sources"])
    manifest["window_rows"] = sum(s.get("window_rows", 0) for s in manifest["sources"])
    manifest["excluded_asset_days"] = sum(len(s.get("excluded_days", []))
                                           for s in manifest["sources"])
    if not manifest["errors"]:
        temporary = csv_path.with_suffix(".csv.part")
        with temporary.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(manifest["schema"])
            writer.writerows(output)
        temporary.replace(csv_path)
        manifest["buckets_sha256"] = sha256(csv_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{manifest['status']}: {manifest['valid_asset_days']} valid asset-days, "
          f"{manifest['excluded_asset_days']} excluded", flush=True)
    if manifest["errors"]:
        raise SystemExit(f"Data build failed; inspect {manifest_path}. No CSV was written.")


if __name__ == "__main__":
    main()
