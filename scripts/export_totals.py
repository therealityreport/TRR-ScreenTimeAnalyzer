#!/usr/bin/env python3
"""CLI for exporting totals JSON into CSV or sheet-ready format."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from screentime.io_utils import setup_logging


LOGGER = logging.getLogger("scripts.export_totals")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert totals JSON to CSV")
    parser.add_argument("totals_json", type=Path, help="Path to *-TOTALS.json")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (defaults to same stem .csv)",
    )
    parser.add_argument(
        "--sheet",
        action="store_true",
        help="Emit CSV formatted for Google Sheets (adds header row)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    totals = json.loads(args.totals_json.read_text(encoding="utf-8"))
    df = pd.DataFrame(totals)
    output_path = args.output or args.totals_json.with_suffix(".csv")
    df.to_csv(output_path, index=False)
    LOGGER.info("Exported totals CSV to %s", output_path)


if __name__ == "__main__":
    main()
