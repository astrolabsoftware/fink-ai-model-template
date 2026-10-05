"""
Check your preprocessing on your alerts, before training.

    python check.py ~/fink-client/ftransfer_ztf_2026-09-28_221528
    python check.py ~/fink-client/ftransfer_ztf_2026-09-28_221528 --all

It runs pre_processing() on the alerts exactly as in production, and stops with
the list of alerts that fail, or if it is too slow for the stream.
"""

import argparse
import sys

from fink_model import FinkModelError, check

parser = argparse.ArgumentParser(description="Check your preprocessing on your alerts.")
parser.add_argument("path", help="file or folder of alerts (.parquet, .avro, .jsonl)")
parser.add_argument("--all", action="store_true", help="read every alert (default: 2000)")
args = parser.parse_args()

try:
    check(args.path, limit=None if args.all else 2000)
except (FinkModelError, FileNotFoundError, ValueError) as exc:
    sys.exit(f"\nNOT OK: {exc}")
