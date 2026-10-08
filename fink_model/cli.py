"""
Check your preprocessing on your alerts, before training.

    fink-model check ~/fink-client/ftransfer_ztf_2026-09-28_221528
    fink-model check ~/fink-client/ftransfer_ztf_2026-09-28_221528 --all

It runs pre_processing() on the alerts exactly as in production, and stops with
the list of alerts that fail, or if it is too slow for the stream.
"""

import argparse
import sys

from fink_model import FinkModelError, check


def main():
    parser = argparse.ArgumentParser(prog="fink-model")
    commands = parser.add_subparsers(dest="command", required=True)
    check_parser = commands.add_parser("check", help="check your preprocessing on your alerts")
    check_parser.add_argument("path", help="file or folder of alerts (.parquet, .avro, .jsonl)")
    check_parser.add_argument("--all", action="store_true", help="read every alert (default: 2000)")
    args = parser.parse_args()

    try:
        check(args.path, limit=None if args.all else 2000)
    except (FinkModelError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"\nNOT OK: {exc}")
