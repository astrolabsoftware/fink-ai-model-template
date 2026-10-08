"""
Read Fink alerts from files, in the same shape as in production.

In production, pre_processing() receives one alert of your data transfer, as a
dict with every column you selected, except the cutouts (images). The readers
below give exactly that, whatever the file format, so a preprocessing that works
on your files also works in the container.

Supported inputs: a file, or a folder read recursively, of
  - .parquet   e.g. the output of fink_datatransfer (ftransfer_ztf_<date>_<id>/)
  - .avro      Avro container files
  - .jsonl     one alert per line

To also read some columns as labels (tnsclass, roid...), see
read_alerts(columns=...).
"""

import json
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, islice, zip_longest
from pathlib import Path

import fastavro
import pyarrow.parquet as pq

# The images, removed before pre_processing() in production too
CUTOUT_FIELDS = {"cutoutScience", "cutoutTemplate", "cutoutDifference"}

FORMATS = (".parquet", ".avro", ".jsonl")

# Parquet files read at the same time. fink_datatransfer writes thousands of small
# files: the time goes into waiting for the disk, not into the CPU.
READ_THREADS = 16

# A complete alert, as decoded from the Avro feed (float32 values)
FULL_ALERT = {
    "objectId": "ZTF21smoketest",
    "candid": 1234567890,
    "candidate": {
        "rb": 0.9200000166893005,
        "drb": 0.8700000047683716,
        "magpsf": 19.299999237060547,
        "sigmapsf": 0.03999999910593033,
        "isdiffpos": "t",
        "ra": 10.684,
        "dec": 41.269,
        "jd": 2459995.5,
        "fid": 1,
    },
    "prv_candidates": [
        {"jd": 2459995.5, "magpsf": 19.1, "isdiffpos": "t", "fid": 1},
        {"jd": 2459990.3, "magpsf": 19.4, "isdiffpos": "t", "fid": 2},
        # Upper limit (non-detection): no magnitude, only the limiting magnitude
        {"jd": 2459993.8, "magpsf": None, "diffmaglim": 20.1, "isdiffpos": None, "fid": 1},
    ],
}

# An alert of a data transfer with a selection of columns: the values are at the
# top level, there is no candidate and no history
FLAT_ALERT = {
    "objectId": "ZTF26smoketest",
    "candid": 1234567891,
    "magpsf": 17.858,
    "sigmapsf": 0.051,
    "fid": 2,
    "jd": 2461314.6,
    "jd_first_real_det": 2461298.7,
    "nalerthist": 29,
    "mag_rate": -0.06,
    "sigma_rate": 0.093,
    "delta_time": 0.913,
    "from_upper": False,
    "lc_features_g": {"amplitude": 0.857, "median": 17.02},
    # Not enough points in this filter: the features are NaN or None
    "lc_features_r": {"amplitude": float("nan"), "median": None},
    "roid": 0,
}

# Incomplete alerts that real streams do contain
INCOMPLETE_ALERTS = [
    {},
    {"candidate": {}},
    {"candidate": None, "prv_candidates": None},
    {"candidate": {"rb": None, "magpsf": "not a number", "drb": float("nan")}},
    # What Avro gives for an empty record: every key is there, every value is None
    {"objectId": None, "candid": None, "candidate": None, "prv_candidates": None},
    {"candidate": {"rb": None, "drb": None, "magpsf": None, "isdiffpos": None}, "prv_candidates": [{}]},
    {"magpsf": None, "mag_rate": None, "from_upper": None, "lc_features_g": None},
]

EXAMPLE_ALERTS = [FULL_ALERT, FLAT_ALERT, *INCOMPLETE_ALERTS]


def alert_files(path):
    """Return the alert files of path (a file, or a folder read recursively), sorted."""
    path = Path(path).expanduser()
    if path.is_file():
        return [path] if path.suffix in FORMATS else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in FORMATS)


def read_alerts(path, limit=None, columns=()):
    """Yield (alert, extra) for each alert of path.

    alert: exactly what pre_processing() receives in production.
    extra: the columns asked for, e.g. columns=["tnsclass"] for labels.
           A field inside the alert is asked for with a dot: "candidate.classtar".

    With a limit, at most limit alerts are read, from files taken in turn from
    each folder (class, night...): every folder is represented without opening
    every file, which is the slow part.
    """
    if not Path(path).expanduser().exists():
        raise FileNotFoundError(f"{path} does not exist: check the path of your alerts.")
    files = alert_files(path)
    if not files:
        raise FileNotFoundError(f"No alert file ({', '.join(FORMATS)}) in {path}")
    if limit:
        files = _one_folder_after_the_other(files)

    count = 0
    for file, records in _read_files(files, limit):
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{file}: an alert must be an object, not {type(record).__name__}")
            alert = {name: value for name, value in record.items() if name not in CUTOUT_FIELDS}
            extra = {name: _get(record, name) for name in columns}
            yield alert, extra
            count += 1
            if limit and count >= limit:
                return


def load_alerts(path, limit=None):
    """Yield the alerts of path one by one, as dicts, like the service does."""
    for alert, _ in read_alerts(path, limit):
        yield alert


def _one_folder_after_the_other(files):
    """Order files by taking one from each folder in turn (random but always the
    same file in each folder), so that the first files cover every folder."""
    folders = {}
    for file in files:
        folders.setdefault(file.parent, []).append(file)
    rng = random.Random(0)
    for group in folders.values():
        rng.shuffle(group)
    return [file for file in chain(*zip_longest(*folders.values())) if file is not None]


def _read_files(files, max_records):
    """Yield (file, records) in order, at most max_records per file. Parquet files
    are read ahead by threads."""
    pool = ThreadPoolExecutor(READ_THREADS)
    jobs = (
        (file, pool.submit(_parquet_batches, file, max_records) if file.suffix == ".parquet" else None)
        for file in files
    )
    ahead = deque(islice(jobs, 2 * READ_THREADS))
    try:
        while ahead:
            file, future = ahead.popleft()
            ahead.extend(islice(jobs, 1))
            if future is None:
                yield file, _records(file, max_records)
            else:
                # Converting to dicts is done here, batch by batch, to save memory
                rows = (row for batch in future.result() for row in batch.to_pylist())
                yield file, islice(rows, max_records)
    finally:
        # The reader stopped early (limit reached): skip the files read ahead
        pool.shutdown(cancel_futures=True)


def _parquet_batches(path, max_records):
    """Read at most max_records rows of one parquet file, as Arrow record batches.
    The cutouts are not read: they are the heavy part, and pre_processing() never
    gets them."""
    parquet = pq.ParquetFile(path)
    names = [name for name in parquet.schema_arrow.names if name not in CUTOUT_FIELDS]
    batches, count = [], 0
    # Small batches when few alerts are needed, so that no row is read for nothing
    for batch in parquet.iter_batches(batch_size=min(500, max_records or 500), columns=names):
        batches.append(batch)
        count += batch.num_rows
        if max_records is not None and count >= max_records:
            break
    return batches


def _records(path, max_records):
    """Yield at most max_records records of one .avro or .jsonl file."""
    if path.suffix == ".avro":
        with open(path, "rb") as f:
            yield from islice(fastavro.reader(f), max_records)
        return
    with open(path) as f:
        yield from islice((json.loads(line) for line in f if line.strip()), max_records)


def _get(record, dotted_name):
    """record["candidate"]["classtar"] for "candidate.classtar", None if missing."""
    value = record
    for key in dotted_name.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value
