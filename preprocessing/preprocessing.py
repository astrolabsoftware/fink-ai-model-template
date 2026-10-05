"""
Preprocessing: turns one raw ZTF alert into the feature vector of your model.

It runs in the preprocessing container, on every alert coming from Kafka. Your
model receives its output, so compute your training features with this same
function.

Input: one alert of the Fink AI feed, as a Python dict. The feed is Avro
(schema ztf_alert_v3.3_inference.avsc), decoded by the service before the call:
    {
      "objectId": "ZTF21...",        # str or None
      "candid": 1500000000000000001, # int or None
      "candidate": {...} or None,    # the detection: rb, drb, magpsf, isdiffpos...
      "prv_candidates": [...] or None,  # 30 days of history, oldest first
    }
  - every field of the schema is there, but any value can be None;
  - Avro floats are float32: 0.92 arrives as 0.9200000166893005;
  - isdiffpos is a string: "t"/"1" (positive) or "f"/"0" (negative);
  - prv_candidates mixes detections and upper limits (non-detections):
    an upper limit has magpsf None and only diffmaglim;
  - the cutouts (images) are removed.

Contract (checked by test_contract.py):
  - pre_processing(alert) takes one alert (a dict) and returns a list of floats;
  - the list always has len(FEATURE_NAMES) values, in the same order;
  - it never fails on a missing or None field: it uses a default value instead.

Only the Python standard library and the packages listed in
preprocessing/requirements.txt are available in the container.
"""

import math

# One name per value returned by pre_processing(), in the same order.
# Changing this list changes the model input: retrain and upload a new version.
FEATURE_NAMES = [
    "rb",  # real/bogus score computed by ZTF
    "drb",  # deep learning real/bogus score
    "magpsf",  # PSF magnitude
    "isdiffpos",  # 1.0 if the subtraction is positive, else 0.0
    "n_prev_det",  # number of previous detections (upper limits excluded)
]


def _to_float(value, default=0.0):
    """Return value as a finite float, or default if it is missing or invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def pre_processing(alert):
    """Return the feature vector of one alert, in the order of FEATURE_NAMES."""
    candidate = alert.get("candidate") or {}
    previous_detections = alert.get("prv_candidates") or []  # detections + upper limits

    return [
        _to_float(candidate.get("rb")),
        _to_float(candidate.get("drb")),
        _to_float(candidate.get("magpsf")),
        1.0 if candidate.get("isdiffpos") in ("t", "1") else 0.0,
        float(sum(1 for p in previous_detections if p and p.get("magpsf") is not None)),
    ]
