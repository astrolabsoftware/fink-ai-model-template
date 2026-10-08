"""
Your preprocessing: turns one alert into the features of your model.

Input: one alert of your data transfer, as a Python dict. With a selection of
columns, the values are at the top level (no candidate, no prv_candidates):
    {
      "objectId": "ZTF26...",        # str
      "candid": 3560118260615015042, # int
      "magpsf": 17.86, "sigmapsf": 0.05, "fid": 2, "jd": 2461314.6,
      "nalerthist": 29, "mag_rate": -0.06, "from_upper": False,
      "lc_features_g": {"amplitude": 0.86, "median": 17.02, ...},
      ...                            # every column you selected
    }
  - any value can be None: mag_rate has no value for some alerts;
  - the light curve features (lc_features_g, lc_features_r) need several
    detections in the filter: they are NaN for a new object;
  - the cutouts (images) are removed.

Contract (checked by test_contract.py):
  - pre_processing(alert) takes one alert (a dict) and returns a list of floats;
  - the list always has len(FEATURE_NAMES) values, in the same order;
  - it never fails on a missing or None field: it uses a default value instead.

Only the Python standard library and the packages listed in
preprocessing/requirements.txt are available in the container.
"""

import math

# Columns copied as they are, in this order
DIRECT = [
    "magpsf",  # PSF magnitude
    "sigmapsf",  # error on magpsf
    "fid",  # filter: 1 = g, 2 = r
    "nalerthist",  # number of alerts already sent for this object
    "mag_rate",  # magnitude change per day since the last measurement
    "sigma_rate",  # error on mag_rate
    "delta_time",  # days since the last measurement
]

# One name per value returned by pre_processing(), in the same order.
# Changing this list changes the model input: retrain your model.
FEATURE_NAMES = DIRECT + [
    "days_since_first",  # days since the first detection of the object
    "from_upper",  # 1.0 if the last measurement was an upper limit
    "amplitude_g",  # half of the magnitude range in g (0.0 if not computed)
    "amplitude_r",  # same in r
    "n_bands_with_lc",  # filters with light curve features: 0, 1 or 2
]


def _to_float(value, default=0.0):
    """Return value as a finite float, or default if it is missing or invalid."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _amplitude(alert, band):
    """Amplitude of the light curve in one filter, None if it is not computed."""
    features = alert.get(band)
    return _to_float(features.get("amplitude"), None) if isinstance(features, dict) else None


def pre_processing(alert):
    """Return the feature vector of one alert, in the order of FEATURE_NAMES."""
    jd = _to_float(alert.get("jd"))
    amplitude_g = _amplitude(alert, "lc_features_g")
    amplitude_r = _amplitude(alert, "lc_features_r")

    return [_to_float(alert.get(name)) for name in DIRECT] + [
        max(0.0, jd - _to_float(alert.get("jd_first_real_det"), jd)),
        1.0 if alert.get("from_upper") is True else 0.0,
        amplitude_g or 0.0,
        amplitude_r or 0.0,
        float((amplitude_g is not None) + (amplitude_r is not None)),
    ]
