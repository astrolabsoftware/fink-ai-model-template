"""
The checks of fink_model.py, as tests.

Run:
    pytest test_contract.py
"""

import json

import pytest

from fink_model import PREPROCESSING_DIR, _check_speed, _compute_features, contract_error, preprocessing
from load_alerts import EXAMPLE_ALERTS, FLAT_ALERT, FULL_ALERT

prep = preprocessing()


@pytest.mark.parametrize("alert", EXAMPLE_ALERTS)
def test_returns_one_number_per_feature(alert):
    """Every alert gives a list of len(FEATURE_NAMES) finite numbers, without raising."""
    error = contract_error(prep.pre_processing(alert), prep.FEATURE_NAMES)
    assert error is None, error


def test_fast_enough_for_the_stream():
    _, ms_per_alert = _compute_features(prep, EXAMPLE_ALERTS * 100)
    _check_speed(ms_per_alert)


def test_input_alert_is_not_modified():
    """The same alert may be reused, so pre_processing() must not change it."""
    before = json.dumps(FULL_ALERT, sort_keys=True)
    prep.pre_processing(FULL_ALERT)
    assert json.dumps(FULL_ALERT, sort_keys=True) == before


def test_feature_names_are_unique():
    """Duplicate names usually mean a copy-paste mistake."""
    assert len(set(prep.FEATURE_NAMES)) == len(prep.FEATURE_NAMES)


def test_features_of_the_example_alert():
    """Values of the preprocessing of this template: adapt or delete with yours."""
    features = dict(zip(prep.FEATURE_NAMES, prep.pre_processing(FLAT_ALERT)))
    assert features["magpsf"] == 17.858 and round(features["days_since_first"], 1) == 15.9
    # amplitude_r is NaN in FLAT_ALERT: only one filter has light curve features
    assert (features["amplitude_g"], features["amplitude_r"], features["n_bands_with_lc"]) == (0.857, 0.0, 1.0)


def test_requirements_file_exists():
    """The container installs its packages from this file (it may be empty)."""
    assert (PREPROCESSING_DIR / "requirements.txt").exists()
