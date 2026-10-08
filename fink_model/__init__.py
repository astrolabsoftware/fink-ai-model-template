"""
Train a model for the Fink service from your notebook, tracked in MLflow.

You mainly use three functions (see train.ipynb):

    X, info = load_features("~/fink-client/ftransfer_ztf_...")   # your alerts
    with start_run("my-model"):                                  # MLflow run
        model.fit(X, y)                                          # your model
        log_model(model)                                         # checked, uploaded

They log everything the service needs, and refuse a model that would not work
in production:
  - pre_processing() must give len(FEATURE_NAMES) finite floats for every alert,
    even an incomplete one;
  - pre_processing() must take at most MAX_MS_PER_ALERT ms per alert;
  - the model must accept the output of pre_processing();
  - preprocessing/ must not change between load_features() and log_model().
"""

import contextlib
import hashlib
import importlib
import math
import os
import re
import sys
import time
from array import array
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fink_model.alerts import EXAMPLE_ALERTS, alert_files, read_alerts  # noqa: E402

# The folder next to fink_model/, at the root of the repository
PREPROCESSING_DIR = Path(__file__).parent.parent / "preprocessing"

# pre_processing() runs on every alert of the stream: above this, the preprocessing
# container cannot follow the stream. 5 ms = 200 alerts/s per CPU core.
MAX_MS_PER_ALERT = 5.0

# The model name also names the Docker images: preprocessing-<name>, model-<name>
MODEL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


class FinkModelError(Exception):
    """The preprocessing or the model would not work in production."""


# What load_features() and start_run() remember for log_model()
_state = {}


def load_features(path, limit=None, columns=()):
    """Read the alerts of path and compute their features with your pre_processing().

    path:    a file or a folder of alerts (.parquet, .avro, .jsonl).
    limit:   read at most this number of alerts, spread over all the files.
    columns: other columns to read, for your labels, e.g. ["tnsclass"] or
             ["candidate.classtar"] for a field inside the alert.

    Returns (X, info): X has one row of FEATURE_NAMES values per alert, info is a
    DataFrame with objectId, candid and the columns asked for, in the same order.
    """
    prep = preprocessing()
    # Incomplete alerts first: the stream contains them, your files maybe not
    _compute_features(prep, EXAMPLE_ALERTS)
    # objectId and candid are always the first columns of info
    columns = [name for name in dict.fromkeys(columns) if name not in ("objectId", "candid")]
    rows, fields = [], []

    def alerts():
        """Yield the alerts one by one: only their features are kept in memory."""
        for alert, extra in read_alerts(path, limit, columns):
            if not rows:
                fields.extend(alert)
            rows.append({"objectId": alert.get("objectId"), "candid": alert.get("candid"), **extra})
            yield alert

    X, ms_per_alert = _compute_features(prep, alerts())
    _check_speed(ms_per_alert)
    info = pd.DataFrame(rows, columns=["objectId", "candid", *columns])

    constant = [name for name, column in zip(prep.FEATURE_NAMES, X.T) if np.all(column == column[0])]
    if len(X) > 1 and len(constant) == len(prep.FEATURE_NAMES):
        raise FinkModelError(
            "Every feature has the same value for every alert: pre_processing() reads "
            "fields that your alerts do not have. Your alerts contain: "
            + ", ".join(fields[:15])
            + (f"... ({len(fields)} columns)" if len(fields) > 15 else "")
        )

    _state.update(
        data_path=str(Path(path).expanduser().resolve()),
        n_files=len(alert_files(path)),
        features=pd.DataFrame(X, columns=prep.FEATURE_NAMES),
        ms_per_alert=ms_per_alert,
        preprocessing_hash=_preprocessing_hash(),
    )

    print(f"{len(X)} alerts, {X.shape[1]} features, {ms_per_alert:.3f} ms per alert")
    if constant:
        print(f"Warning: same value for every alert (wrong field name?): {', '.join(constant)}")
    return X, info


@contextlib.contextmanager
def start_run(model_name):
    """Open an MLflow run that logs your training automatically.

    Parameters, metrics and the training data are logged for you. Call
    log_model(model) inside it to upload the model and its preprocessing.
    """
    if not MODEL_NAME_PATTERN.match(model_name):
        raise FinkModelError(
            f"Invalid model name {model_name!r}: use lowercase letters, digits and dashes."
        )
    if "features" not in _state:
        raise FinkModelError("Load your training alerts with load_features() first.")
    import mlflow  # here, not at the top: it takes seconds, and fink-model check does not need it

    mlflow.set_experiment(model_name)
    # Parameters and training metrics of scikit-learn, XGBoost, LightGBM...
    # The model itself is logged by log_model(), with its preprocessing.
    mlflow.autolog(log_models=False, log_datasets=False, silent=True)

    with mlflow.start_run(run_name=model_name) as run:
        _state.update(model_name=model_name, run_id=run.info.run_id)
        features = _state["features"]
        mlflow.log_input(
            mlflow.data.from_pandas(features, source=_state["data_path"], name="training_alerts"),
            context="training",
        )
        mlflow.log_params(
            {
                "data_path": _state["data_path"],
                "data_files": _state["n_files"],
                "data_alerts": len(features),
                "feature_names": ",".join(features.columns),
            }
        )
        try:
            yield run
        finally:
            # log_model() only works inside this run
            del _state["model_name"], _state["run_id"]


def log_model(model):
    """Check the model against your preprocessing, then upload both to MLflow.

    Nothing is registered: register the run you want to deploy in the MLflow
    interface. Returns the id of the run.
    """
    import mlflow
    from mlflow.models import infer_signature
    run = mlflow.active_run()
    if run is None or run.info.run_id != _state.get("run_id"):
        raise FinkModelError("Call log_model() inside 'with start_run(\"my-model\"):'.")
    if _preprocessing_hash() != _state["preprocessing_hash"]:
        raise FinkModelError(
            "preprocessing/ changed since load_features(): run load_features() again "
            "and retrain, so the model is trained on the features that will be deployed."
        )

    # The features are those of load_features(): preprocessing/ did not change
    features = _state["features"]
    X = features.head(100).to_numpy()
    try:
        predictions = model.predict(X)
    except Exception as exc:
        raise FinkModelError(
            f"The model does not accept the features of preprocessing.py "
            f"({features.shape[1]} values: {', '.join(features.columns)}): {exc}"
        ) from exc
    if len(predictions) != len(X):
        raise FinkModelError(f"{len(predictions)} predictions for {len(X)} alerts.")

    model_name = _state["model_name"]
    mlflow.log_metric("preprocessing_ms_per_alert", _state["ms_per_alert"])

    # The model container serves this model: one row of FEATURE_NAMES per alert
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=infer_signature(X, predictions),
        input_example=X[:2],
        skops_trusted_types=_trusted_types(model),
    )

    # The preprocessing container is built from the files of this folder
    for path in _preprocessing_files():
        mlflow.log_artifact(str(path), artifact_path="preprocessing")

    # Read by the service to name the Docker images
    mlflow.set_tag("model_name", model_name)

    print(f"Uploaded {model_name} (run {run.info.run_id}). To deploy it, register this run in MLflow.")
    return run.info.run_id


def preprocessing():
    """Return your preprocessing module (preprocessing/preprocessing.py), reloaded
    so that your last edits are used."""
    sys.dont_write_bytecode = True
    if str(PREPROCESSING_DIR) not in sys.path:
        sys.path.insert(0, str(PREPROCESSING_DIR))
    import preprocessing

    return importlib.reload(preprocessing)


def check(path, limit=2000, n_shown=5):
    """Check the preprocessing on the alerts of path and print their first features."""
    X, info = load_features(path, limit=limit)
    names = preprocessing().FEATURE_NAMES
    print()
    print("objectId".ljust(14), *(name[:12].rjust(12) for name in names))
    for object_id, row in zip(info["objectId"][:n_shown], X[:n_shown]):
        print(str(object_id)[:14].ljust(14), *(f"{value:12.4g}" for value in row))
    print("\nOK: this preprocessing works in production.")


def contract_error(features, feature_names):
    """Return why features break the contract, or None if they respect it."""
    if not isinstance(features, list):
        return f"pre_processing() must return a list, not {type(features).__name__}"
    if len(features) != len(feature_names):
        return f"{len(features)} values for {len(feature_names)} names in FEATURE_NAMES"
    for name, value in zip(feature_names, features):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{name} is a {type(value).__name__}, not a number"
        if not math.isfinite(value):
            return f"{name} is NaN or infinite"
    return None


def _compute_features(prep, alerts):
    """Run pre_processing() on every alert. Return (X, mean ms per alert).

    alerts can be a generator: the alerts are not kept, only their features.
    """
    n_features = len(prep.FEATURE_NAMES)
    values = array("d")  # 8 bytes per value, instead of ~32 in a list of lists
    failures, n_alerts, seconds = [], 0, 0.0
    for alert in alerts:
        n_alerts += 1
        start = time.perf_counter()
        try:
            features = prep.pre_processing(alert)
        except Exception as exc:
            seconds += time.perf_counter() - start
            error = f"raised {type(exc).__name__}: {exc}"
        else:
            seconds += time.perf_counter() - start
            error = contract_error(features, prep.FEATURE_NAMES)
        if error:
            failures.append(f"{alert.get('objectId')} (candid {alert.get('candid')}): {error}")
        else:
            values.extend(features)

    if failures:
        raise FinkModelError(
            f"pre_processing() fails on {len(failures)} of {n_alerts} alerts, e.g.:\n  "
            + "\n  ".join(failures[:10])
        )
    ms_per_alert = 1000 * seconds / max(1, n_alerts)
    return np.frombuffer(values, dtype=float).reshape(-1, n_features), ms_per_alert


def _trusted_types(model):
    """Types that skops (used by MLflow to save scikit-learn models) refuses by
    default, e.g. the trees of a RandomForest. The model comes from your own
    session, so they are trusted."""
    try:
        import skops.io
    except ImportError:
        return None
    return skops.io.get_untrusted_types(data=skops.io.dumps(model)) or None


def _check_speed(ms_per_alert):
    if ms_per_alert > MAX_MS_PER_ALERT:
        raise FinkModelError(
            f"pre_processing() takes {ms_per_alert:.1f} ms per alert, the limit is "
            f"{MAX_MS_PER_ALERT} ms: it could not follow the stream. Avoid creating "
            "a pandas DataFrame or loading a file for each alert."
        )


def _preprocessing_files():
    # Not the subfolders: the service only copies the files next to preprocessing.py
    return sorted(path for path in PREPROCESSING_DIR.iterdir() if path.is_file())


def _preprocessing_hash():
    digest = hashlib.sha256()
    for path in _preprocessing_files():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
