"""
The whole notebook on a temporary MLflow: load_features -> start_run -> log_model.

Run:
    pytest test_fink_model.py
"""

import json

import mlflow
import pytest
from sklearn.tree import DecisionTreeClassifier

import fink_model
from load_alerts import FLAT_ALERT


@pytest.fixture
def alerts(tmp_path, monkeypatch):
    """A file of alerts, and an empty MLflow in a temporary folder."""
    monkeypatch.chdir(tmp_path)  # the artifacts go to ./mlruns
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    path = tmp_path / "alerts.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({**FLAT_ALERT, "lc_features_r": None, "candid": i, "magpsf": 15 + i / 10, "roid": 3 * (i % 2)})
            for i in range(40)
        )
    )
    return path


def test_model_is_uploaded_with_its_preprocessing(alerts):
    X, info = fink_model.load_features(alerts, columns=["roid"])
    y = (info["roid"] == 3).astype(int)
    model = DecisionTreeClassifier(random_state=0)
    with fink_model.start_run("test-model") as run:
        model.fit(X, y)
        run_id = fink_model.log_model(model)

    # What the model container does: one row of features in, one prediction out
    served = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    assert len(served.predict(X[:1])) == 1
    # What the preprocessing container is built from
    client = mlflow.MlflowClient()
    files = client.list_artifacts(run.info.run_id, "preprocessing")
    assert {file.path for file in files} == {"preprocessing/preprocessing.py", "preprocessing/requirements.txt"}
    # Registering is your decision, in the MLflow interface
    assert not client.search_registered_models()


def test_model_trained_on_other_features_is_refused(alerts):
    X, info = fink_model.load_features(alerts, columns=["roid"])
    model = DecisionTreeClassifier().fit(X[:, :3], info["roid"])
    with fink_model.start_run("test-model"), pytest.raises(fink_model.FinkModelError, match="does not accept"):
        fink_model.log_model(model)
