# Fink model template

Template repository to train a machine learning model on [Fink](https://fink-broker.org) alerts and prepare it for deployment on the Fink AI service.

## Overview

A model deployed on Fink is made of two components, which you provide:

| Component | Location | Description |
|---|---|---|
| Preprocessing | `preprocessing/preprocessing.py` | Converts one alert into the feature vector of the model. Executed in production on every alert of the stream. |
| Model | `train.ipynb` | Trained on the output of the preprocessing. |

Parameters, metrics, training data, model and preprocessing are tracked in [MLflow](https://mlflow.fink-broker.org). From a registered run, the [Fink AI service](https://github.com/Farid841/pre_processing-container-generator-from-mlflow) builds one container per component:

```
Fink alerts ──► preprocessing container ──► model container ──► predictions
                 pre_processing(alert)        model.predict()
```

The `fink_model` package shipped in this repository validates the preprocessing against the production constraints and logs everything the service needs.

## Requirements

- Python 3.10 or later (tested with 3.12)
- An account on https://mlflow.fink-broker.org
- Fink alerts (the training dataset), downloaded with [`fink_datatransfer`](https://github.com/astrolabsoftware/fink-client)

## Installation

```bash
git clone git@github.com:Farid841/model_template.git
cd model_template
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

This installs the dependencies, the `fink_model` package and the `fink-model` command in the virtual environment. The environment must be activated in every new terminal.

## Usage

### 1. Get alerts

`fink_datatransfer` writes a folder such as `ftransfer_ztf_2026-10-06_534834/`. Its path is passed directly to the tools below; no copy or conversion is needed. Files and folders of `.parquet`, `.avro` and `.jsonl` alerts are supported.

`pre_processing()` receives the content selected in the data transfer, during training and in production: select the same content for both. The preprocessing of this template expects a selection of columns (values at the top level), not full alerts (`candidate`, `prv_candidates`).

### 2. Write the preprocessing

Edit `preprocessing/preprocessing.py`:

- `FEATURE_NAMES`: the names of the features, in order;
- `pre_processing(alert)`: takes one alert (a `dict`) and returns one float per name in `FEATURE_NAMES`.

The docstring of the file describes the content of an alert. Packages required by the preprocessing are listed in `preprocessing/requirements.txt`, with pinned versions.

### 3. Check the preprocessing

```bash
fink-model check <path-to-alerts>          # 2000 alerts, sampled across all files
fink-model check <path-to-alerts> --all    # every alert
```

The command runs `pre_processing()` on the alerts as in production (every column of the files, except the cutouts), then prints `OK` or the reason of the failure:

| Condition | Result |
|---|---|
| An alert raises, or returns a wrong number of values, a NaN, a string... | Error, with the alerts concerned |
| More than 5 ms per alert on average | Error: the container cannot keep up with the stream |
| Every feature is constant across all alerts | Error: the preprocessing does not match the alerts. The available columns are listed |
| One feature is constant across all alerts | Warning: usually a wrong field name |

The command exits with a non-zero status on error.

### 4. Train and log the model

Start `jupyter lab`, open `train.ipynb` and edit the marked cells: MLflow login, model name, alerts, labels, model. The notebook relies on three functions:

```python
import fink_model

X, info = fink_model.load_features(ALERTS, columns=["roid"])   # features + label columns
y = (info["roid"] == 3).astype(int)                             # labels

with fink_model.start_run("my-model"):                          # MLflow run
    model.fit(X_train, y_train)
    fink_model.log_model(model)                                 # validated, then uploaded
```

| Function | Description |
|---|---|
| `load_features()` | Computes the features with `pre_processing()`, so that the model is trained on exactly what it receives in production. |
| `start_run()` | Opens the MLflow run. Logs the parameters and metrics of the model (scikit-learn, XGBoost, LightGBM...) and the origin of the training alerts. |
| `log_model()` | Rejects the model if it does not accept the features, or if `preprocessing/` changed since `load_features()`. Otherwise uploads the model and the preprocessing to the run. |

Notes:

- In production the model receives the output of `pre_processing()` and nothing else. Any transformation of `X` (scaling, feature selection...) must be part of the model, for instance with a scikit-learn `Pipeline`.
- The model name uses lowercase letters, digits and dashes (`my-model`). It names the MLflow experiment and the two container images.
- Every execution of the notebook creates a new run in the experiment. Nothing is registered or deployed at this stage.

### 5. Deploy

Deployment is triggered by registering a run. In the MLflow interface, open the run to deploy, then **Register model**, with the same name as the experiment. A run that is not registered is never deployed.

Building the containers and running the model on Fink alerts is covered by the [Fink AI documentation](https://doc.ztf.fink-broker.org/services/fink_ai/).

## Preprocessing contract

1. **`pre_processing()` never raises.** Real alerts have missing and `None` fields: return a default value.
2. **It is fast.** It runs on every alert of the stream: no pandas DataFrame, no file access, no network call per alert.
3. **The order of `FEATURE_NAMES` is the input of the model.** After changing the list, train a new model.
4. **Only the files of `preprocessing/` are uploaded, not its subfolders.** The container receives its `.py` files and `requirements.txt`: no data file, and never secrets.

The contract is verified again by the CI of the Fink AI service when the containers are built: sample alerts, complete and incomplete, are sent through the preprocessing container, then through the model container. The build fails if an alert does not give one finite feature vector of the expected length, or if the model does not accept it.

## Tests

`fink-model check` and the notebook are sufficient to validate a preprocessing. The test suite covers the tools of this repository and is only needed when modifying `fink_model/`:

```bash
pytest
```

## Repository layout

```
model_template/
├── preprocessing/
│   ├── preprocessing.py    feature extraction (to edit)
│   └── requirements.txt    dependencies of the preprocessing (to edit)
├── train.ipynb             training notebook (to edit)
├── fink_model/
│   ├── __init__.py         load_features(), start_run(), log_model()
│   ├── alerts.py           alert readers (.parquet, .avro, .jsonl)
│   └── cli.py              fink-model command
├── tests/
└── pyproject.toml          package metadata and dependencies
```
