# Fink model template

Train a model on [Fink](https://fink-broker.org) alerts and get it ready for production.

You provide two things:

| | Where | Role |
|---|---|---|
| **Preprocessing** | `preprocessing/preprocessing.py` | Turns one alert into the features of your model. Runs in production on every alert of the stream. |
| **Model** | `train.ipynb` | Trained on the output of this preprocessing. |

Parameters, metrics, training data, model and preprocessing are tracked in [MLflow](https://mlflow.fink-broker.org). From what you log, the [Fink AI service](https://github.com/Farid841/pre_processing-container-generator-from-mlflow) builds two containers: **preprocessing** and **model**.

```
Fink alerts ──► preprocessing container ──► model container ──► predictions
                 pre_processing(alert)        model.predict()
```

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Workflow](#workflow)
  1. [Get alerts](#1-get-alerts)
  2. [Write the preprocessing](#2-write-the-preprocessing)
  3. [Check it](#3-check-it)
  4. [Train and log the model](#4-train-and-log-the-model)
  5. [Deploy](#5-deploy)
- [The preprocessing contract](#the-preprocessing-contract)
- [Repository layout](#repository-layout)

## Requirements

- Python 3.10 or later (tested with 3.12)
- An account on https://mlflow.fink-broker.org
- Fink alerts, downloaded with [`fink_datatransfer`](https://github.com/astrolabsoftware/fink-client)

## Installation

```bash
git clone git@github.com:Farid841/model_template.git
cd model_template
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

### 1. Get alerts

`fink_datatransfer` writes a folder such as `ftransfer_ztf_2026-10-06_534834/`. Give its path to the tools below: nothing has to be copied. Files and folders of `.parquet`, `.avro` and `.jsonl` alerts are supported.

**The content selected in the data transfer is what `pre_processing()` receives**, during training and in production:

| Content selected | What an alert looks like |
|---|---|
| A selection of columns | Values at the top level: `magpsf`, `nalerthist`, `lc_features_g`... No `candidate`, no `prv_candidates`. **The preprocessing of this template reads these.** |
| Full alert | `candidate` (the detection) and `prv_candidates` (30 days of history). |

Select the same content when you run your model on Fink alerts: a preprocessing written for one does not work on the other.

### 2. Write the preprocessing

Edit `preprocessing/preprocessing.py`:

- `FEATURE_NAMES`: the names of your features, in order;
- `pre_processing(alert)`: takes one alert (a `dict`) and returns one float per name in `FEATURE_NAMES`.

The docstring of the file shows what an alert contains. Packages used by the preprocessing go in `preprocessing/requirements.txt`, with pinned versions.

### 3. Check it

```bash
python check.py <path-to-your-alerts>          # 2000 alerts, spread over all the files
python check.py <path-to-your-alerts> --all    # every alert
```

`check.py` runs `pre_processing()` on your alerts exactly as in production (every column of your files, except the cutouts), then prints `OK` or the reason of the failure:

| Check | Result |
|---|---|
| An alert raises, or gives a wrong number of values, a NaN, a string... | Refused, with the alerts concerned |
| More than **5 ms per alert** on average | Refused: the container could not follow the stream |
| Every feature has the same value for every alert | Refused: the preprocessing does not match your alerts. The columns of your alerts are listed |
| One feature has the same value for every alert | Warning: usually a wrong field name |

The same checks run with `pytest test_contract.py`, on built-in example alerts.

### 4. Train and log the model

Start `jupyter lab`, open `train.ipynb` and edit the marked cells: MLflow login, model name, alerts, labels, model. The notebook relies on three functions:

```python
X, info = fink_model.load_features(ALERTS, columns=["roid"])   # features + label columns
y = (info["roid"] == 3).astype(int)                             # your labels

with fink_model.start_run("my-model"):                          # everything is logged
    model.fit(X_train, y_train)
    fink_model.log_model(model)                                 # checked, then uploaded
```

| Function | What it does |
|---|---|
| `load_features()` | Computes the features with **your** `pre_processing()`: the model is trained on exactly what it receives in production. |
| `start_run()` | Opens the MLflow run. Logs the parameters and metrics of the model (scikit-learn, XGBoost, LightGBM...) and the origin of the training alerts. |
| `log_model()` | Refuses the model if it does not accept the features, or if `preprocessing/` changed since `load_features()`. Otherwise uploads the model and the preprocessing to the run. |

In production the model receives the output of `pre_processing()` and nothing else: a transformation of `X` (scaling, feature selection...) must be inside the model, with a scikit-learn `Pipeline`.

The model name uses lowercase letters, digits and dashes (`my-model`). It names the MLflow experiment and the two container images.

Every execution of the notebook creates a new run in the experiment. Compare them in the MLflow interface: nothing is registered or deployed at this stage.

### 5. Deploy

Choose the run to deploy and register it yourself: in the MLflow interface, open the run, then **Register model**, with the same name as the experiment. A run that is not registered is never deployed.

Building the two containers and running the model on Fink alerts is covered by the [Fink AI documentation](https://doc.ztf.fink-broker.org/services/fink_ai/).

## The preprocessing contract

1. **`pre_processing()` never fails.** Real alerts have missing and `None` fields: return a default value instead of raising.
2. **It is fast.** It runs on every alert of the stream: no pandas DataFrame, no file, no network call per alert.
3. **The order of `FEATURE_NAMES` is the input of the model.** After changing the list, run the notebook again to train a new model.
4. **Only the files of `preprocessing/` are uploaded, not its subfolders.** The container gets its `.py` files and `requirements.txt`: no data file, and never secrets.

## Repository layout

```
model_template/
├── preprocessing/
│   ├── preprocessing.py   your feature extraction (runs in production)
│   └── requirements.txt   packages needed by preprocessing.py
├── train.ipynb            template notebook: copy it and train your model
├── check.py               checks the preprocessing on your alerts
├── fink_model.py          the functions used by the notebook
├── load_alerts.py         reads your alerts like the service reads Kafka
├── test_contract.py       the same checks, with pytest
├── test_fink_model.py     the whole notebook, on a temporary MLflow
└── requirements.txt       packages needed on your machine
```
