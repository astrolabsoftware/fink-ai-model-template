# Model template

How to put your model in production. You bring two things:

- **your preprocessing**: the code that turns one alert into the features of your model;
- **your model**: trained in a notebook, on the output of this preprocessing.

Everything is tracked in MLflow (parameters, metrics, training data, model, preprocessing) https://mlflow.fink-broker.org , and the [service](https://github.com/Farid841/pre_processing-container-generator-from-mlflow) builds the two containers, **preprocessing** and **AI model**, from what you log.

```
model_template/
├── preprocessing/
│   ├── preprocessing.py   ← ✏️ your feature extraction (runs in production)
│   └── requirements.txt   ← ✏️ packages needed by preprocessing.py
├── train.ipynb            ← ✏️ template notebook: copy it and train your model
├── check.py               ← checks your preprocessing on your alerts
├── fink_model.py          ← the functions used by the notebook
├── load_alerts.py         ← reads your alerts like the service reads Kafka
├── test_contract.py       ← the same checks, with pytest
└── requirements.txt       ← packages needed on your machine
```

## 0. Install

Python 3.10 or later (tested with 3.12). Everything goes in a virtual environment, inside the repository:

```bash
git clone git@github.com:Farid841/model_template.git
cd model_template
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Get alerts

Download Fink alerts with [`fink_datatransfer`](https://github.com/astrolabsoftware/fink-client). It writes a folder like `ftransfer_ztf_2026-09-28_221528/`: you give its path, nothing to copy. Files or folders of `.parquet`, `.avro` and `.jsonl` alerts work too.

## 2. Write your preprocessing

Edit `preprocessing/preprocessing.py`:

- `FEATURE_NAMES`: the names of your features, in order;
- `pre_processing(alert)`: takes one alert (a `dict`) and returns one float per name in `FEATURE_NAMES`.

The docstring at the top of the file shows what an alert contains. If you need to install packages, add them to `preprocessing/requirements.txt`, with pinned versions.

Then check it on your alerts:

```bash
python check.py <PATH_TO_THE_FOLDER_OF_ALERTS>/ftransfer_ztf_xxxx_xx_xx_xxxxx
```

It runs `pre_processing()` on the alerts exactly as in production (only `objectId`, `candid`, `candidate` and `prv_candidates`, as sent by the Fink AI feed), and prints `OK` or what is wrong:

- an alert on which it fails, or gives a wrong number of values, NaN, a string...;
- too slow: more than **5 ms per alert** on average, it could not follow the stream;
- a feature with the same value for every alert (usually a wrong field name).

By default it reads 2000 alerts, spread over all the files; `--all` reads them all.

## 3. Train in the notebook

Run `jupyter lab` (with the virtual environment activated), open `train.ipynb` and change the cells marked ✏️: MLflow login, model name, your alerts, your labels, your model. The notebook uses three functions:

```python
X, info = fink_model.load_features(ALERTS, columns=["tnsclass"])   # features + label columns
with fink_model.start_run("my-model"):                             # everything is logged
    model.fit(X_train, y_train)
    fink_model.log_model(model)                                     # checked, then uploaded
```

- `load_features()` computes the features with **your** `pre_processing()`, so the model is trained on exactly what it will receive in production.
- `start_run()` logs the parameters and metrics of your model automatically (scikit-learn, XGBoost, LightGBM...), and where the training alerts come from.
- `log_model()` refuses the model if it does not accept the features, if the preprocessing is too slow, or if `preprocessing/` changed since `load_features()`. Otherwise it uploads the model and the preprocessing, and prints the version number.

Every run creates a new version in MLflow. Compare them in the MLflow interface: nothing is deployed yet.

## 4. Deploy

Building the two containers and running your model on Fink alerts is covered by the [Fink AI documentation](https://doc.ztf.fink-broker.org/services/fink_ai/).

## The rules

1. **Never let `pre_processing()` fail.** Real alerts have missing and `None` fields: return a default value instead of raising.
2. **Keep it fast.** It runs on every alert of the stream: no pandas DataFrame, no file, no network call per alert.
3. **Keep the order of `FEATURE_NAMES`.** If you change the list, run the notebook again: it trains and uploads a new version.
4. **The files of `preprocessing/` are uploaded, not its subfolders.** The container only gets its `.py` files and `requirements.txt`: no data file, and never secrets.
