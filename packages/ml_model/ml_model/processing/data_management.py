import joblib
import logging
import pandas as pd

from sklearn.pipeline import Pipeline

from ml_model.config import config
from ml_model import __version__ as _version

_logger = logging.getLogger(__name__)

def load_dataset(*, filename: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.DATASET_DIR}/{filename}", index_col=[0])
    return _data

def load_pipeline(*, filename: str) -> Pipeline:

    filepath = config.TRAINED_MODEL_DIR / filename
    saved_pipeline = joblib.load(filename=filepath)

    return saved_pipeline


def save_pipeline(*, pipeline_to_persist) -> None:

    filename = f'ml_model_output_v{_version}.pkl'
    path = config.TRAINED_MODEL_DIR / filename

    remove_old_pipelines(files_to_keep=filename)

    joblib.dump(pipeline_to_persist, path)
    _logger.info(f'\nsaved pipeline: {filename}')


def remove_old_pipelines(*, files_to_keep) -> None:

    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
