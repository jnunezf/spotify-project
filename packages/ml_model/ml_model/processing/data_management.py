import joblib
import logging

from sklearn.pipeline import Pipeline

from ml_model.config import config
from ml_model import __version__ as _version

_logger = logging.getLogger(__name__)

def load_pipeline(*, filename: str) -> Pipeline:

    filepath = config.TRAINED_MODEL_DIR / filename
    saved_pipeline = joblib.load(filename=filepath)

    return saved_pipeline


def save_pipeline(*, pipeline_to_persist) -> None:

    filename = f'ml_model_output_v{_version}.pkl'

    path = config.TRAINED_MODEL_DIR / filename
    joblib.dump(pipeline_to_persist, path)
    _logger.info(f'\nsaved pipeline: {filename}')
