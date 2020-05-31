import pandas as pd
import logging

from ml_model.processing.data_management import load_pipeline
from ml_model import __version__ as _version

_logger = logging.getLogger(__name__)


pipeline_filename = f'ml_model_output_v{_version}.pkl'
_pipeline = load_pipeline(filename=pipeline_filename)

def make_prediction(*, input_json) -> dict:

    data = pd.read_json(input_json)
    prediction = _pipeline.predict(data)

    response = {
        'predictions' : prediction,
        'version' : _version
    }

    _logger.info(
        f"\nMaking predictions with model version: {_version} "
        f"\nInputs: {data} "
        f"\nPredictions: {prediction}"
    )

    return response
