from ml_model.config import config as model_config
from ml_model import __version__ as _version

from api import __version__ as api_version

import pandas as pd
import numpy as np
import json


def test_ep_returns_200(flask_test_client):

    response = flask_test_client.get('/end_point')
    assert response.status_code == 200


def test_version_ep_return_version(flask_test_client):

    response = flask_test_client.get('/version')
    assert response.status_code == 200
    response_json = json.loads(response.data)
    assert response_json['model_version'] == _version
    assert response_json['api_version'] == api_version


def test_svc_ep_return_svc(flask_test_client):

    test_data = np.random.randn(50, 15)
    df = pd.DataFrame(test_data)

    multiple_test_json = json.loads(df.to_json(orient='records'))

    response = flask_test_client.post('/v1/predict/SVC', json=multiple_test_json)

    assert response.status_code == 200

    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert len(response_json['predictions']) == 50
    assert response_version == _version
