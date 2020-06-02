import pandas as pd
import numpy as np

from ml_model.predict import make_prediction


def test_make_single_prediction():

    test_data = np.random.randn(1, 15)
    df = pd.DataFrame(test_data)
    single_test_json = df.to_json(orient='records')

    subject = make_prediction(input_json=single_test_json)

    assert subject is not None
    assert isinstance(subject.get('predictions')[0], np.integer)
    assert subject.get('predictions')[0] >= 0
    assert subject.get('predictions')[0] < 7


def test_make_multiple_prediction():

    test_data = np.random.randn(50, 15)
    original_data_length = len(test_data)
    df = pd.DataFrame(test_data)
    multiple_test_json = df.to_json(orient='records')

    subject = make_prediction(input_json=multiple_test_json)

    assert subject is not None
    assert len(subject.get('predictions')) == 50
    assert len(subject.get('predictions')) == original_data_length
