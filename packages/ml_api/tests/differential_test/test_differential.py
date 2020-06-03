import math
import pytest
from ml_model.config import config
from ml_model.predict import make_prediction
from ml_model.processing.data_management import load_dataset

@pytest.mark.differential
def test_model_prediction_differential(
        *,
        input_data_file:  str = 'previous_input_data.csv',
        output_data_file: str = 'previous_output_data.csv'
    ):

    previous_data_df = load_dataset(filename=output_data_file)
    previous_output_data = previous_data_df.predictions.values

    current_data_df = load_dataset(filename=input_data_file)
    current_data_json = current_data_df.to_json(orient='records')

    current_output_data_json = make_prediction(input_json=current_data_json)
    current_output_data = current_output_data_json.get('predictions')

    assert len(previous_output_data) == len(current_output_data)

    for previous_value, current_value in zip(previous_output_data, current_output_data):

        previous_value = previous_value.item()
        current_value = current_value.item()

        assert math.isclose(
            previous_value,
            current_value,
            rel_tol=config.ACCEPTABLE_MODEL_DIFFERENCE
        )
