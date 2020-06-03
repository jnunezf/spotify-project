"""
This script should only be run in CI.
Never run it locally or you will disrupt the
differential test versioning logic.
"""
import pandas as pd
import numpy as np
from ml_model.predict import make_prediction

from api import config


def capture_previous_data(
        *,
        input_data_file:  str = 'previous_input_data.csv',
        output_data_file: str = 'previous_output_data.csv'
    ):

    input_data = np.random.randn(50, 15)
    input_data_df = pd.DataFrame(input_data)

    input_data_df.to_csv(
        #f'{config.PACKAGE_ROOT.parent}/ml_model/ml_model/datasets/{input_data_file}'
        f'{config.PACKAGE_ROOT}/{input_data_file}'
    )

    input_data_json = input_data_df.to_json(orient='records')
    output_data = make_prediction(input_json=input_data_json)

    output_data_df = pd.DataFrame(output_data)

    output_data_df.to_csv(
        #f'{config.PACKAGE_ROOT.parent}/ml_model/ml_model/datasets/{output_data_file}'
        f'{config.PACKAGE_ROOT}/{output_data_file}'
    )

if __name__ == '__main__':
    capture_previous_data()
