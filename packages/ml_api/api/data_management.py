import pandas as pd
from api import config

def load_dataset(*, filename: str) -> pd.DataFrame:
    _data = pd.read_csv(f"{config.PACKAGE_ROOT}/{filename}", index_col=[0])
    return _data
