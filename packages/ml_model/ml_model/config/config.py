import pathlib
import ml_model

PACKAGE_ROOT = pathlib.Path(ml_model.__file__).resolve().parent

TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'

DATASET_DIR = PACKAGE_ROOT / 'datasets'

ACCEPTABLE_MODEL_DIFFERENCE = 0.05
