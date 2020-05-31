from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml_model import pipeline
from ml_model.processing.data_management import save_pipeline

from ml_model.config import config
from ml_model import __version__ as _version
import logging

_logger = logging.getLogger(__name__)


def run_training() -> None:

    X, y = make_classification(
        n_samples = 1000, n_features = 15, random_state=0, n_classes=7, n_clusters_per_class=3, n_informative = 10
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    pipeline.ml_pipeline.fit(X_train, y_train)

    _logger.info(f'\nsaving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.ml_pipeline)


if __name__ == '__main__':
    run_training()
