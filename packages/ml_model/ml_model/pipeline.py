from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

import logging
_logger = logging.getLogger(__name__)

ml_pipeline = Pipeline(
    [
        (
            'scaler',
            StandardScaler()
        ),
        (
            'svc',
            SVC()
        )
    ]
)
