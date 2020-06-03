from flask import Flask
from api.config import get_logger

_logger = get_logger(logger_name=__name__)


def create_ml_app(*, config_object) -> Flask:

    flask_app = Flask('ml_api')
    flask_app.config.from_object(config_object)

    from api.controller import ml_app
    flask_app.register_blueprint(ml_app)
    _logger.debug('Application instance created')

    return flask_app
