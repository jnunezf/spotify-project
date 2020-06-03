from flask import Blueprint, request, jsonify
from api.config import get_logger
from ml_model.predict import make_prediction
from ml_model import __version__ as _version
from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)

ml_app = Blueprint('ml_app', __name__)

@ml_app.route('/end_point', methods=['GET'])
def end_point():
    if request.method == 'GET':
        _logger.info('End-point status OK')
        return 'end-point-ok '

@ml_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        _logger.info('End-point version')
        return jsonify(
            {
                'model_version' : _version,
                'api_version' : api_version
            }
        )

@ml_app.route('/v1/predict/SVC', methods=['POST'])
def svc():
    if request.method == 'POST':
        _logger.info('End-point SVC')

        import json

        request_json = request.get_json()
        json_data = json.dumps(request.get_json())

        result = make_prediction(input_json=json_data)
        _logger.info(f'Result: {result}')

        predictions = result.get('predictions')
        version = result.get('version')

        response = jsonify(
            {
                'predictions' : predictions.tolist(),
                'version' : version
            }
        )



        return response
