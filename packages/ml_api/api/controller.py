from flask import Blueprint, request, jsonify
from api.config import get_logger, UPLOAD_FOLDER

from api.validation import allowed_file
from werkzeug.utils import secure_filename

from ml_model.predict import make_prediction
from ml_model import __version__ as _version
from api import __version__ as api_version

from neural_network_model.predict import make_single_prediction
import os

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

@prediction_app.route('/predict/classifier', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        # Step 1: check if the post request has the file part
        if 'file' not in request.files:
            return jsonify('No file found'), 400

        file = request.files['file']

        # Step 2: Basic file extension validation
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            # Step 3: Save the file
            # Note, in production, this would require careful
            # validation, management and clean up.
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            _logger.debug(f'Inputs: {filename}')

            # Step 4: perform prediction
            result = make_single_prediction(
                image_name=filename,
                image_directory=UPLOAD_FOLDER)

            _logger.debug(f'Outputs: {result}')

        readable_predictions = result.get('readable_predictions')
        version = result.get('version')

        # Step 5: Return the response as JSON
        return jsonify(
            {'readable_predictions': readable_predictions[0],
             'version': version})
