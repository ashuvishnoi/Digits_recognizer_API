from flask import Blueprint, request, jsonify
from app.service import recognizer_service


digit_recognizer_blueprint = Blueprint('digits_recognizer_api',  __name__)


def digit_recognizer_handler(model):
    @digit_recognizer_blueprint.route("/recognizer/digits", methods=['POST'])
    def digits_handler():
        if request.is_json:
            response = recognizer_service(request.get_json(), model)
            return jsonify(response)
        else:
            return jsonify({'message': 'Input should be json.'})
