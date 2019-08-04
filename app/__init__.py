from flask_cors import CORS
from flask import Flask, jsonify
from flask_swagger import swagger
from app.controller import digit_recognizer_handler
from keras.models import load_model
from config import MODEL_PATH

# add Cross-Origin-Resource-Sharing support

model = load_model(MODEL_PATH)
digit_recognizer_blueprint_cors = digit_recognizer_handler(model)
CORS(digit_recognizer_blueprint_cors)

#  Start the flask server
server = Flask(__name__)
server.register_blueprint(digit_recognizer_blueprint_cors)


@server.route("/", methods=['POST', 'GET'])
def spec():
    return jsonify(swagger(server))
