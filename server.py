
from flask import Flask, request, jsonify
import numpy as np
import pickle

from test import Setting

app = Flask(__name__)

global_model = Setting.create_cnn_gru_model(input_shape, num_classes)
global_weights = global_model.get_weights()

@app.route('/update_model', methods=['POST'])
def update_model():
    global global_weights
    data = request.get_json()

    # Extract client weights
    client_weights = pickle.loads(bytes.fromhex(data['weights']))

    # Average the weights (simple averaging for illustration)
    global_weights = [(global_w + client_w) / 2 for global_w, client_w in zip(global_weights, client_weights)]

    # Update the global model with new weights
    global_model.set_weights(global_weights)

    return jsonify({'message': 'Model updated successfully'})

@app.route('/get_model', methods=['GET'])
def get_model():
    global global_weights
    weights_hex = [w.tobytes().hex() for w in global_weights]
    return jsonify({'weights': weights_hex})

if __name__ == '__main__':
    app.run(port=5000)