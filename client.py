
import requests
import numpy as np
import pickle

from Model import Model

def train_and_send_model(client_id, local_model, X_train, y_train):
    # Train local model
    local_model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Get weights
    local_weights = local_model.get_weights()

    # Send weights to the server
    weights_hex = [w.tobytes().hex() for w in local_weights]
    print(pickle.dumps(local_weights).hex())
    response = requests.post('http://localhost:5000/update_model', json={'client_id': client_id, 'weights': pickle.dumps(local_weights).hex()})

    print(response.json())

def get_global_model():
    response = requests.get('http://localhost:5000/get_model')
    data = response.json()
    global_weights = [np.frombuffer(bytes.fromhex(w), dtype=np.float32) for w in data['weights']]
    #weights_hex = [w.tobytes().hex() for w in global_weights]
    #print(weights_hex)
    return global_weights

input_shape = (50, 1)  # Assuming time-series data with 50 timesteps and 1 feature
num_classes = 10

# Example usage for client-side
local_model = Model.create_cnn_gru_model(input_shape, num_classes)

# Simulated local training data
X_train_client = np.random.rand(100, 50, 1)
y_train_client = np.random.randint(0, num_classes, 100)

# Train and send model updates to server
train_and_send_model(client_id=1, local_model=local_model, X_train=X_train_client, y_train=y_train_client)

# Get global model from server
#global_weights = get_global_model()
#local_model.set_weights(global_weights)