# server.py
from flask import Flask, render_template
from flask_socketio import SocketIO
import tensorflow as tf
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app)

# Create a simple neural network using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128, activation='relu', input_shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Training data
xs = np.array([1, 2, 3, 4], dtype=float)
ys = np.array([2, 4, 6, 8], dtype=float)

# Train the model
model.fit(xs, ys, epochs=100, verbose=0)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('predict')
def handle_predict(data):
    input_data = np.array([float(data['input'])])
    prediction = model.predict(input_data).flatten()[0]
    socketio.emit('prediction', {'prediction': prediction})

if __name__ == '__main__':
    socketio.run(app)
