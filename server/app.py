from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers
import requests

# Your n8n webhook URL (change if needed)
N8N_WEBHOOK_URL = "https://to7a3.app.n8n.cloud/webhook/f17e458d-9059-42c2-8d14-57acda06fc41"

# --- Custom Attention Layers ---
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(1, 7, padding='same', activation='sigmoid')
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.fc1 = layers.Dense(self.channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(self.channels, activation='sigmoid')
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis=[1, 2])
        gmp = tf.reduce_max(inputs, axis=[1, 2])
        gap_fc = self.fc2(self.fc1(gap))
        gmp_fc = self.fc2(self.fc1(gmp))
        attention = gap_fc + gmp_fc
        attention = tf.expand_dims(tf.expand_dims(attention, 1), 1)
        return inputs * attention

# --- Load Your Model ---
model = tf.keras.models.load_model(
    r"F:\ai ass\brain_tumor_model (1)\brain_tumor_model (1).keras",
    custom_objects={
        "ChannelAttention": ChannelAttention,
        "SpatialAttention": SpatialAttention
    }
)

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224)) / 255.0
    return np.expand_dims(image, axis=0)

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# --- Route ---
@app.route('/yolo', methods=['POST'])
def classify_image():
    if 'image_0' not in request.files:
        return jsonify({'error': 'No image found'}), 400

    file = request.files['image_0']
    image = preprocess_image(file.read())

    prediction = model.predict(image)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    idx = np.argmax(prediction[0])
    confidence = float(prediction[0][idx])
    label = class_names[idx]

    result = {
        'label': label,
        'confidence': confidence
    }

    # # Send result to n8n webhook
    try:
        n8n_response = requests.post(N8N_WEBHOOK_URL, json=result)
        n8n_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return jsonify({
            'error': 'Failed to send result to n8n',
            'details': str(e)
        }), 500

    return jsonify(result)

# --- Run Server ---
if __name__ == '__main__':
    app.run(port=5678)
