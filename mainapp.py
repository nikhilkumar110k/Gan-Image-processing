import numpy as np
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

generator = load_model('generator_model.h5')

fashion_mnist_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

app = Flask(__name__)

def generate_image(label):
    if label not in fashion_mnist_labels:
        raise ValueError(f"Invalid label. Choose from: {fashion_mnist_labels}")

    latent_dim = 100  
    latent_vector = np.random.randn(1, latent_dim)  

    label_index = fashion_mnist_labels.index(label)
    label_vector = np.array([[label_index]]) 

    generated_img = generator.predict([latent_vector, label_vector])
    generated_img = (generated_img[0, :, :, 0] * 0.5 + 0.5)  
    return generated_img

@app.route('/')
def index():
    return render_template('index.html', labels=fashion_mnist_labels)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    label = data.get('label')

    try:
        generated_img = generate_image(label)

        fig, ax = plt.subplots()
        ax.imshow(generated_img)
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return jsonify({'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
