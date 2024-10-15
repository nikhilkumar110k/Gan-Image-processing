import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

generator = load_model('generator_model.h5')

fashion_mnist_labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]




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

try:
    generated_img = generate_image("Ankle boot")  

    plt.imshow(generated_img)
    plt.axis('off')
    plt.show()
except Exception as e:
    print(f"Error: {e}")
