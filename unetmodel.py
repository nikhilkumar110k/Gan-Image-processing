from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization,Resizing
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

def sharpen_layer(x):
    kernel = tf.constant([[[[0, -1, 0]], [[-1, 5, -1]], [[0, -1, 0]]]], dtype=tf.float32)
    kernel = tf.repeat(kernel, repeats=x.shape[-1], axis=-1)   
    return tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding="SAME")

def unet_with_sharpening_fashion(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)  

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)  

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)  

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    u5 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4) 
    
    resized_c3 = Resizing(6, 6)(c3)
    u5 = concatenate([u5, resized_c3])

    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(u5)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = sharpen_layer(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)  

    resized_c2 = Resizing(12, 12)(c2)
    u6 = concatenate([u6, resized_c2])

    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    c6 = sharpen_layer(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)  

    resized_c1 = Resizing(24, 24)(c1)
    u7 = concatenate([u7, resized_c1])

    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    c7 = sharpen_layer(c7)

    outputs = Conv2D(3, (1, 1), activation='sigmoid')(c7)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


unet_fashion_mnist_model = unet_with_sharpening_fashion((28, 28, 1)) 
unet_fashion_mnist_model.summary()

def grayscale_to_rgb(grayscale_img, color_choice):
    color_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "yellow": [1, 1, 0],
        "cyan": [0, 1, 1],
        "magenta": [1, 0, 1]
    }
    
    color = np.array(color_map.get(color_choice, [1, 1, 1])) 
    rgb_img = np.stack([grayscale_img * color[i] for i in range(3)], axis=-1)
    
    return rgb_img


generator = load_model('D:/Gan & image processing module/generator_model')
discriminator = load_model('D:/Gan & image processing module/discriminator_model')
combined_model = load_model('D:/Gan & image processing module/combined_model_model')

unet_fashion_mnist_model = unet_with_sharpening_fashion((28, 28, 1))

def grayscale_to_rgb(grayscale_img, color_choice):
    color_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "yellow": [1, 1, 0],
        "cyan": [0, 1, 1],
        "magenta": [1, 0, 1]
    }
    
    color = np.array(color_map.get(color_choice, [1, 1, 1])) 
    rgb_img = np.stack([grayscale_img * color[i] for i in range(3)], axis=-1)
    
    return rgb_img

latent_dim = 100  
noise = np.random.randn(1, latent_dim)  
generated_img = generator.predict(noise)

generated_img = generated_img.reshape((28, 28, 1))

sharpened_colored_img = unet_fashion_mnist_model.predict(generated_img[np.newaxis, ...])

user_color = "blue"
recolored_img = grayscale_to_rgb(sharpened_colored_img[0, ..., 0], user_color)

plt.imshow(recolored_img)
plt.axis('off')
plt.show()

