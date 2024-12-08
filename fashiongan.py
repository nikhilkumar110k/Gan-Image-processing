from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization, Reshape, Conv2DTranspose, Conv2D, Flatten, Embedding, Multiply, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

latent_dim = 100  
num_classes = 10  
img_shape = (28, 28, 1)

def build_generator(latent_dim, num_classes):
    noise = Input(shape=(latent_dim,))
    
    label = Input(shape=(1,))
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    conditioned_noise = Multiply()([noise, label_embedding])
    
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    conditioned_noise = Multiply()([noise, label_embedding])
    
    x = Dense(128 * 7 * 7)(conditioned_noise)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Reshape((7, 7, 128))(x)

    x = Conv2DTranspose(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = Conv2DTranspose(1, kernel_size=3, padding="same", activation="tanh")(x)

    model = Model([noise, label], x)
    return model

def build_discriminator(img_shape, num_classes):
    img = Input(shape=img_shape)
    
    label = Input(shape=(1,))  
    
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))  
    
    label_embedding = Reshape(img_shape)(label_embedding)  

    concatenated = tf.concat([img, label_embedding], axis=-1) 

    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(concatenated)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    validity = Activation('sigmoid')(x)

    model = Model([img, label], validity)
    return model

discriminator = build_discriminator(img_shape, num_classes)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

generator = build_generator(latent_dim, num_classes)

z = Input(shape=(latent_dim,))
label = Input(shape=(1,))

img = generator([z, label])

discriminator.trainable = False

validity = discriminator([img, label])

combined_model = Model([z, label], validity)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

batch_size = 32
epochs = 30000
sample_period = 200

(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = (x_train / 127.5) - 1  
x_train = np.expand_dims(x_train, axis=-1) 

ones = np.ones((batch_size, 1))
zeros = np.zeros((batch_size, 1))

if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

def sample_images(epoch, generator):
    noise = np.random.randn(25, latent_dim)
    sampled_labels = np.array([num for _ in range(5) for num in range(5)])  

    gen_imgs = generator.predict([noise, sampled_labels])
    gen_imgs = 0.5 * gen_imgs + 0.5  

    fig, axs = plt.subplots(5, 5)
    count = 0
    for i in range(5):
        for j in range(5):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            count += 1
    fig.savefig("gan_images/%d.png" % epoch)
    plt.close()

for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs, real_labels = x_train[idx], y_train[idx]

    noise = np.random.randn(batch_size, latent_dim)

    fake_labels = np.random.randint(0, num_classes, batch_size)
    fake_imgs = generator.predict([noise, fake_labels])

    d_loss_real = discriminator.train_on_batch([real_imgs, real_labels], ones)
    d_loss_fake = discriminator.train_on_batch([fake_imgs, fake_labels], zeros)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.randn(batch_size, latent_dim)
    fake_labels = np.random.randint(0, num_classes, batch_size)
    g_loss = combined_model.train_on_batch([noise, fake_labels], ones)
    print(f"epoch: {epoch}/{epochs}, d_loss: {d_loss[0]}, g_loss: {g_loss}")

    if epoch % sample_period == 0:
        sample_images(epoch, generator)

generator.save('generator_model')
discriminator.save('discriminator_model')
<<<<<<< HEAD
combined_model.save('combined_model_model')
=======
combined_model.save('combined_model_model')
>>>>>>> 6fac32f190d094170cdf4c6863e6c8258b163948
