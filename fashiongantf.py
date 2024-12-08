import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Custom Image Dataset Loader class remains the same
class CustomImageDataset:
    def __init__(self, csv_file, img_dir, image_size=(64, 64), num_rows=None):
        self.data_frame = pd.read_csv(csv_file)
        if num_rows is not None:
            self.data_frame = self.data_frame.head(num_rows)
        self.img_dir = img_dir
        self.image_size = image_size

    def load_data(self):
        images = []
        for index, row in self.data_frame.iterrows():
            img_base = row['image']
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                temp_path = os.path.join(self.img_dir, img_base + ext)
                if os.path.exists(temp_path):
                    img_path = temp_path
                    break
            if img_path is None:
                raise FileNotFoundError(f"No image file found for {img_base}")

            try:
                image = Image.open(img_path).convert('RGB')
                image = image.resize(self.image_size)
                image = np.array(image) / 255.0 * 2 - 1  # Scale to [-1, 1]
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
                raise e

            images.append(image)

        return np.array(images)

# Generator Model
class Generator(Model):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        self.model = tf.keras.Sequential([
            layers.Dense(8 * 8 * 128, input_dim=latent_dim),
            layers.Reshape((8, 8, 128)),
            layers.UpSampling2D(),
            layers.Conv2D(128, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.UpSampling2D(),
            layers.Conv2D(64, kernel_size=3, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            
            layers.UpSampling2D(),
            layers.Conv2D(3, kernel_size=3, padding='same'),
            layers.Activation('tanh')
        ])

    def call(self, inputs):
        return self.model(inputs)

# Discriminator Model
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(64, 64, 3)),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.25),
            
            layers.Conv2D(512, kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

# GAN Model
class GAN(Model):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super(GAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.generator.latent_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images
            generated_images = self.generator(noise, training=True)

            # Get predictions
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            # Calculate losses
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            real_loss = self.loss_fn(tf.ones_like(real_output) * 0.9, real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            disc_loss = (real_loss + fake_loss) / 2

        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        return {"d_loss": disc_loss, "g_loss": gen_loss}

def save_generated_images(model, epoch, latent_dim=100, num_images=10, save_dir='gan_images2'):
    os.makedirs(save_dir, exist_ok=True)
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = model.generator(noise, training=False)
    
    # Rescale images from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2.0
    
    for i in range(num_images):
        img = generated_images[i].numpy()
        img = (img * 255).astype(np.uint8)
        img_path = os.path.join(save_dir, f'epoch_{epoch}_image_{i}.png')
        Image.fromarray(img).save(img_path)

def main():
    # Parameters
    LATENT_DIM = 100
    BATCH_SIZE = 64
    EPOCHS = 4000
    
    # Load and prepare data
    print("Current working directory:", os.getcwd())
    
    csv_path = 'archive/images.csv'
    img_dir = 'archive/images_original'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found at {img_dir}")

    # Create dataset
    dataset = CustomImageDataset(csv_file=csv_path, img_dir=img_dir, num_rows=500)
    images = dataset.load_data()
    
    # Create models
    generator = Generator(LATENT_DIM)
    discriminator = Discriminator()
    gan = GAN(generator, discriminator)

    # Compile models
    generator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
    gan.compile(
        g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn=tf.keras.losses.BinaryCrossentropy()
    )

    # Training loop
    dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(1000).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        for batch in dataset:
            metrics = gan.train_step(batch)
            print(f"d_loss: {metrics['d_loss']:.4f}, g_loss: {metrics['g_loss']:.4f}")
        
        if (epoch + 1) % 10 == 0:
            save_generated_images(gan, epoch + 1)

    # Save models
    generator.save_weights('generator_weights.h5')
    discriminator.save_weights('discriminator_weights.h5')

if __name__ == "__main__":
    main()