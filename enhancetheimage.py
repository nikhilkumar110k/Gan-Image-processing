import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the image
image = cv2.imread(r'D:\Gan & image processing module\WhatsApp Image 2024-11-27 at 04.12.08_fa9aef12.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load a super-resolution model from TensorFlow Hub
model_url = "https://tfhub.dev/captain-pool/esrgan-tf2/1"  # Replace with a valid model URL
model = hub.load(model_url)

# Prepare the image
image = tf.convert_to_tensor(image, dtype=tf.float32)
image = image[tf.newaxis, ...]

# Enhance the image
enhanced_image = model(image)

# Convert the enhanced image back to numpy and save
enhanced_image = enhanced_image.numpy().squeeze()
enhanced_image = np.clip(enhanced_image, 0, 255).astype('uint8')
enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

cv2.imwrite('D:/Gan & image processing module/enhanced_image_tf.png', enhanced_image)
