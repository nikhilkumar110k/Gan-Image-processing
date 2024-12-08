import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input: Path to the RGB image
image_path = r"D:\Gan & image processing module\WhatsApp Image 2024-11-27 at 04.12.08_fa9aef12.jpg"

# Output: Directory to save the colored images
output_dir = "colored_images"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

def apply_color_to_image(image_path, color):

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    height, width, _ = image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(3):  # Loop through the R, G, B channels
        colored_image[:, :, i] = binary_mask * (color[i] / 255.0)

    # Combine the original image with the colored mask
    result_image = cv2.addWeighted(image, 0.7, colored_image, 0.3, 0)

    # Save the resulting colored image
    output_file_name = f"colored_image_{color[0]}_{color[1]}_{color[2]}.png"
    output_path = os.path.join(output_dir, output_file_name)
    cv2.imwrite(output_path, result_image)

    # Display the original image and the colored result side by side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Colored Image")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Colored image successfully saved to: {output_path}")

# Example Usage: Apply a red color to the image
desired_color = (0, 255, 0)  # Green
apply_color_to_image(image_path, desired_color)