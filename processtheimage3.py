import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Input: Path to the RGB image
image_path = r"D:\Gan & image processing module\gan_images2\epoch_1540_image_2.png"

# Output: Directory to save the images with background removed
output_dir = "background_removed_images"
os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

def apply_color_and_remove_background(image_path, color):
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Generate a binary mask (segmentation)
    _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Create a 3-channel version of the binary mask
    binary_mask_3d = cv2.merge([binary_mask, binary_mask, binary_mask])

    # Create the colored mask
    height, width, _ = image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(3):  # Loop through the R, G, B channels
        colored_image[:, :, i] = binary_mask * (color[i] / 255.0)

    # Combine the original image with the colored mask using weights
    result_image = cv2.addWeighted(image, 0.7, colored_image, 0.3, 0)

    # Add an alpha channel for transparency
    alpha_channel = binary_mask  # Use the binary mask as the alpha channel
    transparent_image = cv2.merge((*cv2.split(result_image), alpha_channel))

    # Save the resulting transparent image
    output_file_name = f"colored_image_bg_removed_{color[0]}_{color[1]}_{color[2]}.png"
    output_path = os.path.join(output_dir, output_file_name)
    cv2.imwrite(output_path, transparent_image)

    # Display the images: original and background-removed
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Background Removed (Transparent)")
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # Alpha not visible in matplotlib
    plt.axis("off")
    plt.show()

    print(f"Image with background removed successfully saved to: {output_path}")

# Example Usage: Apply a green color to the image and remove the background
desired_color = (0, 255, 0)  # Green
apply_color_and_remove_background(image_path, desired_color)