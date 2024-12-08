import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Input path for the mask image
mask_path = r"D:\Gan & image processing module\enhanced_image_tf.png"  # Use the uploaded mask

# Directory for output colored images
output_dir = "colored_images"
os.makedirs(output_dir, exist_ok=True)

def apply_color_to_mask(mask_path, color):
    """
    Converts a grayscale mask to a colored image using the specified color.
    
    Args:
        mask_path (str): Path to the grayscale mask image.
        color (tuple): RGB values for the desired color (e.g., (255, 0, 0) for red).
    """
    # Load the grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask image: {mask_path}")
        return

    # Normalize the mask to binary (0 or 1)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    # Create a blank RGB image with the same dimensions
    colored_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Apply the specified color to the object in the mask
    for i in range(3):  # For R, G, B channels
        colored_image[:, :, i] = binary_mask * (color[i] / 255.0)

    # Save the colored image
    output_path = os.path.join(output_dir, f"colored_image_{color}.png")
    cv2.imwrite(output_path, colored_image)

    # Display the original mask and the colored result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Mask (Grayscale)")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Colored Image")
    plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Colored image saved to: {output_path}")


# Example: Apply red color to the mask
desired_color = (255, 0, 0)  # User input for color (Red)
apply_color_to_mask(mask_path, desired_color)
