import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

enhanced_image_path = r"gan_images2/epoch_1550_image_1.png"  # Path to the enhanced image

output_dir = "colored_enhanced_images"
os.makedirs(output_dir, exist_ok=True)

def apply_color_to_enhanced_image(image_path, color):
    
    enhanced_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if enhanced_image is None:
        print(f"Could not read enhanced image: {image_path}")
        return

    _, binary_mask = cv2.threshold(enhanced_image, 1, 255, cv2.THRESH_BINARY)

    colored_image = np.zeros((enhanced_image.shape[0], enhanced_image.shape[1], 3), dtype=np.uint8)

    for i in range(3): 
        colored_image[:, :, i] = binary_mask * (color[i] / 255.0)

    output_path = os.path.join(output_dir, f"colored_enhanced_image_{color}.png")
    cv2.imwrite(output_path, colored_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Enhanced Image (Grayscale)")
    plt.imshow(enhanced_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Colored Enhanced Image")
    plt.imshow(cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    print(f"Colored enhanced image saved to: {output_path}")


desired_color = (0, 0, 255)  
apply_color_to_enhanced_image(enhanced_image_path, desired_color)
