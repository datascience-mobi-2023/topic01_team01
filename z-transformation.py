from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob


folder_path = 'Bilder/Training/'

# Load images and convert them to array 
def load_images(folder_path):
    images = []
    file_paths = glob.glob(folder_path + '*.gif')  # Retrieve all GIF files in the folder
    for file_path in file_paths:
        with Image.open(file_path) as img:
            images.append(np.array(img))
    return np.array(images)  

# Define the method for z-transformation
def z_transform(images):
    mean = np.mean(images, axis=(1,2))
    std = np.std(images, axis=(1, 2))
    transformed_images = (images - mean[:, np.newaxis, np.newaxis]) / std[:, np.newaxis, np.newaxis]
    return transformed_images

# Load images from the specified folder path
images = load_images(folder_path)
images = images.astype(np.float32)

# Apply Z-transformation to the images
z_transformed_images = z_transform(images)

# Calculate mean and standard deviation
mean_values = np.mean(z_transformed_images, axis=(1,2))
std_values = np.std(z_transformed_images, axis=(1,2))

# Display mean and standard deviation to test if z-transformation worked
for i in range(len(z_transformed_images)):
    print(f"Image {i+1}:")
    print("Mean:", mean_values[i])
    print("Standard Deviation:", std_values[i])
    print()

print("Overall Mean:", np.mean(mean_values))    #overall mean is 3.6302386e-10 -- near 0
print("Overall Standard Deviation:", np.mean(std_values)) # overall std is 1.0

# Function to display original and z-transformed images side by side
def visualize_images(original_images, transformed_images):
    num_images = 2  # Number of images to display
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 4 * num_images))

    for i in range(num_images):
        axes[i, 0].imshow(original_images[i], cmap='gray')
        axes[i, 0].set_title("Original Image")

        axes[i, 1].imshow(transformed_images[i], cmap='gray')
        axes[i, 1].set_title("Z-Transformed Image")

    plt.tight_layout()
    plt.show()

# Load original images
original_images = load_images(folder_path)

subset_indices = [0, 1]  # Positions of the images in the dataset array that you want to display.
subset_original_images = original_images[subset_indices]
subset_transformed_images = z_transformed_images[subset_indices]


# Visualize original and z-transformed images
visualize_images(subset_original_images, subset_transformed_images)

