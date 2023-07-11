import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os



folder = "Bilder\Training"
pixelValues = []
for filename in os.listdir(folder): 
    if filename.endswith("gif"): 
        imagePath = os.path.join(folder, filename)
        image = imageio.imread(imagePath) 
        image_pixelValues = image.flatten() 
        pixelValues.append(image_pixelValues)
matrix = np.column_stack(pixelValues) 
print(matrix) 
print(matrix.shape) 
        
#Plot 
hist, edges = np.histogram(matrix)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixel values")
plt.ylabel("Frequency")
plt.title("Frequency of pixel values ​​before transformation")
plt.show()

#Z transformation
def z_transformation(Matrix):
    transformed_matrix = []
    for row in Matrix:
        sd = np.std(row) 
        if sd == 0:
            tranformed_row = row - mean 
        else:
            mean = np.mean(row) 
            transformed_row = (row - mean) / sd 
        transformed_matrix.append(transformed_row) 
    return transformed_matrix
new_matrix = matrix
transformierte_Matrix = z_transformation(new_matrix) 

# Convert all transformed values ​​to a flat array so I don't get 120 histograms output:
transformed_values = np.concatenate(transformierte_Matrix)

# Plot histogram for the transformed values
hist, edges = np.histogram(transformed_values)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixel values")
plt.ylabel("Frequency")
plt.title("Frequency of pixel values after transformation")
plt.show()
average = np.mean(transformed_values)
standardDeviation = np.std(transformed_values)
print("Mean of tansformed data : ", average)
print("Standard deviation of performed data", standardDeviation)
