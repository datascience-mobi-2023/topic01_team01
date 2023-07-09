from matplotlib import transforms
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import pandas as pd


Ordner = "Bilder\Training"
Pixelwerte = []
for dateiname in os.listdir(Ordner): 
    if dateiname.endswith("gif"): 
        bildpfad = os.path.join(Ordner, dateiname)
        bild = imageio.imread(bildpfad) 
        bild_pixelwerte = bild.flatten() 
        Pixelwerte.append(bild_pixelwerte)
Matrix = np.column_stack(Pixelwerte) 
print(Matrix) 
print(Matrix.shape) 
        
#Plot 
hist, edges = np.histogram(Matrix)
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
neue_matrix = Matrix
transformierte_Matrix = z_transformation(neue_matrix) 

# Convert all transformed values ​​to a flat array so I don't get 120 histograms output:
transformierte_werte = np.concatenate(transformierte_Matrix)
#transformierte_werte = transformierte_werte[~np.isnan(transformierte_werte)] #NaN-Werte (not a number) können auftreten, wenn sd=0, versucher ich noch zu beheben. Aber was genau sollen wir dann tun?

# Plot histogram for the transformed values
hist, edges = np.histogram(transformierte_werte)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixel values")
plt.ylabel("Frequency")
plt.title("Frequency of pixel values after transformation")
plt.show()
Mittelwert = np.mean(transformierte_werte)
Standardabweichung = np.std(transformierte_werte)
print("Mean of tansformed data : ", Mittelwert)
print("Standard deviation of performed data", Standardabweichung)
