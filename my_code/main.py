#%%
from matplotlib import transforms
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn import decomposition
from PIL import Image
from libs.packages.classes import z_transformation
from libs.packages.classes import plot_gallery

#path to pictures --> changes for everyone
Ordner = r"C:\Users\emili\Downloads\topic01_team01\Bilder\Training"

# list with pixel data
Pixelwerte = []
for dateiname in os.listdir(Ordner): # open data from 'Ordner'
    if dateiname.endswith("gif"): # open only pictures
        bildpfad = os.path.join(Ordner, dateiname)
        bild = imageio.imread(bildpfad) 
        bild_pixelwerte = bild.flatten() # flat list --> vector like
        Pixelwerte.append(bild_pixelwerte) # add pixel values of new picture to list
Matrix = np.column_stack(Pixelwerte) # add list to martix
print(Matrix) # Matrix with pixel values of all pictures, 1 picture = 1 column --> z-transformation over row 
print(Matrix.shape) # check if it worked:  77760 rows, 120 coloumns 
        
# histogram before z-transformation
hist, edges = np.histogram(Matrix)

plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelwerte")
plt.ylabel("Häufigkeit")
plt.title("Häufigkeit der Pixelwerte vor der Transformation")
plt.show()


## z-transformation

# definition for z_transformation in classes

neue_matrix = Matrix
transformierte_Matrix = z_transformation(neue_matrix) 

# transform into flar array  
transformierte_werte = np.concatenate(transformierte_Matrix)
# transformierte_werte = transformierte_werte[~np.isnan(transformierte_werte)] #NaN-Werte (not a number) when  sd=0

# histogram after z-transformation
hist, edges = np.histogram(transformierte_werte)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelwerte")
plt.ylabel("Häufigkeit")
plt.title("Häufigkeit der Pixelwerte nach der Transformation")
plt.show()

Mittelwert = np.mean(transformierte_werte)
Standardabweichung = np.std(transformierte_werte)
print(Mittelwert)
print(Standardabweichung)

marix_fertig = np.transpose(transformierte_Matrix)



## PCA

pca_estimator = decomposition.PCA(n_components=10, svd_solver="randomized", whiten=True) # number of PCs not sure
pca_estimator.fit(marix_fertig) 

# get transformed data
transformierte_Daten = pca_estimator.transform(marix_fertig)

# correlation matrix
korrelationsmatrix = np.corrcoef(transformierte_Daten, rowvar=False)
df = pd.DataFrame(data=korrelationsmatrix) 
# print(df)

# eignfaces
eigenfaces = pca_estimator.components_
plot_gallery(
    "Eigenfaces - PCA using randomized SVD", eigenfaces
    )

# %%
