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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from test_data import transformierte_Daten_testdata 
from sklearn.metrics import confusion_matrix
import seaborn

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
# print(Matrix) # Matrix with pixel values of all pictures, 1 picture = 1 column --> z-transformation over row 
# print(Matrix.shape) # check if it worked:  77760 rows, 120 coloumns 


'''
# histogram before z-transformation
hist, edges = np.histogram(Matrix)

plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelvalues")
plt.ylabel("Frequency")
plt.title("Frequency of pixelvalues before z-transformation - training")
plt.show()
'''


## z-transformation

# definition for z_transformation in classes

neue_matrix = Matrix
transformierte_Matrix = z_transformation(neue_matrix) 

# transform into flar array  
transformierte_werte = np.concatenate(transformierte_Matrix)
# transformierte_werte = transformierte_werte[~np.isnan(transformierte_werte)] #NaN-Werte (not a number) when  sd=0

'''
# histogram after z-transformation
hist, edges = np.histogram(transformierte_werte)
plt.bar(edges[:-1], hist, width=0.9)
plt.xlabel("Pixelvalues")
plt.ylabel("Frequency")
plt.title("Frequency of pixelvalues after z-transformation")
plt.show()
'''

Mittelwert = np.mean(transformierte_werte)
Standardabweichung = np.std(transformierte_werte)
print("Mean after z-transformation:", Mittelwert)
print("sd after z-transformation:", Standardabweichung)
print()

marix_fertig = np.transpose(transformierte_Matrix)



## PCA

pca_estimator = decomposition.PCA(n_components=20, svd_solver="randomized", whiten=True) # number of PCs not sure
pca_estimator.fit(marix_fertig) 

# get transformed data
transformierte_Daten = pca_estimator.transform(marix_fertig)

# correlation matrix
korrelationsmatrix = np.corrcoef(transformierte_Daten, rowvar=False)
df = pd.DataFrame(data=korrelationsmatrix) 
# print("Correlationmatrix:", df)

# variance ratio
variance_ratio = pca_estimator.explained_variance_ratio_
sum_variance_ratio = np.sum(variance_ratio)
# print("varinace ratio:", variance_ratio)
# print()
# print("total varinace ratio:", sum_variance_ratio * 100,"%")

# eignfaces
eigenfaces = pca_estimator.components_
# plot_gallery(
    # "Eigenfaces - PCA using randomized SVD", eigenfaces
    # )



'''
# Visualisierung der kumulativen Varianz
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Amount of PCs')
plt.ylabel('Kumulative variance')
plt.title('Kumulative variance')
plt.show()
'''


## KNN

# Assume you have a target variable for classification
# Modify the code according to your specific target variable
target = ['subject01', 'subject02', 'subject03', 'subject04', 'subject05', 'subject06', 'subject07', 'subject08', 'subject09', 'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15']  # Target labels corresponding to each image




# Assume you have already split the data into training and test sets

# Modify the code according to your specific training and test sets
X_train_pca = transformierte_Daten[:120]  # Training data after PCA transformation (120 samples)
X_test_pca = transformierte_Daten_testdata [:45]  # Test data after PCA transformation
y_train = target * 8  # Target labels for training data (repeated 8 times for each subject)
y_test = target * 3  # Target labels for test data

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=50)  # You can adjust the number of neighbors

# Train the classifier on the training data
knn.fit(X_train_pca, y_train)

# Predict the labels for the test data
y_pred = knn.predict(X_test_pca)

# Evaluate the accuracy of the classifier
accuracy = knn.score(X_test_pca, y_test)
print("Accuracy:", accuracy)

y_true = y_test

con_matrix = confusion_matrix(y_true, y_pred)
seaborn.heatmap(con_matrix, xticklabels=target , yticklabels= target)

# problem: everytime we run the code again the confusion matrix and the accuracy changes - why????
# problem accuracy really low: 13% at highest
