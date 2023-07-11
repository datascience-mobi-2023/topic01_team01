#%%
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn
from matplotlib import transforms
from PIL import Image
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from libs.packages.classes import z_transformation
from libs.packages.classes import plot_gallery
from test_data import transformierte_Daten_testdata 



## Preparation

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
'''
plot_gallery(
    "Eigenfaces - PCA using randomized SVD", eigenfaces
     )
'''


'''
# Visualisierung der kumulativen Varianz
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Amount of PCs')
plt.ylabel('Kumulative variance')
plt.title('Kumulative variance')
plt.show()
'''



## KNN

target = ['subject01', 'subject02', 'subject03', 'subject04', 'subject05', 'subject06', 'subject07', 'subject08', 'subject09', 'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15']  # Target labels corresponding to each image

X_train_pca = transformierte_Daten[:120]  # Training data after PCA transformation (120 samples)
X_test_pca = transformierte_Daten_testdata [:45]  # Test data after PCA transformation
y_train = target * 8  # Target labels for training data (repeated 8 times for each subject)
y_test = target * 3  # Target labels for test data (repeated 3 times for each subject)

# Create a k-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=42)  # change number of k here

# Train the classifier on the training data
knn.fit(X_train_pca, y_train)

# Predict the labels for the test data
y_pred = knn.predict(X_test_pca)


# Evaluation
# accuracy 
accuracy = knn.score(X_test_pca, y_test)
print("Accuracy:", accuracy) #best: 1, worst: 0

#presicion
y_true = y_test
precision = precision_score(y_true, y_pred, average='macro') #change maybe to micro?
print('precision:', precision) #best: 1, worst: 0

#recall
recall = recall_score(y_true, y_pred, average='macro') #change maybe to micro?
print('recall:', recall) #best: 1, worst: 0

#f1
f1 = f1_score(y_true, y_pred, average='macro')
print('f1:', f1) #best: 1, worst: 0

# PROBLEM: everytime we run the code again the confusion matrix and the accuracy changes 
# PROBLEM: accuracy really low: 17% at highest
# PROBLEM: z-transformation and PCA of test set in different file: test_data.py
# PROBLEM: number of PCs cant be changed - it has to be 20 otherways there s a value error



## plotting

# confusion matrix 
cf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
seaborn.heatmap(cf_matrix, xticklabels=target , yticklabels=target)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# Plot: Accuracy depending on K
k_values = range(1, 121)  
accuracies_k = []  

plt.figure(figsize=(8, 6))

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    accuracy_k = knn.score(X_test_pca, y_test)
    accuracies_k.append(accuracy_k)

plt.plot(k_values, accuracies_k)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy depending on K')
plt.show()
#should it look like that??


# Plot: Accuracy depending on PC
PC_values = range(1, 21)  
accuracies_pc = []  

plt.figure(figsize=(8, 6))

for n in PC_values:
    pca_estimator = decomposition.PCA(n_components=n, svd_solver="randomized", whiten=True) # number of PCs not sure
    pca_estimator.fit(marix_fertig) 
    accuracy_pc = knn.score(X_test_pca, y_test)
    accuracies_pc.append(accuracy_pc)


plt.plot(PC_values, accuracies_pc)
plt.xlabel('PC')
plt.ylabel('Accuracy')
plt.title('Accuracy depending on PCs')
plt.show()
# Accuracy depending on PCs: just a horizontal line --> guess bc of our PC Problem
