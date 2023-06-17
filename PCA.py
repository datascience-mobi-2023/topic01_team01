from sklearn.decomposition import PCA
from Versuch_z_transformation import transformierte_werte
from Versuch_z_transformation import transformierte_Matrix
import numpy as np
import matplotlib.pyplot as plt

n_components = 20 #checked with scree plot (elobow method)
pca = PCA(n_components=n_components)

#Fit the model to the transformed data
pca.fit(transformierte_Matrix)

#Acess the principal components and the variance ratio
principal_components = pca.components_  
explained_variance_ratio = pca.explained_variance_ratio_  

#Transform data to lower-dimensional space
transformed_data = pca.transform(transformierte_Matrix)

#Check if it worked
# 1. Explained Variance Ratio
print("Explained variance ratio for each component:")
print(explained_variance_ratio)

# 2. Total Variance Explained
total_variance_explained = np.sum(explained_variance_ratio) #85%
print("Total variance explained:", total_variance_explained)
