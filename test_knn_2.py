import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from PCA import transformed_data
from PCA import pca
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.decomposition import PCA

#Load the labels or target values for your training data
target = ['subject01', 'subject02', 'subject03', 'subject04', 'subject05', 'subject06', 'subject07', 'subject08', 'subject09', 'subject10', 'subject11', 'subject12', 'subject13', 'subject14', 'subject15']

# Load the training images
training_folder = "Bilder/Training"
training_images = []

for filename in os.listdir(training_folder):
    if filename.endswith(".gif") and filename != "Readme.txt":
        image_path = os.path.join(training_folder, filename)
        image = imageio.imread(image_path)
        training_images.append(image.flatten())

# Convert the training images to a numpy array
X_train = np.array(training_images[1:])  # Exclude the first image (Readme.txt)
y_train = np.repeat(target, 8)[:X_train.shape[0]]  # Repeat the target labels 8 times and select the first 120 labels

# Perform PCA on the training data
pca = PCA(n_components=80)
transformed_data = pca.fit_transform(X_train)

# Load the test images
test_folder = "Bilder/test"
test_images = []

for filename in os.listdir(test_folder):
    if filename.endswith(".gif"):
        image_path = os.path.join(test_folder, filename)
        image = imageio.imread(image_path)
        test_images.append(image.flatten())

# Convert the test images to a numpy array
X_test = np.array(test_images[:45])  # Select the first 45 test images
y_test = np.repeat(target, 3)[:45]  # Repeat the target labels 3 times and select the first 45 labels


# Perform KNN classification using the transformed data
# Try different value of k for optimal solution
k_values = [1, 3, 5, 7, 9]
accuracy_scores = []

for k in k_values:
    # Create and fit the KNN classifier on the transformed data
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance', algorithm='auto',metric='manhattan', leaf_size=30)
    knn.fit(transformed_data, y_train)

    # Predict the labels for the transformed test data
    transformed_test = pca.transform(X_test)
    y_pred = knn.predict(transformed_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
  
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)
    print("Overall Accuracy for k =", k, ":", overall_accuracy)


# Plot accuracy scores vs. k values
plt.plot(k_values, accuracy_scores, marker='o')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k')
plt.show()

# Create and plot confusion matrix for k=7
k = 7
knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='manhattan', leaf_size=30)
knn.fit(transformed_data, y_train)
transformed_test = pca.transform(X_test)
y_pred = knn.predict(transformed_test)
confusion = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', xticklabels=target, yticklabels=target)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (k={k})')
plt.show()

#accuracy = 75% - neighbors=7, weights='distance',algorithm='auto',metic='manhattan',leaf=30, pca=80 components
#accuracy = 71% - neighbors=7, weights='distance',algorithm='auto',metic='manhattan',leaf=30
#accuracy = 71% - neighbors=7, weights='distance',algorithm='auto',metic='manhattan',leaf=20
#accuracy = 71% - neighbors=7, weights='distance',algorithm='auto',metic='manhattan',leaf=10
#accuracy = 71% - neighbors=7, weights='distance',algorithm='brute',metic='manhattan',leaf=10
#optimal number of components = 80