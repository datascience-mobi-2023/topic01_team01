import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from PCA import transformed_data
from PCA import pca
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
pca = PCA(n_components=80) #best value shown in graph
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
precision_scores = []
recall_scores = []
f1_scores = []

for k in k_values:
    # Create and fit the KNN classifier on the transformed data
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance', algorithm='auto',metric='manhattan', leaf_size=30)
    knn.fit(transformed_data, y_train)

    # Predict the labels for the transformed test data
    transformed_test = pca.transform(X_test)
    y_pred = knn.predict(transformed_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
  
# Plot accuracy scores vs. k values
plt.plot(k_values, accuracy_scores, marker='o', label='Accuracy')
plt.plot(k_values, precision_scores, marker='o', label='Precision')
plt.plot(k_values, recall_scores, marker='o', label='Recall')
plt.plot(k_values, f1_scores, marker='o', label='F1-Score')
plt.xlabel('k')
plt.ylabel('Score')
plt.title('Performance Metrics vs. k')
plt.legend()
plt.show()

# Create and plot confusion matrix for k=7
k = 7
knn = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='auto', metric='manhattan', leaf_size=30)
knn.fit(transformed_data, y_train)
transformed_test = pca.transform(X_test)
y_pred = knn.predict(transformed_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy for k =", k, ":", accuracy)

# Calculate precision, recall, and F1 score
classification_report = classification_report(y_test, y_pred, target_names=target)
print("Classification Report for k =", k)
print(classification_report)

# Create and plot confusion matrix for k=7
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

#overall accuracy of the algorithm for k=7 is 76%
