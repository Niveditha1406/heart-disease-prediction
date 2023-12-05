import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage import io, color, transform
from sklearn.externals import joblib
def preprocess_image(image_path):
img = io.imread(image_path)
img = transform.resize(img, (100, 100)) # Resize images to a consistent size
img = color.rgb2gray(img) # Convert images to grayscale
return img.flatten()
data = []
labels = []
dataset_path = "path/to/dataset"
for class_name in os.listdir(dataset_path):
class_path = os.path.join(dataset_path, class_name)
for image_name in os.listdir(class_path):
image_path = os.path.join(class_path, image_name)
img_features = preprocess_image(image_path)
data.append(img_features)
labels.append(class_name)
20
data = np.array(data)
labels = np.array(labels)
# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
# Train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model, 'plant_species_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
# Load the trained model and label encoder
loaded_model = joblib.load('plant_species_model.pkl')
loaded_label_encoder = joblib.load('label_encoder.pkl')
21
# Preprocess a new image for prediction
new_image_path = "path/to/new/image.jpg"
new_image_features = preprocess_image(new_image_path)
# Make predictions
new_image_features = np.array([new_image_features])
prediction = loaded_model.predict(new_image_features)
# Decode the prediction
predicted_class = loaded_label_encoder.inverse_transform(prediction)[0]
print(f"The predicted plant species is: {predicted_class}")
