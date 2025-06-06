minor_project_2_skin_cancer_classification_using_cnn# -*- coding: utf-8 -*-
"""Minor_Project-2_Skin Cancer classification using CNN.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1V4PTW6CXuRLrNwJFduxrVmv4OasC7Jzo
"""
import tensorflow as tf
from tensorflow.keras import layers,models,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

train_dir="D:/chandu/Minor-2/Skin cancer ISIC The International Skin Imaging Collaboration/Train"
test_dir="D:/chandu/Minor-2/Skin cancer ISIC The International Skin Imaging Collaboration/Test"

image_height,image_width=150,150
batch_size=32

train_datagen= ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen=ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_height,image_width),
    class_mode='categorical',
    batch_size=32
    )

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=(image_height,image_width),
    class_mode='categorical',
    batch_size=32
)

class_names = list(train_generator.class_indices.keys())
print(f"Classes: {class_names}")

print(train_generator.class_indices)
print(test_generator.class_indices)

print(train_generator.samples)
print(test_generator.samples)

print("Entering Training.")
import sklearn
from sklearn.model_selection import StratifiedKFold

X = np.random.rand(100, 150, 150, 3)
y = np.random.randint(0, 7, 100)


n_splits = 5  # Number of folds
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold + 1}/{n_splits}")

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create data generators for this fold
    train_datagen_fold = ImageDataGenerator(
        rescale=1.0/255.0,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen_fold = ImageDataGenerator(rescale=1.0/255.0)

    model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(image_height,image_width,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
    ])
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    # Train the model on this fold
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=test_generator
    )

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
# prompt: save model as .keras zip file
# Assuming 'model' is your trained CNN
tf.keras.models.save_model(model,'skin_cancer_model.h5')
#save_model(model, 'skin_cancer_model.h5')  # Saves in HDF5 format
# OR (for newer TF versions)

