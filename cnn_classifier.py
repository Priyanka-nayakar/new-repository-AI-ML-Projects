# Simple CNN Image Classifier using Keras
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Dummy image dataset
X = np.random.rand(100, 28, 28, 1)
y = np.random.randint(0, 2, 100)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=16)

print("Model trained successfully!")
