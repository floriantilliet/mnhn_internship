import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import numpy as np
import os

model= tf.keras.models.Sequential([
        keras.Input(shape=(1)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense((8004), activation="relu"),
        tf.keras.layers.Reshape((2001,4)),
        tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
        ])

# Compilation du modèle avec l'optimiseur Adam et la perte binaire croisée
model.compile(optimizer='adam', loss='mae')

# Entraînement du modèle sur 100 epochs
with tf.device('/GPU:0'):
        model.fit(tf.random.normal((1,1)),tf.random.normal((1,1)), epochs=100)

os.chdir('/home/florian/projet/generators')
model.save('generator' + '.h5')  # creates a HDF5 file 'my_model.h5'
del model