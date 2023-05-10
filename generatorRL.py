import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import numpy as np
import os

model_name='new_cut_2001_KC_G'
predictor = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

generator= tf.keras.models.Sequential([
        keras.Input(shape=(1)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense((8004), activation="relu"),
        tf.keras.layers.Reshape((2001,4)),
        tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
        ])


class RL(keras.Model):
    def __init__(self, predictor, generator,requested_peak):
        super(RL, self).__init__()
        self.predictor = predictor
        self.generator = generator
        self.requested_peak=requested_peak

    def compile(self, g_optimizer, loss_fn):
        super(RL, self).compile()
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self,requested_peak):

        with tf.GradientTape() as tape:
            predictions = self.predictor(self.generator(tf.random.normal((1, 1))))
            g_loss = self.loss_fn(self.requested_peak, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"g_loss": g_loss}


rl = RL(predictor=predictor, generator=generator,requested_peak=0.8)

rl.compile(g_optimizer=keras.optimizers.Adam(learning_rate=1),
         loss_fn=keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.SUM))

# To limit the execution time, we only train on 100 batches. You can train on
# the entire dataset. You will need about 20 epochs to get nice results.
rl.fit(tf.random.normal((1, 1)), epochs=50)

os.chdir('/home/florian/projet/generators')
rl.generator.save('generatorRL' + '.h5')  # creates a HDF5 file 'my_model.h5'

del generator, predictor