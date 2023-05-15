import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import numpy as np
import os
import pandas as pd

with np.load('/home/florian/projet/r6.16/seq.npz') as f:
    X_2L = f['2L']
    X_2R = f['2R']
    X_3L = f['3L']
    X_3R = f['3R']
    X_4 = f['4']
    X_X = f['X']
    X_Y = f['Y']

class SequenceFeeder(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size, max_data=2**20, WINDOW=10000):
        self.x = x_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW

        self.max_data = max_data
        # n_data = min(len(self.x)-self.WINDOW, max_data)
        self.all_indices = np.arange(len(x_set) - WINDOW + 1)
        # self.indices = np.random.choice(self.indices, size=max_data, replace=False)
        # self.indices=np.clip(self.indices,self.WINDOW//2,len(self.x)-self.WINDOW//2 -1)
        # self.indices=np.unique(self.indices)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        window_indices = batch_indices.reshape(-1, 1) + np.arange(self.WINDOW).reshape(1, -1)
        batch_x = self.x[window_indices]
        return batch_x 
        
    def on_epoch_end(self):
        self.indices = np.random.choice(self.all_indices, size=self.max_data, replace=False)
        # np.random.shuffle(self.indices)

# model_name='new_cut_2001_KC_G'
# predictor = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

discriminator = keras.Sequential(
    [
        layers.Conv1D(64, kernel_size=(5), input_shape=(10000,4)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, kernel_size=(11)),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(1,activation="sigmoid"),
    ],
    name="discriminator",
)

latent_dim=10

# generator= tf.keras.models.Sequential([
#         keras.Input(shape=(latent_dim)),
#         tf.keras.layers.Dense(10, activation='relu'),
#         tf.keras.layers.Dense((40000), activation="relu"),
#         tf.keras.layers.Reshape((10000,4)),
#         tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
#         ])

generator = tf.keras.models.Sequential([
    keras.Input(shape=(latent_dim)),
    tf.keras.layers.Dense(200*4, activation='relu'),
    tf.keras.layers.Reshape((200,4)),
    tf.keras.layers.Conv1D(32, kernel_size=(5), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv1D(16, kernel_size=(11), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Conv1D(8, kernel_size=(19), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense((40000), activation="relu"),
    tf.keras.layers.Reshape((10000,4)),
    tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
    ])



class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.g_steps_per_d_step = 3

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_sequences):
        # Sample random points in the latent space
        batch_size = tf.shape(real_sequences)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode them to fake sequences
        generated_sequences = self.generator(random_latent_vectors)

        # Combine them with real sequences
        combined_sequences = tf.concat([generated_sequences, real_sequences], axis=0)

        # Assemble labels discriminating real from fake sequences
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # # Add random noise to the labels - important trick!
        # labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_sequences)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        for i in range(self.g_steps_per_d_step):
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Assemble labels that say "all real sequences"
            misleading_labels = tf.zeros((batch_size, 1))

            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                predictions = self.discriminator(self.generator(random_latent_vectors))
                g_loss = self.loss_fn(misleading_labels, predictions)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss":d_loss, "g_loss":g_loss}

# To limit the execution time, we only train on 100 batches. You can train on
# the entire dataset. You will need about 20 epochs to get nice results.
# Prepare the dataset. We use both the training & test MNIST digits.

batch_size = 256
x_train = SequenceFeeder(X_2L.astype('float32'), batch_size=batch_size, max_data=2**18)
# x_test = SequenceFeeder(X_2R.astype('float32'), batch_size=batch_size, max_data=2**5)

# all_digits = np.concatenate([x_train, x_test])

# dataset = tf.data.Dataset.from_tensor_slices(all_digits)
# dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0005),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    # loss_fn=keras.losses.MAE
    loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction='sum_over_batch_size')
)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='g_loss',patience=3, mode="max",restore_best_weights=True)

with tf.device('/GPU:0'):
    history=gan.fit(x_train, epochs=20)#,callbacks=[early_stop_callback])

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

# print(history.history)

model_name='NEW_3Gsteps_differentlearningrates'

os.chdir('/home/florian/projet/generators')
generator.save(model_name + '_G.h5')  # creates a HDF5 file 'my_model.h5'
discriminator.save(model_name + '_D.h5')
# save to csv: 
hist_csv_file = model_name+'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)