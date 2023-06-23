import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
import numpy as np
import os
import pandas as pd
from keras import backend
import keras.backend as k
import random as rd

with np.load('/home/florian/projet/r6.16/seq.npz') as f:
    X_2L = f['2L']
    X_2R = f['2R']
    X_3L = f['3L']
    X_3R = f['3R']
    X_4 = f['4']
    X_X = f['X']
    X_Y = f['Y']

pred_name='new_cut_2001_KC_G'
predictorKC = load_model('/home/florian/projet/models/'+ pred_name +'/'+ pred_name+ '.h5', compile=False)
pred_name='new_cut_2001_T1'
predictorT1 = load_model('/home/florian/projet/models/'+ pred_name +'/'+ pred_name+ '.h5', compile=False)

D={}
D['meanseq_entropy']=[]
D['mean_entropy']=[]
D['wasserstein']=[]
# D['peak1KC']=[]
# D['peak1T1']=[]
# D['peak2KC']=[]
# D['peak2T1']=[]

def sequence_entropy(seq):
    return(sum(np.sum(-seq*np.log(seq+k.epsilon()), axis=1)))

# implementation of wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

requested_floor_height=0.01
requested_peak_height=0.8  
KC_peak_loc=3000
T1_peak_loc=7000

def wasserstein_entropy_peak_loss(y_true, y_pred):
    seq_entropy=tf.math.reduce_sum(tf.math.reduce_sum(-y_pred*tf.math.log(y_pred+k.epsilon()),axis=2),axis=1)#entropie des seq
    meanseq=tf.math.reduce_sum(y_pred,axis=0)/y_true.shape[0]#séquence moyenne  

    predictions_peak1KC = predictorKC(y_pred[:,KC_peak_loc-1000:KC_peak_loc+1001])#prédictions signal ATACseq
    predictions_peak1T1 = predictorT1(y_pred[:,KC_peak_loc-1000:KC_peak_loc+1001])

    predictions_peak2KC = predictorKC(y_pred[:,T1_peak_loc-1000:T1_peak_loc+1001])#prédictions signal ATACseq
    predictions_peak2T1 = predictorT1(y_pred[:,T1_peak_loc-1000:T1_peak_loc+1001])

    D['wasserstein']+=[(14000-(tf.reduce_sum(y_true*tf.ones(y_true.shape)))).numpy()]
    D['mean_entropy']+=[(tf.reduce_sum(seq_entropy)/seq_entropy.shape[0]).numpy()]
    D['meanseq_entropy']+=[(tf.math.reduce_sum(tf.math.reduce_sum(-meanseq*tf.math.log(meanseq+k.epsilon()),axis=1),axis=0)).numpy()]
    
    # D['peak1KC']+=[(tf.math.abs(requested_peak_height-tf.math.reduce_sum(predictions_peak1KC)/predictions_peak1KC.shape[0])*7000).numpy()]
    # # D['peak1T1']+=[(tf.math.abs(requested_floor_height-tf.math.reduce_sum(predictions_peak1T1)/predictions_peak1T1.shape[0])*7000).numpy()]
    # D['peak1T1']+=[((requested_peak_height-tf.math.reduce_sum(predictions_peak1KC)/predictions_peak1KC.shape[0])*7000).numpy()]

    # D['peak2KC']+=[(tf.math.abs(requested_peak_height-tf.math.reduce_sum(predictions_peak2T1)/predictions_peak2T1.shape[0])*7000).numpy()]
    # # D['peak2T1']+=[(tf.math.abs(requested_floor_height-tf.math.reduce_sum(predictions_peak2KC)/predictions_peak2KC.shape[0])*7000).numpy()]
    # D['peak2T1']+=[((requested_peak_height-tf.math.reduce_sum(predictions_peak2T1)/predictions_peak2T1.shape[0])*7000).numpy()]

    return ((14000-(tf.reduce_sum(y_true*tf.ones(y_true.shape))/y_true.shape[0])*14000)
        +tf.reduce_sum(seq_entropy)/seq_entropy.shape[0]
        -tf.math.reduce_sum(tf.math.reduce_sum(-meanseq*tf.math.log(meanseq+k.epsilon()),axis=1),axis=0))
        # +(requested_peak_height-tf.math.reduce_sum(predictions_peak1KC)/predictions_peak1KC.shape[0])*7000
        # # +tf.math.abs(requested_peak_height-tf.math.reduce_sum(predictions_peak1KC)/predictions_peak1KC.shape[0])*7000
        # +tf.math.abs(requested_floor_height-tf.math.reduce_sum(predictions_peak1T1)/predictions_peak1T1.shape[0])*7000
        # +(requested_peak_height-tf.math.reduce_sum(predictions_peak2T1)/predictions_peak2T1.shape[0])*7000
        # # +tf.math.abs(requested_peak_height-tf.math.reduce_sum(predictions_peak2T1)/predictions_peak2T1.shape[0])*7000
        # +tf.math.abs(requested_floor_height-tf.math.reduce_sum(predictions_peak2KC)/predictions_peak2KC.shape[0])*7000)


class SequenceFeeder(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size, max_data=2**20, WINDOW=50000):
        self.x = x_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW
        self.max_data = max_data
        self.all_indices = np.arange(len(x_set) - WINDOW + 1)
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

# discriminator = keras.Sequential(
#     [
#         layers.Conv1D(64, kernel_size=(5), input_shape=(10000,4)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.Conv1D(32, kernel_size=(11)),
#         layers.LeakyReLU(alpha=0.2),
#         layers.GlobalMaxPooling1D(),
#         layers.Dense(1,activation="sigmoid"),
#         layers.Lambda(lambda x: 2*x-1)
#     ],
#     name="discriminator",
# )

discriminator = keras.Sequential(
    [
        layers.Conv1D(64, kernel_size=(5), input_shape=(50000,4)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(32, kernel_size=(9)),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv1D(16, kernel_size=(13)),
        layers.LeakyReLU(alpha=0.2),
        layers.GlobalMaxPooling1D(),
        layers.Dense(1,activation="sigmoid"),
        layers.Lambda(lambda x: 2*x-1)
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
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(16, kernel_size=(11), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Conv1D(8, kernel_size=(19), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense((200000), activation="relu"),
    tf.keras.layers.Reshape((50000,4)),
    tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
    ])

# generator = tf.keras.models.Sequential([
#     keras.Input(shape=(8000)),
#     # tf.keras.layers.Dense(200*4, activation='relu'),
#     tf.keras.layers.Reshape((2000,4)),
#     tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=(300), padding='valid', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=(112), padding='valid', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Conv1DTranspose(filters=16, kernel_size=(91), padding='valid', activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Reshape((10000,4)),
#     tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, axis=2))
#     ])

class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.g_steps_per_d_step = 1
        self.d_steps_per_g_step = 1

    # def call(self, inputs):
    #     return self.generator(inputs)

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(GAN, self).compile(run_eagerly=True)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def train_step(self, real_sequences):
        
        for i in range(self.g_steps_per_d_step):
            #generate a sample sequence
            # self.generator(tf.random.normal((1,10)))

            # Sample random points in the latent space
            batch_size = tf.shape(real_sequences)[0]
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Decode them to fake sequences
            generated_sequences = self.generator(random_latent_vectors)

            # Combine them with real sequences
            combined_sequences = tf.concat([generated_sequences, real_sequences], axis=0)

            # Assemble labels discriminating real from fake sequences, noise added
            labels = tf.concat(
                [-1*tf.ones((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
            )

            # # Add random noise to the labels - important trick!
            # labels += 0.1 * tf.random.uniform(tf.shape(labels))

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator(combined_sequences)
                d_loss = (1-self.d_loss_fn(labels, predictions))*14000
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
        )

        for i in range(self.g_steps_per_d_step):
            # Sample random points in the latent space
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

            # Assemble labels that say "all real sequences"
            misleading_labels = tf.ones((batch_size, 1))
            
            # Train the generator (note that we should *not* update the weights
            # of the discriminator)!
            with tf.GradientTape() as tape:
                generated_sequences = self.generator(random_latent_vectors)
                predictions = self.discriminator(generated_sequences)
                g_loss=self.g_loss_fn(predictions,generated_sequences)
            grads = tape.gradient(g_loss, self.generator.trainable_weights)
            self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"d_loss":d_loss, "g_loss":g_loss}



batch_size = 128
x_train = SequenceFeeder(np.concatenate((X_2L,X_2R,X_3L,X_3R,X_4,X_X)).astype('float32'), batch_size=batch_size, max_data=2**10)
# x_test = SequenceFeeder(X_2R.astype('float32'), batch_size=batch_size, max_data=2**5)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    # d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    # g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    d_optimizer=keras.optimizers.RMSprop(lr=0.0003),
    g_optimizer=keras.optimizers.RMSprop(lr=0.03),
    # loss_fn=keras.losses.MAE
    # loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction='sum_over_batch_size')
    d_loss_fn = wasserstein_loss,
    g_loss_fn= wasserstein_entropy_peak_loss
)

model_name='GAN_50k_allchr'

os.chdir('/home/florian/projet/generators/')

# os.mkdir(model_name)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='g_loss',patience=3, mode="max",restore_best_weights=True)
checkpoint= tf.keras.callbacks.ModelCheckpoint(filepath='/home/florian/projet/generators/'+model_name)

with tf.device('/GPU:0'):
    history=gan.fit(x_train, epochs=10000)#,callbacks=[checkpoint])

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    gen_losses = pd.DataFrame(D)

generator.save(model_name + '_G.h5')  # creates a HDF5 file 'my_model.h5'
discriminator.save(model_name + '_D.h5')

# save to csv: 
hist_csv_file = model_name+'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

gen_losses_csv_file = model_name+'gen_loss_history.csv'
with open(gen_losses_csv_file, mode='w') as f:
    gen_losses.to_csv(f)
