#import libraries
import os
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd


#generator weighted homebrew (new weights each batch) without Ns
class MyHbWeightedSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, max_data=2**20, WINDOW=501,bin=1000):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW

        self.max_data = max_data
        # bin_values, bin_edges = np.histogram(self.y, bins=1000)
        # bin_indices = np.digitize(self.y, bin_edges)
        # bin_indices[bin_indices == 1001] = 1000
        # bin_indices -= 1
        # self.weights = len(self.y) / (1000 * bin_values[bin_indices])
        n_data = min(len(self.x)-self.WINDOW+1, max_data)
        N_loc=np.sum(self.x, axis=1) == 0
        Bases_loc=np.convolve(N_loc, np.ones(self.WINDOW), "same")
        Bases_loc=np.asarray(Bases_loc,dtype="int")
        self.indices = np.arange(len(Bases_loc))
        #self.indices = self.indices[Bases_loc==0]
        # self.indices=np.unique(self.indices)
        # self.indices = np.random.choice(self.indices, size=max_data, replace=False)
        self.indices = self.indices[(Bases_loc==0) & (self.y!=0)]
        self.indices=np.clip(self.indices,self.WINDOW//2,len(self.x)-self.WINDOW//2 -1)
        self.indices=np.unique(self.indices)
        # self.indices=self.indices[:self.max_data:]
        self.on_epoch_end()
        # self.indices = np.random.choice(self.indices, size=self.max_data, replace=False)
        # # self.indices = np.arange(self.WINDOW//2, n_data+self.WINDOW//2)[Bases_loc==0]
        # np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        window_indices = batch_indices.reshape(-1, 1) + np.arange(-(self.WINDOW//2), self.WINDOW//2 + 1).reshape(1, -1)
        batch_x = self.x[window_indices]
        batch_y = self.y[batch_indices]
        # batch_weights = self.weights[batch_indices]
        bin_values, bin_edges = np.histogram(batch_y, bins=1000)
        bin_indices = np.digitize(batch_y, bin_edges)
        bin_indices[bin_indices == 1000+1] = 1000
        bin_indices -= 1
        batch_weights = batch_size / (1000 * bin_values[bin_indices])
        return batch_x, batch_y#, batch_weights
        
    def on_epoch_end(self):
        self.indices = np.random.choice(self.indices, size=self.max_data, replace=False)
        np.random.shuffle(self.indices)

#gen validation
class MyValidSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, max_data=2**20, WINDOW=501,bin=1000):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW

        self.max_data = max_data
        bin_values, bin_edges = np.histogram(self.y, bins=1000)
        bin_indices = np.digitize(self.y, bin_edges)
        bin_indices[bin_indices == 1000+1] = 1000
        bin_indices -= 1
        self.weights = len(self.y) / (1000 * bin_values[bin_indices])
        n_data = min(len(self.x)-self.WINDOW+1, max_data)
        N_loc=np.sum(self.x, axis=1) == 0
        Bases_loc=np.convolve(N_loc, np.ones(self.WINDOW), "same")
        Bases_loc=np.asarray(Bases_loc,dtype="int")
        self.indices = np.arange(len(Bases_loc))
        self.indices = self.indices[(Bases_loc==0) & (self.y!=0)]
        self.indices=np.clip(self.indices,self.WINDOW//2,len(self.x)-self.WINDOW//2 -1)
        self.indices=np.unique(self.indices)
        self.indices = np.random.choice(self.indices, size=self.max_data, replace=False)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        window_indices = batch_indices.reshape(-1, 1) + np.arange(-(self.WINDOW//2), self.WINDOW//2 + 1).reshape(1, -1)
        batch_x = self.x[window_indices]
        batch_y = self.y[batch_indices]
        batch_weights = self.weights[batch_indices]
        # bin_values, bin_edges = np.histogram(batch_y, bins=1000)
        # bin_indices = np.digitize(batch_y, bin_edges)
        # bin_indices[bin_indices == 1001] = 1000
        # bin_indices -= 1
        # batch_weights = batch_size / (1000 * bin_values[bin_indices])
        return batch_x, batch_y#, batch_weights


#cor_losses
import keras.backend as K
def mae_cor(y_true, y_pred):
    """Compute loss with Mean absolute error and correlation.
        :Example:
        >>> model.compile(optimizer = 'adam', losses = mae_cor)
        >>> load_model('file', custom_objects = {'mae_cor : mae_cor})
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))

    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))

    return (1 - cor) + mae

def mse_cor(y_true, y_pred):
    """Compute loss with Mean squared error and correlation.
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X*Y)
    sigma_X = K.sqrt(K.sum(X*X))
    sigma_Y = K.sqrt(K.sum(Y*Y))

    cor = sigma_XY/(sigma_X*sigma_Y + K.epsilon())
    mse = K.mean((y_true - y_pred)**2)

    cor_mse=(1 - cor) + mse
    #print(tf.math.is_nan(cor_mse))
    if tf.math.is_nan(cor_mse):
        print(X,Y)
        print('oui')
    #print(type(tf.math.is_nan(cor_mse)))

    return cor_mse

if __name__ == "__main__":
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=(5), activation='relu', input_shape=(501,4)),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(32, kernel_size=(11), activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Conv1D(16, kernel_size=(19), activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    model2.compile(loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['mae'],
        run_eagerly=True)

    #get all chr and their respective lenghts
    # X_2L=np.load('/home/florian/projet/r6.16/seq.npz')['2L']
    # X_2R=np.load('/home/florian/projet/r6.16/seq.npz')['2R']
    # X_3L=np.load('/home/florian/projet/r6.16/seq.npz')['3L']
    # X_3R=np.load('/home/florian/projet/r6.16/seq.npz')['3R']
    # X_4=np.load('/home/florian/projet/r6.16/seq.npz')['4']
    # X_X=np.load('/home/florian/projet/r6.16/seq.npz')['X']
    # X_Y=np.load('/home/florian/projet/r6.16/seq.npz')['Y']

    with np.load('/home/florian/projet/r6.16/seq.npz') as f:
        X_2L = f['2L']
        X_2R = f['2R']
        X_3L = f['3L']
        X_3R = f['3R']
        X_4 = f['4']
        X_X = f['X']
        X_Y = f['Y']

    #create scATAC values for each chr

    with np.load('/home/florian/projet/scATACseq_14chr.npz') as f:
        Y_2L=f['2L'][0]
        Y_2R=f['2R'][0]
        Y_3L=f['3L'][0]
        Y_3R=f['3R'][0]
        Y_4=f['4'][0]
        Y_X=f['X'][0]
        Y_Y=f['Y'][0]

    cut=500
    Y_2L[Y_2L >= cut] = cut
    Y_2L=Y_2L/cut

    Y_2R[Y_2R >= cut] = cut
    Y_2R=Y_2R/cut

    Y_3L[Y_3L >= cut] = cut
    Y_3L=Y_3L/cut

    Y_3R[Y_3R >= cut] = cut
    Y_3R=Y_3R/cut

    Y_4[Y_4 >= cut] = cut
    Y_4=Y_4/cut

    Y_X[Y_X >= cut] = cut
    Y_X=Y_X/cut

    Y_Y[Y_Y >= 80] = 80
    Y_Y=Y_Y/80

    #generates homebrew weighted values
    x=np.concatenate((X_2L,X_4,X_3R))
    y=np.concatenate((Y_2L,Y_4,Y_3R))
    x_valid=X_2R
    y_valid=Y_2R
    batch_size = 256
    gen = MyHbWeightedSequence(x, y, batch_size, max_data=2**20)
    gen_valid = MyValidSequence(x_valid, y_valid, batch_size, max_data=2**15)

    model_name='new_cut_weightless_501bp'

    dir='/home/florian/projet/models/' + model_name + '/'

    #training with checkpoint saving
    print(tf.config.list_physical_devices())
    with tf.device('/GPU:0'):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= dir+'cp.cpkt',
                                                         verbose=1)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
        # RL_callback=tf.keras.callbacks.ReduceLROnPlateau(patience=3)
        history=model2.fit(gen,validation_data=gen_valid,epochs=50,verbose=1, callbacks=[cp_callback,early_stop_callback])

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    os.chdir(dir)

    # save to csv: 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model2.save(model_name + '.h5')  # creates a HDF5 file 'my_model.h5'
    del model2