#import libraries
import os
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt
# import pyBigWig as pbg
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd


#cor_losses
import keras.backend as K
#generator weighted homebrew (new weights each batch) without Ns
class MyHbWeightedSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, max_data=2**20, WINDOW=2001):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW
        self.max_data = max_data
        # n_data = min(len(self.x)-self.WINDOW+1, max_data)
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
        bin_values, bin_edges = np.histogram(batch_y, bins=500)
        bin_indices = np.digitize(batch_y, bin_edges)
        bin_indices[bin_indices == 501] = 500
        bin_indices -= 1
        batch_weights = 1 / bin_values[bin_indices]
        # print(batch_x, batch_y, batch_weights)
        return batch_x, batch_y, batch_weights
        
    def on_epoch_end(self):
        self.indices = np.random.choice(self.indices, size=self.max_data, replace=False)
        np.random.shuffle(self.indices)

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
    mse = K.sqrt(K.mean(K.abs(y_true - y_pred)**2))

    cor_mse=(1 - cor) + mse
    #print(tf.math.is_nan(cor_mse))
    if tf.math.is_nan(cor_mse):
        print(X,Y)
        print('oui')
    #print(type(tf.math.is_nan(cor_mse)))

    return cor_mse

if __name__ == "__main__":
    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=(5), activation='relu', input_shape=(2001,4)),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(32, kernel_size=(11), activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Conv1D(32, kernel_size=(19), activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=(2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation="sigmoid")
        ])

    model2.compile(loss=mse_cor,
        optimizer='adam',
        metrics=['mae'],
        run_eagerly=True)

    #get all chr and their respective lenghts
    X_2L=np.load('/home/florian/projet/r6.16/seq.npz')['2L']
    X_2R=np.load('/home/florian/projet/r6.16/seq.npz')['2R']
    X_3L=np.load('/home/florian/projet/r6.16/seq.npz')['3L']
    X_3R=np.load('/home/florian/projet/r6.16/seq.npz')['3R']
    X_4=np.load('/home/florian/projet/r6.16/seq.npz')['4']
    X_X=np.load('/home/florian/projet/r6.16/seq.npz')['X']
    X_Y=np.load('/home/florian/projet/r6.16/seq.npz')['Y']

    #create scATAC values for each chr
    Y_2L=np.load('/home/florian/projet/scATACseq_KC.npz')['2L'][0]
    Y_2L[Y_2L >= 20] = 20
    Y_2L=Y_2L/20

    Y_2R=np.load('/home/florian/projet/scATACseq_KC.npz')['2R'][0]
    Y_2R[Y_2R >= 20] = 20
    Y_2R=Y_2R/20

    Y_3L=np.load('/home/florian/projet/scATACseq_KC.npz')['3L'][0]
    Y_3L[Y_3L >= 20] = 20
    Y_3L=Y_3L/20

    Y_3R=np.load('/home/florian/projet/scATACseq_KC.npz')['3R'][0]
    Y_3R[Y_3R >= 20] = 20
    Y_3R=Y_3R/20

    Y_4=np.load('/home/florian/projet/scATACseq_KC.npz')['4'][0]
    Y_4[Y_4 >= 20] = 20
    Y_4=Y_4/20

    Y_X=np.load('/home/florian/projet/scATACseq_KC.npz')['X'][0]
    Y_X[Y_X >= 20] = 20
    Y_X=Y_X/20

    Y_Y=np.load('/home/florian/projet/scATACseq_KC.npz')['Y'][0]
    Y_Y[Y_Y >= 20] = 20
    Y_Y=Y_Y/20

    #generates homebrew weighted values
    x=np.concatenate((X_2R,X_3L))
    y=np.concatenate((Y_2R,Y_3L))
    x_valid=X_3R
    y_valid=Y_3R
    batch_size = 2048
    gen = MyHbWeightedSequence(x, y, batch_size, max_data=2**22)
    gen_valid = MyHbWeightedSequence(x_valid, y_valid, batch_size, max_data=2**14)

    dir='/home/florian/projet/models/test_KC1/'

    #training with checkpoint saving
    print(tf.config.list_physical_devices())
    with tf.device('/GPU:0'):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= dir+'cp.cpkt',
                                                         save_weights_only=True,
                                                         verbose=1)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3,restore_best_weights=True)
        history=model2.fit(gen,validation_data=gen_valid,epochs=200,verbose=1, callbacks=[cp_callback,early_stop_callback])

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    os.chdir(dir)

    # save to json:  
    hist_json_file = 'history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)