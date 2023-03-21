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


#generator weighted homebrew (new weights each batch) without Ns
class MyHbWeightedSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, max_data=2**20, WINDOW=2001):
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
        bin_indices[bin_indices == 1001] = 1000
        bin_indices -= 1
        batch_weights = batch_size / (1000 * bin_values[bin_indices])
        return batch_x, batch_y, batch_weights
        
    def on_epoch_end(self):
        self.indices = np.random.choice(self.indices, size=self.max_data, replace=False)
        np.random.shuffle(self.indices)

#gen validation
class MyValidSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, max_data=2**20, WINDOW=2001):
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
        # batch_weights = self.weights[batch_indices]
        bin_values, bin_edges = np.histogram(batch_y, bins=1000)
        bin_indices = np.digitize(batch_y, bin_edges)
        bin_indices[bin_indices == 1001] = 1000
        bin_indices -= 1
        batch_weights = batch_size / (1000 * bin_values[bin_indices])
        return batch_x, batch_y, batch_weights

#gen pred
class MyPredSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size, WINDOW=2001):
        self.x = x_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW
        self.indices = np.arange(len(self.x))
        self.indices=self.indices[self.WINDOW//2:len(self.x)-self.WINDOW//2 -1][::10]

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        window_indices = batch_indices.reshape(-1, 1) + np.arange(-(self.WINDOW//2), self.WINDOW//2 + 1).reshape(1, -1)
        batch_x = self.x[window_indices]
        return batch_x
    
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

    return cor_mse


#get all chr and their respective lenghts
X_2L=np.load('/home/florian/projet/r6.16/seq.npz')['2L']
X_2R=np.load('/home/florian/projet/r6.16/seq.npz')['2R']
X_3L=np.load('/home/florian/projet/r6.16/seq.npz')['3L']
X_3R=np.load('/home/florian/projet/r6.16/seq.npz')['3R']
X_4=np.load('/home/florian/projet/r6.16/seq.npz')['4']
X_X=np.load('/home/florian/projet/r6.16/seq.npz')['X']
X_Y=np.load('/home/florian/projet/r6.16/seq.npz')['Y']

os.chdir('/home/florian/projet/cell_types')
#fichiers=os.listdir('/home/florian/projet/cell_types')


fichiers=['scATACseq_T3.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_KC_AB.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_T1.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_Ensheathingglia.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_T2.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_KC_G.dedup.no_blacklist.RPGCnormalized.bw.npz',
 'scATACseq_KC_ABp.dedup.no_blacklist.RPGCnormalized.bw.npz']

for file in fichiers:


    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(64, kernel_size=(5), activation='relu', input_shape=(2001,4)),
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

    model2.compile(loss=mae_cor,
        optimizer='adam',
        metrics=['mae'],
        run_eagerly=True)


    #create scATAC values for each chr
    cut=40
    Y_2L=np.load('/home/florian/projet/cell_types/'+file)['2L'][0]
    Y_2L[Y_2L >= cut] = cut
    Y_2L=Y_2L/cut

    Y_2R=np.load('/home/florian/projet/cell_types/'+file)['2R'][0]
    Y_2R[Y_2R >= cut] = cut
    Y_2R=Y_2R/cut

    Y_3L=np.load('/home/florian/projet/cell_types/'+file)['3L'][0]
    Y_3L[Y_3L >= cut] = cut
    Y_3L=Y_3L/cut

    Y_3R=np.load('/home/florian/projet/cell_types/'+file)['3R'][0]
    Y_3R[Y_3R >= cut] = cut
    Y_3R=Y_3R/cut

    Y_4=np.load('/home/florian/projet/cell_types/'+file)['4'][0]
    Y_4[Y_4 >= cut] = cut
    Y_4=Y_4/cut

    Y_X=np.load('/home/florian/projet/cell_types/'+file)['X'][0]
    Y_X[Y_X >= cut] = cut
    Y_X=Y_X/cut

    Y_Y=np.load('/home/florian/projet/cell_types/'+file)['Y'][0]
    Y_Y[Y_Y >= 10] = 10
    Y_Y=Y_Y/10

    #generates homebrew weighted values
    x=np.concatenate((X_X,X_4,X_3L))
    y=np.concatenate((Y_X,Y_4,Y_3L))
    x_valid=X_3R
    y_valid=Y_3R
    batch_size = 1024
    gen = MyHbWeightedSequence(x, y, batch_size, max_data=2**20)
    gen_valid = MyValidSequence(x_valid, y_valid, batch_size, max_data=2**14)

    dir='/home/florian/projet/models/test_mae_{}/'.format(file)

    #training with checkpoint saving
    print(tf.config.list_physical_devices())
    with tf.device('/GPU:0'):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath= dir+'cp.cpkt',
                                                         save_weights_only=True,
                                                         verbose=1)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)
        history=model2.fit(gen,validation_data=gen_valid,epochs=20,verbose=1, callbacks=[cp_callback,early_stop_callback])

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    os.chdir(dir)

    # save to csv: 
    hist_csv_file = 'history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    model2.load_weights('/home/florian/projet/models/test_mae_{}/cp.cpkt'.format(file))

    X_chr2L=MyPredSequence(X_2L,2048)
    X_chr2R=MyPredSequence(X_2R,2048)
    X_chr3L=MyPredSequence(X_3L,2048)
    X_chr3R=MyPredSequence(X_3R,2048)
    X_chr4=MyPredSequence(X_4,2048)
    X_chrX=MyPredSequence(X_X,2048)
    X_chrY=MyPredSequence(X_Y,2048)

    preds={}
    preds['pred2L']=np.concatenate((np.zeros(100),model2.predict(X_chr2L,batch_size=2048).ravel(),np.zeros(100)))
    preds['pred2R']=np.concatenate((np.zeros(100),model2.predict(X_chr2R,batch_size=2048).ravel(),np.zeros(100)))
    preds['pred3L']=np.concatenate((np.zeros(100),model2.predict(X_chr3L,batch_size=2048).ravel(),np.zeros(100)))
    preds['pred3R']=np.concatenate((np.zeros(100),model2.predict(X_chr3R,batch_size=2048).ravel(),np.zeros(100)))
    preds['pred4']=np.concatenate((np.zeros(100),model2.predict(X_chr4,batch_size=2048).ravel(),np.zeros(100)))
    preds['predX']=np.concatenate((np.zeros(100),model2.predict(X_chrX,batch_size=2048).ravel(),np.zeros(100)))
    preds['predY']=np.concatenate((np.zeros(100),model2.predict(X_chrY,batch_size=2048).ravel(),np.zeros(100)))

    os.chdir('/home/florian/projet/models')
    np.savez_compressed('preds_test_mae_{}'.format(file),**preds)