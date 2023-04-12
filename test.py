#import libraries
import os
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.stats
from keras.models import load_model
from tensorflow import keras

model_name='new_cut_10LR'

model2 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

# get all chr
X_2L=np.load('/home/florian/projet/r6.16/seq.npz')['2L']
X_2R=np.load('/home/florian/projet/r6.16/seq.npz')['2R']
X_3L=np.load('/home/florian/projet/r6.16/seq.npz')['3L']
X_3R=np.load('/home/florian/projet/r6.16/seq.npz')['3R']
X_4=np.load('/home/florian/projet/r6.16/seq.npz')['4']
X_X=np.load('/home/florian/projet/r6.16/seq.npz')['X']
X_Y=np.load('/home/florian/projet/r6.16/seq.npz')['Y']

# X_2L=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['2L']
# X_2R=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['2R']
# X_3L=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['3L']
# X_3R=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['3R']
# X_4=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['4']
# X_X=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['X']
# X_Y=np.load('/home/florian/projet/r6.16/seq_reverse_complement.npz')['Y']

#generator for predictions
class MyPredSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size, WINDOW=501):
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

X_chr2L=MyPredSequence(X_2L,256)
X_chr2R=MyPredSequence(X_2R,256)
X_chr3L=MyPredSequence(X_3L,256)
X_chr3R=MyPredSequence(X_3R,256)
X_chr4=MyPredSequence(X_4,256)
X_chrX=MyPredSequence(X_X,256)
X_chrY=MyPredSequence(X_Y,256)

preds={}
preds['pred2L']=np.concatenate((np.zeros(25),model2.predict(X_chr2L,batch_size=256).ravel(),np.zeros(25)))
preds['pred2R']=np.concatenate((np.zeros(25),model2.predict(X_chr2R,batch_size=256).ravel(),np.zeros(25)))
preds['pred3L']=np.concatenate((np.zeros(25),model2.predict(X_chr3L,batch_size=256).ravel(),np.zeros(25)))
preds['pred3R']=np.concatenate((np.zeros(25),model2.predict(X_chr3R,batch_size=256).ravel(),np.zeros(25)))
preds['pred4']=np.concatenate((np.zeros(25),model2.predict(X_chr4,batch_size=256).ravel(),np.zeros(25)))
preds['predX']=np.concatenate((np.zeros(25),model2.predict(X_chrX,batch_size=256).ravel(),np.zeros(25)))
preds['predY']=np.concatenate((np.zeros(25),model2.predict(X_chrY,batch_size=256).ravel(),np.zeros(25)))

os.chdir('/home/florian/projet/models')
np.savez_compressed('preds_'+model_name,**preds)


