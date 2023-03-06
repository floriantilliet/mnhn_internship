#import libraries
import os
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt
import pyBigWig as pbg
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.stats

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
    mse = K.sqrt(K.mean(K.abs(y_true - y_pred)**2))

    return (1 - cor) + mse

#get all chr and their respective lenghts
X_2L=np.load('/home/florian/projet/r6.16/seq.npz')['2L']
X_2R=np.load('/home/florian/projet/r6.16/seq.npz')['2R']
X_3L=np.load('/home/florian/projet/r6.16/seq.npz')['3L']
X_3R=np.load('/home/florian/projet/r6.16/seq.npz')['3R']
X_4=np.load('/home/florian/projet/r6.16/seq.npz')['4']
X_X=np.load('/home/florian/projet/r6.16/seq.npz')['X']
X_Y=np.load('/home/florian/projet/r6.16/seq.npz')['Y']

#create scATAC values for each chr
Y_2L=np.load('/home/florian/projet/scATACseq_14chr.npz')['2L'][0]
Y_2L[Y_2L >= 400] = 400
Y_2L=Y_2L/400

Y_2R=np.load('/home/florian/projet/scATACseq_14chr.npz')['2R'][0]
Y_2R[Y_2R >= 400] = 400
Y_2R=Y_2R/400

Y_3L=np.load('/home/florian/projet/scATACseq_14chr.npz')['3L'][0]
Y_3L[Y_3L >= 400] = 400
Y_3L=Y_3L/400

Y_3R=np.load('/home/florian/projet/scATACseq_14chr.npz')['3R'][0]
Y_3R[Y_3R >= 400] = 400
Y_3R=Y_3R/400

Y_4=np.load('/home/florian/projet/scATACseq_14chr.npz')['4'][0]
Y_4[Y_4 >= 400] = 400
Y_4=Y_4/400

Y_X=np.load('/home/florian/projet/scATACseq_14chr.npz')['X'][0]
Y_X[Y_X >= 400] = 400
Y_X=Y_X/400

Y_Y=np.load('/home/florian/projet/scATACseq_14chr.npz')['Y'][0]
Y_Y[Y_Y >= 400] = 400
Y_Y=Y_Y/400

#create binned scATAC values for each chr
Y_2L_bins=[]
for i in Y_2L[::10]:
    Y_2L_bins+=[i]*10
while len(Y_2L_bins)>len(X_2L):
    Y_2L_bins.pop()
Y_2L_bins=np.array(Y_2L_bins)

Y_2R_bins=[]
for i in Y_2R[::10]:
    Y_2R_bins+=[i]*10
while len(Y_2R_bins)>len(X_2R):
    Y_2R_bins.pop()
Y_2R_bins=np.array(Y_2R_bins)

Y_3L_bins=[]
for i in Y_3L[::10]:
    Y_3L_bins+=[i]*10
while len(Y_3L_bins)>len(X_3L):
    Y_3L_bins.pop()
Y_3L_bins=np.array(Y_3L_bins)

Y_3R_bins=[]
for i in Y_3R[::10]:
    Y_3R_bins+=[i]*10
while len(Y_3R_bins)>len(X_3R):
    Y_3R_bins.pop()
Y_3R_bins=np.array(Y_3R_bins)

Y_4_bins=[]
for i in Y_4[::10]:
    Y_4_bins+=[i]*10
while len(Y_4_bins)>len(X_4):
    Y_4_bins.pop()
Y_4_bins=np.array(Y_4_bins)

Y_X_bins=[]
for i in Y_X[::10]:
    Y_X_bins+=[i]*10
while len(Y_X_bins)>len(X_X):
    Y_X_bins.pop()
Y_X_bins=np.array(Y_X_bins)

Y_Y_bins=[]
for i in Y_Y[::10]:
    Y_Y_bins+=[i]*10
while len(Y_Y_bins)>len(X_Y):
    Y_Y_bins.pop()
Y_Y_bins=np.array(Y_Y_bins)


#create data set for prediction
start, stop = 350_000, 450_000

X_chr2L=[]
for i in range(start,stop):
    X_chr2L.append(X_2L[i-1000:i+1001])
X_chr2L = np.array(X_chr2L)

X_chr2R=[]
for i in range(start,stop):
    X_chr2R.append(X_2R[i-1000:i+1001])
X_chr2R = np.array(X_chr2R)

X_chr3L=[]
for i in range(start,stop):
    X_chr3L.append(X_3L[i-1000:i+1001])
X_chr3L = np.array(X_chr3L)

X_chr3R=[]
for i in range(start,stop):
    X_chr3R.append(X_3R[i-1000:i+1001])
X_chr3R = np.array(X_chr3R)

X_chr4=[]
for i in range(start,stop):
    X_chr4.append(X_4[i-1000:i+1001])
X_chr4 = np.array(X_chr4)

X_chrX=[]
for i in range(start,stop):
    X_chrX.append(X_X[i-1000:i+1001])
X_chrX = np.array(X_chrX)

X_chrY=[]
for i in range(start,stop):
    X_chrY.append(X_Y[i-1000:i+1001])
X_chrY = np.array(X_chrY)


model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(128, kernel_size=(6), activation='relu', input_shape=(2001,4)),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, kernel_size=(12), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(32, kernel_size=(12), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])
model2.compile(loss=mse_cor,
    optimizer='adam',
    metrics=['mae'])
model2.load_weights('/home/florian/projet/models/3_chr/cp.cpkt')


#pred for each dropout rate:
preds={}

preds['pred2L']=model2.predict(X_chr2L,batch_size=1024)
preds['pred2R']=model2.predict(X_chr2R,batch_size=1024)
preds['pred3L']=model2.predict(X_chr3L,batch_size=1024)
preds['pred3R']=model2.predict(X_chr3R,batch_size=1024)
preds['pred4']=model2.predict(X_chr4,batch_size=1024)
preds['predX']=model2.predict(X_chrX,batch_size=1024)
preds['predY']=model2.predict(X_chrY,batch_size=1024)

os.chdir('/home/florian/projet/models')
np.savez_compressed('preds_3chr',**preds)

