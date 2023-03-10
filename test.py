#import libraries
import os
import random as rd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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

    cor_mse=(1 - cor) + mse
    if tf.math.is_nan(cor_mse):
        print(X,Y)
        print('oui')

    return cor_mse

#create model
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=(5), activation='relu', input_shape=(2001,4)),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(32, kernel_size=(11), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Conv1D(32, kernel_size=(19), activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=(2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
    ])

model2.compile(loss=mse_cor,
    optimizer='adam',
    metrics=['mae'],
    run_eagerly=True)

#get all chr
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

# #create data set for prediction
# start, stop = 300_000, 310_000

# X_chr2L=[]
# for i in range(start,stop):
#     X_chr2L.append(X_2L[i-1000:i+1001])
# X_chr2L = np.array(X_chr2L)

# X_chr2R=[]
# for i in range(start,stop):
#     X_chr2R.append(X_2R[i-1000:i+1001])
# X_chr2R = np.array(X_chr2R)

# X_chr3L=[]
# for i in range(start,stop):
#     X_chr3L.append(X_3L[i-1000:i+1001])
# X_chr3L = np.array(X_chr3L)

# X_chr3R=[]
# for i in range(start,stop):
#     X_chr3R.append(X_3R[i-1000:i+1001])
# X_chr3R = np.array(X_chr3R)

# X_chr4=[]
# for i in range(start,stop):
#     X_chr4.append(X_5[i-1000:i+1001])
# X_chr4 = np.array(X_chr4)

# X_chrX=[]
# for i in range(start,stop):
#     X_chrX.append(X_X[i-1000:i+1001])
# X_chrX = np.array(X_chrX)

# X_chrY=[]
# for i in range(start,stop):
#     X_chrY.append(X_Y[i-1000:i+1001])
# X_chrY = np.array(X_chrY)

#generator for predictions without Ns
class MyPredSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, batch_size, WINDOW=2001):
        self.x = x_set
        self.batch_size = batch_size
        self.WINDOW = WINDOW
        self.indices = np.arange(len(self.x))
        self.indices=self.indices[self.WINDOW//2:len(self.x)-self.WINDOW//2 -1]

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        window_indices = batch_indices.reshape(-1, 1) + np.arange(-(self.WINDOW//2), self.WINDOW//2 + 1).reshape(1, -1)
        batch_x = self.x[window_indices]
        return batch_x


model2.load_weights('/home/florian/projet/models/test_KC1/cp.cpkt')

X_chr2L=MyPredSequence(X_2L,1024)
X_chr2R=MyPredSequence(X_2R,1024)
X_chr3L=MyPredSequence(X_3L,1024)
X_chr3R=MyPredSequence(X_3R,1024)
X_chr4=MyPredSequence(X_4,1024)
X_chrX=MyPredSequence(X_X,1024)
X_chrY=MyPredSequence(X_Y,1024)

preds={}
preds['pred2L']=model2.predict(X_chr2L,batch_size=1024)
preds['pred2R']=model2.predict(X_chr2R,batch_size=1024)
preds['pred3L']=model2.predict(X_chr3L,batch_size=1024)
preds['pred3R']=model2.predict(X_chr3R,batch_size=1024)
preds['pred4']=model2.predict(X_chr4,batch_size=1024)
preds['predX']=model2.predict(X_chrX,batch_size=1024)
preds['predY']=model2.predict(X_chrY,batch_size=1024)

os.chdir('/home/florian/projet/models')
np.savez_compressed('preds_test_KC1',**preds)

pred2L=np.load('/home/florian/projet/models/preds_test_KC1.npz')['pred2L']
pred2R=np.load('/home/florian/projet/models/preds_testKC1.npz')['pred2R']
pred3L=np.load('/home/florian/projet/models/preds_test_KC1.npz')['pred3L']
pred3R=np.load('/home/florian/projet/models/preds_test_KC1.npz')['pred3R']
pred4=np.load('/home/florian/projet/models/preds_test_KC1.npz')['pred4']
predX=np.load('/home/florian/projet/models/preds_test_KC1.npz')['predX']
predY=np.load('/home/florian/projet/models/preds_test_KC1.npz')['predY']

corr2L=np.corrcoef(Y_2L[1000:-1001],pred2L.ravel())[0][1]
corr2R=np.corrcoef(Y_2R[1000:-1001],pred2R.ravel())[0][1]
corr3L=np.corrcoef(Y_3L[1000:-1001],pred3L.ravel())[0][1]
corr3R=np.corrcoef(Y_3R[1000:-1001],pred3R.ravel())[0][1]
corr4=np.corrcoef(Y_4[1000:-1001],pred4.ravel())[0][1]
corrX=np.corrcoef(Y_X[1000:-1001],predX.ravel())[0][1]
corrY=np.corrcoef(Y_Y[1000:-1001],predY.ravel())[0][1]

print("correlation 2L", str(corr2L))
print("correlation 2R", str(corr2R))
print("correlation 3L", str(corr3L))
print("correlation 3R", str(corr3R))
print("correlation 4", str(corr4))
print("correlation X", str(corrX))
print("correlation Y", str(corrY))

