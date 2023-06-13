import tensorflow as tf
import numpy as np
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import math as mt
from tensorflow import keras
from keras import layers
import os
import keras.backend as k
import logomaker as lm

def ohd(seq):
    seq2=list()
    mapping = {"[1 0 0 0]":"A", "[0 1 0 0]":"C", "[0 0 1 0]":"G", "[0 0 0 1]":"T"}
    for i in seq:
        seq2.append(mapping[str(i)] if str(i) in mapping.keys() else "x")
    return ''.join(seq2)

def compute_gc_content(one_hot_sequence):
    gc_count = sum(base[1]+base[2] for base in one_hot_sequence)
    gc_content = gc_count / len(one_hot_sequence)
    return gc_content

def fast_pred(input,model):
    return model(tf.expand_dims(tf.cast(input,tf.float32),0))

def fast_pred_seq(input,window_start,model,size=2001):
    X=[]
    for i in range(window_start,window_start+size):
        X.append(np.array(fast_pred(input[i-(size//2):i+(size//2)+1],model))[0])
    return(X)

def get_max(array,n):
    return ((-array).argsort()[:n])

def get_min(array,n):
    return (array.argsort()[:n])

def mutation(window_start,chr, vals, size=2001):
    mut=np.copy(chr)
    for i in vals:
        if 0 < i <= size:
            mut[i+window_start]=np.roll(mut[i+window_start],1)
        elif 2001 < i <= 2*size:
            mut[i+window_start-size]=np.roll(mut[i+window_start-size],1)
        elif 4002 < i <= 3*size:
            mut[i+window_start-2*size]=np.roll(mut[i+window_start-2*size],1)
        else:
            mut[i+window_start-3*size]=np.roll(mut[i+window_start-3*size],1)
    return mut

def aimed_mutation(window_start,chr, vals,size=2001):
    mut=np.copy(chr)
    for i in vals:
        if 0 < i <= size:
            mut[i+window_start]=[1,0,0,0]
        elif 2001 < i <= 2*size:
            mut[i+window_start-size]=[0,1,0,0]
        elif 4002 < i <= 3*size:
            mut[i+window_start-2*size]=[0,0,1,0]
        else:
            mut[i+window_start-3*size]=[0,0,0,1]
    return mut

def compute_saliency_map(input_seq, model):
    # Compute the gradients of the output with respect to the input
    input_seq=tf.cast(input_seq,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        output = model(tf.expand_dims(input_seq,0))
    grads = tape.gradient(output, input_seq)
    

    # Compute the saliency map
    saliency_map = grads #tf.multiply(input_seq, grads)

    # Sum the saliency map over the channel dimension
    saliency_map = tf.reduce_sum(saliency_map, axis=-1)
    return saliency_map

def compute_saliency_channels(input_seq, model):
    # Compute the gradients of the output with respect to the input
    input_seq=tf.cast(input_seq,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        output = model(tf.expand_dims(input_seq,0))
    grads = tape.gradient(output, input_seq)

    # Compute the saliency map
    saliency_map = grads #tf.multiply(input_seq, grads)
    
    return saliency_map

def window_map(input,window_start,model,size=2001):
    model=model
    Y=np.zeros(size*2)
    for i in range (-(size//2),size//2):
        x=np.array(compute_saliency_map(tf.cast(input[window_start+i:window_start+size+i],tf.float32),model))
        y=np.concatenate((np.zeros(size//2+i),x,np.zeros(size//2+1-i)))
        Y=np.vstack([Y,y])
    return(Y)

def undifined_rate(seq):
    c=0
    for i in seq:
        if 0.2<max(i)<0.3:
            c+=1
    return(c/len(seq))   

def sequence_entropy(seq):
    return(sum(np.sum(-seq*np.log(seq+k.epsilon()), axis=1)))
    
def GC_content(seq):
    c=0
    for i in seq:
        if i == "G":
            c+=1
        elif i == "C":
            c+1
    return (round(c/len(seq)*100,2))

def force_ohe(seq):
    max=0
    seq2=np.copy(seq)
    for i in range(len(seq2)):
        new=[0,0,0,0]
        max=np.max(seq2[i])
        if max==0.25:
            new[np.random.randint(0,3)]=1
        else:
            new=np.where(seq2[i]==np.max(seq2[i]),1,0)
        seq2[i]=new
    return (seq2)

def random_ohe_sequence(length):
    return(np.eye(4)[np.random.randint(0,3,length)])

def fast_pred(input,model):
    return model(tf.cast(input,tf.float32))

def fast_pred_seq(input,window_start,model,size=2001):
    X=[]
    for i in range(window_start,window_start+size):
        X.append(np.array(fast_pred(input[:,i-(size//2):i+(size//2)+1],model))[0])
    return(X)

def fast_pred_whole_seq(input,model,size=2001):
    X=[]
    length=len(np.array(input[0]))
    for i in range(size//2,length-(size//2),10):
        X.append(np.array(fast_pred(input[:,i-(size//2):i+(size//2)+1],model))[0])
    return([0 for i in range(0,size//2,10)] + X +[0 for i in range(0,size//2,10)])


from numpy.lib.stride_tricks import as_strided
def sliding_window_view(x, window_shape, axis = None, *,
                        
                        subok=False, writeable=False):

    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    if axis is None:
        axis = tuple(range(x.ndim))
        if len(window_shape) != len(axis):
            raise ValueError(f'Since axis is `None`, must provide '
                             f'window_shape for all dimensions of `x`; '
                             f'got {len(window_shape)} window_shape elements '
                             f'and `x.ndim` is {x.ndim}.')
    else:
        axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
        if len(window_shape) != len(axis):
            raise ValueError(f'Must provide matching length window_shape and '
                             f'axis; got {len(window_shape)} window_shape '
                             f'elements and {len(axis)} axes elements.')

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)

def seq_similarity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Les séquences doivent avoir la même longueur.")
    
    hamming_distance = sum(a != b for a, b in zip(seq1, seq2))
    similarity = 1 - (hamming_distance / len(seq1))
    
    return similarity

with np.load('/home/florian/projet/r6.16/seq.npz') as f:
    X_2L = f['2L']
    X_3R = f['3R']

model_name='GAN_differential_peaks'
modelgen = load_model('/home/florian/projet/generators/'+model_name+'_G.h5', compile=False)
modeldis = load_model('/home/florian/projet/generators/'+model_name+'_D.h5', compile=False)

L1=[]
L2=[]
L3=[]
for i in range (100):
    a=modelgen.predict(tf.random.normal((1, 10)))
    b=modeldis.predict(a)
    L1.append(undifined_rate(a[0]))
    L2.append(b[0][0])
    # L3.append(GC_content(ohd(force_ohe(a[0]))))
mean_score=sum(L2)/len(L2)
mean_undifined_rate=sum(L1)/len(L1)
# mean_GC_content=sum(L3)/len(L3)

L2L=[]
L3R=[]
for i in [i for i in range(0,len(X_2L),len(X_2L)//100-100)]:
    L2L.append(modeldis.predict(sliding_window_view(X_2L, (10000, 4))[i])[0][0])
mean_score2L=sum(L2L)/len(L2L)

for i in [i for i in range(0,len(X_3R),len(X_3R)//100-100)]:
    L3R.append(modeldis.predict(sliding_window_view(X_3R, (10000, 4))[i])[0][0])
mean_score3R=sum(L3R)/len(L3R)

LRD=[]
for i in range(100):
    LRD.append(modeldis.predict(random_ohe_sequence(10_000).reshape(1,10000,4))[0][0])
mean_scoreRD=sum(LRD)/len(LRD)

seq_generated1 = modelgen.predict(tf.random.normal((1, 10)))
seq_generated2 = modelgen.predict(tf.random.normal((1, 10)))
seq_generated3 = modelgen.predict(tf.random.normal((1, 10)))
seq_generated4 = modelgen.predict(tf.random.normal((1, 10)))

df = pd.read_csv ('/home/florian/projet/generators/'+model_name+'history.csv')

#global plot:
start = 4975
stop= start+50

figs,axes = plt.subplots(6,1,figsize=(15,10))
lm.Logo(pd.DataFrame(np.array(seq_generated1).reshape(10_000,4)[start:stop], columns=["A", "C", "G", "T"]),ax=axes[2])
lm.Logo(pd.DataFrame(np.array(seq_generated2).reshape(10_000,4)[start:stop], columns=["A", "C", "G", "T"]),ax=axes[3])
lm.Logo(pd.DataFrame(np.array(seq_generated3).reshape(10_000,4)[start:stop], columns=["A", "C", "G", "T"]),ax=axes[4])
lm.Logo(pd.DataFrame(np.array(seq_generated4).reshape(10_000,4)[start:stop], columns=["A", "C", "G", "T"]),ax=axes[5])

# plt.subplot(3,4,1)
# plt.plot(df['g_loss'])
# plt.title("generator loss")
# plt.subplot(3,4,2)
# plt.plot(df['d_loss'],color='darkorange')
# plt.title("discriminator loss")
plt.subplot(3,2,1)
plt.plot(df['g_loss'],label="generator loss")
plt.plot(df['d_loss'],label='discriminator loss')
plt.legend()


plt.subplot(3,2,2)
plt.axis([0, 10, 0, 10])
plt.text(0,9,"model name: " + model_name)
plt.text(0,8,"list of parameters:")
plt.text(0,7,"Dloss: wasserstein, Gloss: entropy, optimizer: RMS, dLR=0.0003 gLR=0.03 ")
plt.text(0,6,"feeder info: batches of size 512 of random sequences among 2**10 per epoch on 2L")
plt.text(0,5,"discriminator output: custom sigmoid | g_steps_per_d_step = 1")
# plt.text(0,4,"mean rate of undifined bases: "+ str(round(mean_undifined_rate,4)) + "mean GC content: " + str(round(mean_GC_content,1))+" %")
plt.text(0,3,"mean score on generated sequences: " + str(round(mean_score,4)))
plt.text(0,2,"mean score on 2L sequences (training set): " + str(round(mean_score2L,4)))
plt.text(0,1,"mean score on 3R sequences (unknown set): " + str(round(mean_score3R,4)))
plt.text(0,0,"mean score on random sequences: " + str(round(mean_scoreRD,4)))
plt.axis("off")
plt.show()
plt.imsave("mnhn_internship/im1.png")
plt.close()

seq_generated = modelgen.predict(tf.random.normal((1, 10)))
lm.Logo(pd.DataFrame(np.array(seq_generated).reshape(10_000,4)[0:20], columns=["A", "C", "G", "T"]))

model_name='new_cut_2001_KC_G'
modelKC_G = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_KC_AB'
modelKC_AB = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_KC_ABp'
modelKC_ABp = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

model_name='new_cut_2001_T1'
modelT1 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_T2'
modelT2 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_T2a'
modelT2a = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_T3'
modelT3 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_T4'
modelT4 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_T1'
modelT5 = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

model_name='new_cut_2001_Subperineurialglia'
modelPerineurialglia = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)
model_name='new_cut_2001_Perineurialglia'
modelSubperineurialglia = load_model('/home/florian/projet/models/'+ model_name +'/'+ model_name+ '.h5', compile=False)

start1=1000

fast_mod1_T1=fast_pred_seq(seq_generated,start1,modelT1)
fast_mod1_T2=fast_pred_seq(seq_generated,start1,modelT2)
fast_mod1_T2a=fast_pred_seq(seq_generated,start1,modelT2a)
fast_mod1_T3=fast_pred_seq(seq_generated,start1,modelT3)
fast_mod1_T4=fast_pred_seq(seq_generated,start1,modelT4)
fast_mod1_T5=fast_pred_seq(seq_generated,start1,modelT5)
fast_mod1_KC_G=fast_pred_seq(seq_generated,start1,modelKC_G)
fast_mod1_KC_AB=fast_pred_seq(seq_generated,start1,modelKC_AB)
fast_mod1_KC_ABp=fast_pred_seq(seq_generated,start1,modelKC_ABp)
fast_mod1_Perineurialglia=fast_pred_seq(seq_generated,start1,modelPerineurialglia)
fast_mod1_Subperineurialglia=fast_pred_seq(seq_generated,start1,modelSubperineurialglia)

plt.figure(figsize=(15,10))

plt.subplot(6,2,1)
plt.plot(fast_mod1_KC_G[::10],label='KC_G',alpha=0.5)
plt.plot(pred100[start1:stop1:10],label='base_sequence')
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,2)
plt.plot(fast_mod1_T1[::10],label='T1',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,3)
plt.plot(fast_mod1_T2[::10],label='T2',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,4)
plt.plot(fast_mod1_T2a[::10],label='T2a',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,5)
plt.plot(fast_mod1_T3[::10],label='T3',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,6)
plt.plot(fast_mod1_T4[::10],label='T4',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,7)
plt.plot(fast_mod1_T5[::10],label='T5',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,8)
plt.plot(fast_mod1_KC_AB[::10],label='KC_AB',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,9)
plt.plot(fast_mod1_KC_ABp[::10],label='KC_ABp',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,10)
plt.plot(fast_mod1_Perineurialglia[::10],label='Perineurialglia',alpha=0.5)
plt.legend()
plt.ylim(0,1)

plt.subplot(6,2,11)
plt.plot(fast_mod1_Subperineurialglia[::10],label='Subperineurialglia',alpha=0.5)
plt.legend()
plt.ylim(0,1)
plt.show()
plt.imsave("mnhn_internship/im0.png")
plt.close()