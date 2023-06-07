import numpy as np
import tensorflow as tf
import keras.backend as k
from numpy.lib.stride_tricks import as_strided

#general functions
def get_max(array,n):
    return ((-array).argsort()[:n])

def get_min(array,n):
    return (array.argsort()[:n])

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

def smooth(x,window):
    a=np.convolve(x, np.ones(window)/window, mode='valid')
    return(np.concatenate((np.zeros((window)//2),a,np.zeros((window)//2-1))))

#DNA sequence related functions:
def ohd(seq):
    """
    Applies one hot decoding to a DNA sequence.

    Parameters
    ----------
    seq: np.ndarray, shape=(n,4)
        2D-array, a one hot encoded DNA sequence

    Returns
    -------
    seq_decoded: string,
        corresponding DNA sequence as a string

    See also
    --------
    ohe : one hot encoding function
    force_ohe : force one hot encoding function

    Notes
    -----
    """
    seq_decoded=list()
    mapping = {"[1 0 0 0]":"A", "[0 1 0 0]":"C", "[0 0 1 0]":"G", "[0 0 0 1]":"T"}
    for i in seq:
        seq_decoded.append(mapping[str(i)] if str(i) in mapping.keys() else "x")
    return ''.join(seq_decoded)

def ohe(seq):
    seq2=list()
    mapping = {"A":[1, 0, 0, 0],"C": [0, 1, 0, 0],"G":[0, 0, 1, 0],"T":[0, 0, 0, 1]}
    for i in seq:
      seq2.append(mapping[i.upper()] if i.upper() in mapping.keys() else [0, 0, 0, 0]) 
    return np.array(seq2)

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

def GC_content(seq):
    c=0
    for i in seq:
        if i == "G":
            c+=1
        elif i == "C":
            c+1
    return (round(c/len(seq)*100,2))

def sequence_entropy(seq):
    return(sum(np.sum(-seq*np.log(seq+k.epsilon()), axis=1)))

def seq_similarity(seq1, seq2):
    if len(seq1) != len(seq2):
        raise ValueError("Les séquences doivent avoir la même longueur.")
    
    hamming_distance = sum(a != b for a, b in zip(seq1, seq2))
    similarity = 1 - (hamming_distance / len(seq1))
    
    return similarity

def undifined_rate(seq):
    c=0
    for i in seq:
        if 0.2<max(i)<0.3:
            c+=1
    return(c/len(seq))   

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

#Sequence prediction related functions 
def fast_pred(input,model):
    return model(tf.expand_dims(tf.cast(input,tf.float32),0))

def fast_pred_seq(input,window_start,model,size=2001):
    X=[]
    for i in range(window_start,window_start+size):
        X.append(np.array(fast_pred(input[i-(size//2):i+(size//2)+1],model))[0])
    return(X)

def fast_pred_whole_seq(input,model,size=2001):
    X=[]
    length=len(np.array(input[0]))
    for i in range(size//2,length-(size//2),10):
        X.append(np.array(fast_pred(input[:,i-(size//2):i+(size//2)+1],model))[0])
    return([0 for i in range(0,size//2,10)] + X +[0 for i in range(0,size//2,10)])

#Saliency maps related functions
def compute_saliency_map(input_seq, model):
    input_seq=tf.cast(input_seq,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        output = model(tf.expand_dims(input_seq,0))
    grads = tape.gradient(output, input_seq)
    
    saliency_map = grads

    saliency_map = tf.reduce_sum(saliency_map, axis=-1)
    return saliency_map

def compute_saliency_channels(input_seq, model):
    input_seq=tf.cast(input_seq,tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_seq)
        output = model(tf.expand_dims(input_seq,0))
    grads = tape.gradient(output, input_seq)

    saliency_map = grads
    
    return saliency_map

def window_map(input,window_start,model,size=2001):
    model=model
    Y=np.zeros(size*2)
    for i in range (-(size//2),size//2):
        x=np.array(compute_saliency_map(tf.cast(input[window_start+i:window_start+size+i],tf.float32),model))
        y=np.concatenate((np.zeros(size//2+i),x,np.zeros(size//2+1-i)))
        Y=np.vstack([Y,y])
    return(Y)





