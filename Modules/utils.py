import numpy as np
import tensorflow as tf
import keras.backend as k
from numpy.lib.stride_tricks import as_strided
import pandas as pd

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
    seq: np.ndarray, shape=(n,4),
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
    """
    Applies one hot encoding to a DNA sequence.

    Parameters
    ----------
    seq: string,
        a DNA sequence as a string

    Returns
    -------
    seq_encoded: np.ndarray, shape=(n,4),
        2D-array, a one hot encoded DNA sequence
    """
    seq_encoded=list()
    mapping = {"A":[1, 0, 0, 0],"C": [0, 1, 0, 0],"G":[0, 0, 1, 0],"T":[0, 0, 0, 1]}
    for i in seq:
      seq_encoded.append(mapping[i.upper()] if i.upper() in mapping.keys() else [0, 0, 0, 0]) 
    return np.array(seq_encoded)

def force_ohe(seq):
    """
    forces the one hot encoded format to an array of shape (n,4)

    Parameters
    ----------
    seq: np.ndarray, shape=(n,4),
        2D-array

    Returns
    -------
    seq_forced: np.ndarray, shape=(n,4),
        2D-array, a one hot encoded DNA sequence
    """
    max=0
    seq_forced=np.copy(seq)
    for i in range(len(seq_forced)):
        new=[0,0,0,0]
        max=np.max(seq_forced[i])
        if max==0.25:
            new[np.random.randint(0,3)]=1
        else:
            new=np.where(seq_forced[i]==np.max(seq_forced[i]),1,0)
        seq_forced[i]=new
    return (seq_forced)

def random_ohe_sequence(length):
    """
    generates a random one hot encoded DNA sequence

    Parameters
    ----------
    length: int,
        an integer, size wanted for the generated sequence

    Returns
    -------
    rand: np.ndarray, shape=(length,4),
        a random one hot encoded DNA sequence
    """
    rand=np.eye(4)[np.random.randint(0,3,length)]
    return(rand)

def GC_content(seq):
    """
    computes the GC content of a DNA sequence

    Parameters
    ----------
    seq: string,
        a DNA sequence as a string

    Returns
    -------
    cont: float,
        the percentage of G and C bases present in the sequence
    """
    gc_count = sum(base[1]+base[2] for base in seq)
    gc_content = gc_count / len(seq)
    return gc_content

def sequence_entropy(seq):
    """
    computes the entropy of a DNA sequence

    Parameters
    ----------
    seq: np.ndarray, shape=(n,4),
        2D-array, a one hot encoded DNA sequence

    Returns
    -------
    ent: float,
        the entropy of the sequence
    """
    ent=sum(np.sum(-seq*np.log(seq+k.epsilon()), axis=1))
    return(ent)

def seq_similarity(seq1, seq2):
    """
    computes the similarity between two DNA sequences

    Parameters
    ----------
    seq1, seq2: np.ndarray, shape=(n,4),
        2D-arrays, one hot encoded DNA sequences

    Returns
    -------
    similarity: float,
        the simmilarity between two DNA sequences
    """
    if len(seq1) != len(seq2):
        raise ValueError("Les séquences doivent avoir la même longueur.")
    
    hamming_distance = sum(a != b for a, b in zip(seq1, seq2))
    similarity = 1 - (hamming_distance / len(seq1))
    
    return similarity

def undifined_rate(seq):
    """
    computes the percentage of not clearly defined bases in a one hot encoded sequence

    Parameters
    ----------
    seq: np.ndarray, shape=(n,4),
        2D-array, a one hot encoded DNA sequence

    Returns
    -------
    ent: float,
        the percentage of not clearly defined bases in a one hot encoded sequence
    """
    c=0
    for i in seq:
        if 0.2<max(i)<0.3:
            c+=1
    rate=c/len(seq)
    return(rate)   

def kmer_counts(one_hots, k, order='ACGT', includeN=True, as_pandas=True):
    """Compute kmer occurences in one-hot encoded sequence."""
    # Convert input into list-like of one_hot 2D-arrays
    # If 3D-array optionnally use faster implementation
    fast = False
    if isinstance(one_hots, dict):
        one_hots = list(one_hots.values())
    elif isinstance(one_hots, np.ndarray):
        if one_hots.ndim == 2:
            # single array turned into list of one array
            one_hots = [one_hots]
        elif one_hots.ndim == 3:
            # Check that last dimension is 4
            assert one_hots.shape[2] == 4
            fast = True
    if fast:  # Faster on 3D array
        # Initialise kD array
        all_counts = np.zeros(tuple(5 for i in range(k)), dtype=int)
        if k == 1:
            # Count each base
            all_counts[:4] = one_hots.sum(axis=(0, 1))
            # Count leftover as Ns
            all_counts[4] = (len(one_hots) * one_hots.shape[1]
                             - all_counts[:4].sum())
        else:
            # Convert one_hot to integer tokens
            tokens = (np.argmax(one_hots, axis=-1)
                      + 4 * (np.sum(one_hots, axis=-1) != 1))
            # Get kmers with sliding_window_view
            kmers = sliding_window_view(tokens, (1, k)).reshape(-1, k)
            # Count kmers in the kD array
            np.add.at(all_counts, tuple(kmers[:, i] for i in range(k)), 1)
    else:  # Iterate over one-hot encoded arrays
        # Initialise kD array
        all_counts = np.zeros(tuple(5 for i in range(k)), dtype=int)
        for oh in one_hots:
            # Check that arrays are 2D with a shape of 4 in the 2nd dimension
            assert oh.ndim == 2
            assert oh.shape[1] == 4
            if k == 1:
                # Count each base
                all_counts[:4] += oh.sum(axis=0)
                # Count leftover as Ns
                all_counts[4] += len(oh) - oh.sum()
            else:
                # Convert one_hot to integer tokens
                tokens = np.argmax(oh, axis=-1) + 4*(np.sum(oh, axis=-1) != 1)
                # Get kmers with sliding_window_view
                kmers = sliding_window_view(tokens, k)
                # Count kmers in the kD array
                np.add.at(all_counts, tuple(kmers[:, i] for i in range(k)), 1)
    # Format output
    if includeN:
        order += 'N'
    else:
        all_counts = all_counts[tuple(slice(0, -1) for i in range(k))]
    if as_pandas:
        ser = pd.Series(
            all_counts.ravel(),
            index=pd.MultiIndex.from_product([list(order)]*k))
        return ser.sort_index()
    else:
        return all_counts

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





