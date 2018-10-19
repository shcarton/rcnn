
import gzip
import random
import json

import theano
import numpy as np

from nn import EmbeddingLayer
from utils import say, load_embedding_iterator

def read_rationales(path):
    data = [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    return data

def read_annotations(path):
    data_x, data_y = [ ], [ ]
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            y, sep, x = line.partition("\t")
            x, y = x.split(), y.split()
            if len(x) == 0: continue
            y = np.asarray([ float(v) for v in y ], dtype = theano.config.floatX)
            data_x.append(x)
            data_y.append(y)
    say("{} examples loaded from {}\n".format(
            len(data_x), path
        ))
    say("max text length: {}\n".format(
        max(len(x) for x in data_x)
    ))

    #so data_x is a list of lists of string tokens, and data_y is a list of theano vectors representing the N different class values for each x
    return data_x, data_y

def create_embedding_layer(path):
    embedding_layer = EmbeddingLayer(
            n_d = 200,
            vocab = [ "<unk>", "<padding>" ],
            embs = load_embedding_iterator(path),
            oov = "<unk>",
            #fix_init_embs = True
            fix_init_embs = False
        )
    return embedding_layer


def create_batches(x, y, batch_size, padding_id, sort=True, max_size = 50000, num_policy_samples = 1, sort_each_batch=False, return_indices=False, z=None):
    '''

    :param x: list of numpy arrays
    :param y: numpy vector
    :param batch_size:
    :param padding_id:
    :param sort:
    :param max_size: max size in bytes a batch is allowed to be. If a batch exceeds this, its length gets shortened.
    :param num_policy_samples:
    :return:
    '''
    batches_x, batches_y = [], []
    batches_i = []
    batches_z = []
    num_examples = len(x)
    num_batches = (num_examples-1)/batch_size + 1

    indices = list(range(num_examples))
    if sort:
        indices = sorted(indices, key=lambda i: len(x[i]))
        x = [ x[i] for i in indices ]
        z = [ z[i] for i in indices ] if z is not None else None
        y = [ y[i] for i in indices ] if y is not None else None

    batch_start = 0
    batch_end = 0

    max_batch_elements = max_size/num_policy_samples

    while batch_end < num_examples:

        batch_width = len(x[batch_start])
        batch_length = 0

        #Make sure we don't end up with a batch that contains too many items
        while batch_end < num_examples and batch_length < batch_size and batch_width*batch_length <= max_batch_elements:
            batch_end += 1
            batch_length += 1
            if len(x[batch_end-1]) > batch_width:
                batch_width = len(x[batch_end-1])


        bx, by, bz, bi = create_one_batch(
                    x[batch_start:batch_end],
                    y[batch_start:batch_end] if y is not None else None,
                    padding_id,
                    indices = indices[batch_start:batch_end],
                    sort = sort_each_batch,
                    return_indices = True,
                    lstz = z[batch_start:batch_end] if z is not None else None
                )

        # bi = indices[batch_start:batch_end]


        batches_x.append(bx)
        batches_y.append(by)
        batches_z.append(bz)
        batches_i.append(bi)

        batch_start = batch_end

    # for i in xrange(M):
    #     bx, by = create_one_batch(
    #                 x[i*batch_size:(i+1)*batch_size],
    #                 y[i*batch_size:(i+1)*batch_size] if y is not None else None,
    #                 padding_id
    #             )
    #     batches_x.append(bx)
    #     batches_y.append(by)

    #if the examples were sorted before batching (to minimize padding)
    if sort:
        random.seed(5817)
        randomized_indices = range(len(batches_x))
        random.shuffle(randomized_indices)
        batches_x = [ batches_x[i] for i in randomized_indices ]
        batches_y = [ batches_y[i] for i in randomized_indices ] if y is not None else None
        batches_z = [ batches_z[i] for i in randomized_indices ] if z is not None else None
        batches_i = [ batches_i[i] for i in randomized_indices ]

    assert(np.sum([b.shape[1] for b in batches_x]) == len(x))

    r =[batches_x, batches_y]

    if z is not None:
        r.append(batches_z)

    if return_indices:
        r.append(batches_i)


    return tuple(r)

def create_one_batch(lstx, lsty, padding_id, sort=False, indices= None, return_indices = False, lstz=None):

    if sort and lsty is not None: #sorting is based on y value
        sorted_indices= np.argsort([float(y) for y in lsty])
        lstx = [lstx[i] for i in sorted_indices]
        lsty = [lsty[i] for i in sorted_indices]
        lstz = [lstz[i] for i in sorted_indices] if lstz is not None else None
        indices = [indices[i] for i in sorted_indices]
    max_len = max(len(x) for x in lstx)
    assert min(len(x) for x in lstx) > 0
    bx = np.column_stack([ np.pad(x, (max_len-len(x),0), "constant", constant_values=padding_id) for x in lstx ])
    by = np.vstack(lsty).astype(theano.config.floatX) if lsty is not None else None
    bz = np.column_stack([ np.pad(z, (max_len-len(z),0), "constant", constant_values=0) for z in lstz ]) if lstz is not None else None
    bi = np.vstack(indices).astype(theano.config.floatX) if indices is not None else None

    return bx, by, bz, bi

