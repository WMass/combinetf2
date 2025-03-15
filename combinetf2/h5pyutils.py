import math

import numpy as np
import tensorflow as tf


def maketensor(h5dset):
    if "original_shape" in h5dset.attrs:
        shape = h5dset.attrs["original_shape"]
    else:
        shape = h5dset.shape

    if h5dset.size == 0:
        return tf.zeros(shape, h5dset.dtype)

    # read directly from hdf5 dataset to the underlying buffer of a tensor
    # this requires that the tensor is located on the CPU, so force the device
    with tf.device(tf.config.list_logical_devices("CPU")[0]):
        atensor = tf.zeros(h5dset.shape, h5dset.dtype)
        # zero tensors have a special flag set, using the identity clears this implicitly
        atensor = tf.identity(atensor)

    # read into the underlying array
    h5dset.read_direct(atensor.__array__())

    # the reshape operation is needed in case the hdf5 dataset was flattened
    # this also triggers a copy of the tensor to the default device (e.g. GPU)
    # if needed (ie if not the default CPU device)
    atensor = tf.reshape(atensor, shape)
    return atensor


def makesparsetensor(h5group):
    indices = maketensor(h5group["indices"])
    values = maketensor(h5group["values"])
    dense_shape = h5group.attrs["dense_shape"]

    return tf.sparse.SparseTensor(indices, values, dense_shape)


def writeFlatInChunks(arr, h5group, outname, maxChunkBytes=1024**2):
    arrflat = arr.reshape(-1)

    esize = np.dtype(arrflat.dtype).itemsize
    nbytes = arrflat.size * esize

    # special handling for empty datasets, which should not use chunked storage or compression
    if arrflat.size == 0:
        chunksize = 1
        chunks = None
        compression = None
    else:
        chunksize = int(min(arrflat.size, max(1, math.floor(maxChunkBytes / esize))))
        chunks = (chunksize,)
        compression = "gzip"

    h5dset = h5group.create_dataset(
        outname,
        arrflat.shape,
        chunks=chunks,
        dtype=arrflat.dtype,
        compression=compression,
    )

    # write in chunks, preserving sparsity if relevant
    for ielem in range(0, arrflat.size, chunksize):
        aout = arrflat[ielem : ielem + chunksize]
        if np.count_nonzero(aout):
            h5dset[ielem : ielem + chunksize] = aout

    h5dset.attrs["original_shape"] = np.array(arr.shape, dtype="int64")

    return nbytes


def writeSparse(indices, values, dense_shape, h5group, outname, maxChunkBytes=1024**2):
    outgroup = h5group.create_group(outname)

    nbytes = 0
    nbytes += writeFlatInChunks(indices, outgroup, "indices", maxChunkBytes)
    nbytes += writeFlatInChunks(values, outgroup, "values", maxChunkBytes)
    outgroup.attrs["dense_shape"] = np.array(dense_shape, dtype="int64")

    return nbytes


def simple_sparse_slice0end(in_sparse, end):
    """
    Slice a tf.sparse.SparseTensor along axis 0 starting 0 to the 'end'.
    """

    # Convert dense_shape, indices, and values to tensors if they aren't already
    dense_shape = in_sparse.dense_shape
    indices = in_sparse.indices
    values = in_sparse.values

    # Compute output dense shape after slicing
    out_shape = tf.concat([[end], dense_shape[1:]], axis=0)

    # Filter rows: select entries where indices[:, 0] < end
    mask = indices[:, 0] < end
    selected_indices = tf.boolean_mask(indices, mask)
    selected_values = tf.boolean_mask(values, mask)

    # Return the sliced sparse tensor
    return tf.sparse.SparseTensor(
        indices=selected_indices, values=selected_values, dense_shape=out_shape
    )
