import itertools
from collections import OrderedDict
import numpy as np
from keras import backend as K
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def sp_matrix_to_sp_tensor_value(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensorValue
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensorValue
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return K.tf.SparseTensorValue(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def sp_matrix_to_sp_tensor(x):
    """
    Converts a Scipy sparse matrix to a tf.SparseTensor
    :param x: a Scipy sparse matrix
    :return: tf.SparseTensor
    """
    if not hasattr(x, 'tocoo'):
        try:
            x = sp.coo_matrix(x)
        except:
            raise TypeError('x must be convertible to scipy.coo_matrix')
    else:
        x = x.tocoo()
    return K.tf.SparseTensor(
        indices=np.array([x.row, x.col]).T,
        values=x.data,
        dense_shape=x.shape
    )


def node_feat_norm(feat_list, norm='ohe'):
    """
    Apply one-hot encoding or z-score to a list of node features
    """
    if norm == 'ohe':
        fnorm = OneHotEncoder(sparse=False, categories='auto')
    elif norm == 'zscore':
        fnorm = StandardScaler()
    else:
        raise ValueError('Possible feat_norm: ohe, zscore')
    fnorm.fit(np.vstack(feat_list))
    feat_list = [fnorm.transform(feat_.astype(np.float32)) for feat_ in feat_list]
    return feat_list


def get_sw_key(sess):
    return [op.name
            for op in sess.graph.get_operations()
            if op.type == "Placeholder" and op.name.endswith('sample_weights')][-1] \
           + ':0'
           
def product_dict(kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    if len(keys) > 0:
        for instance in itertools.product(*vals):
            yield OrderedDict(zip(keys, instance))
    else:
        for _ in [dict(), ]:
            yield _
