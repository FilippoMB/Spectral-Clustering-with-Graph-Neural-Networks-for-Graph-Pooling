from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.backend import tf
from keras.engine import Layer
from scipy.stats import cauchy
from spektral.layers.convolutional import filter_dot
import numpy as np


class MincutPool(Layer):
    """
    **Mode**: single, batch.

    This layer computes a soft clustering of the input graphs, and adds two
    additional unsupervised loss terms (the layer can be used without a
    supervised loss to compute node clustering, simply by minimizing the
    unsupervised loss).

    **Input**

    - node features of shape `(n_nodes, n_features)`;
    - adjacency matrix of shape `(n_nodes, n_nodes)`;
    - (optional) graph IDs of shape `(n_nodes, )` (since graph batch mode is not
    supported, this can only be used in single mode with all-zeros IDs);

    **Output**

    - reduced node features of shape `(k, n_features)`;
    - reduced adjacency matrix of shape `(k, k)`;
    - reduced graph IDs with shape `(k, )` (graph batch mode);

    **Arguments**

    - `k`: number of nodes to keep;
    - `h`: number of units in the hidden layer;
    - `return_mask`: boolean, whether to return the cluster assignment matrix,
    - `kernel_initializer`: initializer for the kernel matrix;
    - `bias_initializer`: initializer for the bias vector;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `bias_regularizer`: regularization applied to the bias vector;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    - `bias_constraint`: constraint applied to the bias vector.
    """
    def __init__(self,
                 k,
                 h=None,  # hidden layer size (None = no hidden layer)
                 return_mask=True,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MincutPool, self).__init__(**kwargs)
        self.k = k
        self.h = h
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]

        # Optional hidden layer
        if self.h is None:
            H_ = F
        else:
            H_ = self.h
            self.kernel_in = self.add_weight(shape=(F, H_),
                                             name='kernel_in',
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             constraint=self.kernel_constraint)

            if self.use_bias:
                self.bias_in = self.add_weight(shape=(H_,),
                                               name='bias_in',
                                               initializer=self.bias_initializer,
                                               regularizer=self.bias_regularizer,
                                               constraint=self.bias_constraint)

        # Output layer
        self.kernel_out = self.add_weight(shape=(H_, self.k),
                                          name='kernel_out',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias_out = self.add_weight(shape=(self.k,),
                                            name='bias_out',
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)

        super(MincutPool, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = None

        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(A) == 3

        # Optionally compute hidden layer
        if self.h is None:
            Hid = X
        else:
            Hid = K.dot(X, self.kernel_in)
            if self.use_bias:
                Hid = K.bias_add(Hid, self.bias_in)
            if self.activation is not None:
                Hid = self.activation(Hid)

        # Compute cluster assignment matrix
        S = K.dot(Hid, self.kernel_out)
        if self.use_bias:
            S = K.bias_add(S, self.bias_out)
        S = activations.softmax(S, axis=-1)  # Apply softmax to get cluster assignments

        # MinCut regularization
        A_pooled = matmul_AT_B_A(S, A)
        num = tf.trace(A_pooled)

        D = degree_matrix(A)
        den = tf.trace(matmul_AT_B_A(S, D))
        cut_loss = -(num / den)
        if batch_mode:
            cut_loss = K.mean(cut_loss)
        self.add_loss(cut_loss)

        # Orthogonality regularization
        SS = matmul_AT_B(S, S)
        I_S = tf.eye(self.k)
        ortho_loss = tf.norm(
            SS / tf.norm(SS, axis=(-1, -2)) - I_S / tf.norm(I_S), axis=(-1, -2)
        )
        if batch_mode:
            ortho_loss = K.mean(cut_loss)
        self.add_loss(ortho_loss)

        # Pooling
        X_pooled = matmul_AT_B(S, X)
        A_pooled = tf.linalg.set_diag(A_pooled, tf.zeros(K.shape(A_pooled)[:-1]))  # Remove diagonal
        A_pooled = normalize_A(A_pooled)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.segment_mean(I, I)
            I_pooled = tf_repeat_1d(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        X_shape_out = X_shape[:-2] + (self.k,) + X_shape[-1:]
        A_shape_out = A_shape[:-2] + (self.k, self.k)

        output_shape = [X_shape_out, A_shape_out]

        if len(input_shape) == 3:
            I_shape_out = A_shape[:-2] + (self.k, )
            output_shape.append(I_shape_out)

        if self.return_mask:
            S_shape_out = A_shape[:-1] + (self.k, )
            output_shape.append(S_shape_out)

        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'h': self.h,
            'return_mask': self.return_mask,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(MincutPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DiffPool(Layer):
    """
    A DiffPool layer as presented by [Ying et al.](https://arxiv.org/abs/1806.08804).

    **Mode**: single, batch.

    This layer computes a soft clustering of the input graphs, and adds two
    additional unsupervised loss terms (the layer can be used without a
    supervised loss to compute node clustering, simply by minimizing the
    unsupervised loss).

    **Input**

    - node features of shape `(n_nodes, n_features)`;
    - adjacency matrix of shape `(n_nodes, n_nodes)`;
    - (optional) graph IDs of shape `(n_nodes, )` (since graph batch mode is not
    supported, this can only be used in single mode with all-zeros IDs);

    **Output**

    - reduced node features of shape `(k, n_features)`;
    - reduced adjacency matrix of shape `(k, k)`;
    - reduced graph IDs with shape `(k, )` (graph batch mode);

    **Arguments**

    - `k`: number of nodes to keep;
    - `channels`: number of output channels (and size of the hidden layer);
    - `return_mask`: boolean, whether to return the cluster assignment matrix,
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    """

    def __init__(self,
                 k,
                 channels=None,  # channels of internal GNNs
                 return_mask=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DiffPool, self).__init__(**kwargs)
        self.k = k
        self.channels = channels
        self.return_mask = return_mask
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]

        if self.channels is None:
            self.channels = F

        self.kernel_emb = self.add_weight(shape=(F, self.channels),
                                          name='kernel_emb',
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        self.kernel_pool = self.add_weight(shape=(F, self.k),
                                           name='kernel_pool',
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        super(DiffPool, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = None

        N = K.shape(A)[-1]
        # Check if the layer is operating in batch mode (X and A have rank 3)
        batch_mode = K.ndim(A) == 3

        # Get normalized adjacency
        if K.is_sparse(A):
            I_ = tf.sparse.eye(N, dtype=A.dtype)
            A_ = tf.sparse.add(A, I_)
        else:
            I_ = tf.eye(N, dtype=A.dtype)
            A_ = A + I_
        fltr = normalize_A(A_)

        # Node embeddings
        Z = K.dot(X, self.kernel_emb)
        Z = filter_dot(fltr, Z)
        if self.activation is not None:
            Z = self.activation(Z)

        # Compute cluster assignment matrix
        S = K.dot(X, self.kernel_pool)
        S = filter_dot(fltr, S)
        S = activations.softmax(S, axis=-1)  # softmax applied row-wise

        # Link prediction loss
        S_gram = matmul_A_BT(S, S)
        if K.is_sparse(A):
            LP_loss = tf.sparse.add(A, -S_gram)  # A/tf.norm(A) - S_gram/tf.norm(S_gram)
        else:
            LP_loss = A - S_gram
        LP_loss = tf.norm(LP_loss, axis=(-1, -2))
        if batch_mode:
            LP_loss = K.mean(LP_loss)
        self.add_loss(LP_loss)

        # Entropy loss
        entr = tf.negative(tf.reduce_sum(tf.multiply(S, K.log(S + K.epsilon())), axis=-1))
        entr_loss = K.mean(entr, axis=-1)
        if batch_mode:
            entr_loss = K.mean(entr_loss)
        self.add_loss(entr_loss)

        # Pooling
        X_pooled = matmul_AT_B(S, Z)
        A_pooled = matmul_AT_B_A(S, A)

        output = [X_pooled, A_pooled]

        if I is not None:
            I_mean = tf.segment_mean(I, I)
            I_pooled = tf_repeat_1d(I_mean, tf.ones_like(I_mean) * self.k)
            output.append(I_pooled)

        if self.return_mask:
            output.append(S)

        return output

    def compute_output_shape(self, input_shape):
        X_shape = input_shape[0]
        A_shape = input_shape[1]
        X_shape_out = X_shape[:-2] + (self.k, self.channels)
        A_shape_out = A_shape[:-2] + (self.k, self.k)

        output_shape = [X_shape_out, A_shape_out]

        if len(input_shape) == 3:
            I_shape_out = A_shape[:-2] + (self.k,)
            output_shape.append(I_shape_out)

        if self.return_mask:
            S_shape_out = A_shape[:-1] + (self.k,)
            output_shape.append(S_shape_out)

        return output_shape

    def get_config(self):
        config = {
            'k': self.k,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(DiffPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TopKPool(Layer):
    """
    A top-k / gPool layer as presented by
    [Gao & Ji (2017)](https://openreview.net/forum?id=HJePRoAct7).
    Note that due to the lack of sparse-sparse matrix multiplication, this layer
    temporarily makes the adjacency matrix dense in order to compute \(A^2\)
    (needed to preserve connectivity after pooling).

    **Mode**: graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)` (with optional `batch`
    dimension);
    - adjacency matrix of shape `(n_nodes, n_nodes)` (with optional `batch`
    dimension);
    - graph IDs of shape `(n_nodes, )` (graph batch mode);
    - (optional) edge features of shape `(n_nodes, n_nodes, n_edge_features)`
    (with optional `batch` dimension);

    **Output**

    - reduced node features of shape `(k, n_features)` (with optional batch
    dimension);
    - reduced adjacency matrix of shape `(k, k)` (with optional batch
    dimension);
    - reduced graph IDs with shape `(n_graphs * k, )` (graph batch mode);
    - (optional) edge features of shape `(k, k, n_edge_features)`  (with
    optional `batch` dimension);

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;

    """

    def __init__(self, ratio,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TopKPool, self).__init__(**kwargs)
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        F = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.top_k_var = tf.Variable(0.0, validate_shape=False)
        super(TopKPool, self).build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1])
        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, K.l2_normalize(self.kernel))
        N = K.shape(X)[-2]
        indices = top_k(y[:, 0], I, self.ratio, self.top_k_var)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices), (N,))

        # Multiply X and y to make layer differentiable
        features = X * K.sigmoid(y)

        axis = 0 if len(K.int_shape(A)) == 2 else 1  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.boolean_mask(features, mask, axis=axis)

        # Compute A^2
        if A_is_sparse:
            A_dense = tf.sparse.to_dense(A)
        A_squared = K.dot(A, A_dense)

        # Reduce A
        A_pooled = tf.boolean_mask(A_squared, mask, axis=axis)
        A_pooled = tf.boolean_mask(A_pooled, mask, axis=axis + 1)
        if A_is_sparse:
            A_pooled = tf.contrib.layers.dense_to_sparse(A_pooled)

        # Reduce I
        if I is not None:
            I_pooled = tf.boolean_mask(I[:, None], mask)[:, 0]

        if I is not None:
            return [X_pooled, A_pooled, I_pooled, mask]
        else:
            return [X_pooled, A_pooled, mask]

    def compute_output_shape(self, input_shape):
        return input_shape + [(input_shape[0][:-1])]

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(TopKPool, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SAGPool(Layer):
    """
    A self-attention graph pooling layer as presented by
    [Lee et al. (2019)](https://arxiv.org/abs/1904.08082) and
    [Knyazev et al. (2019](https://arxiv.org/abs/1905.02850).

    This layer computes the following operations:

    $$
    y = GNN(X, A); \\;\\;\\;\\;
    \\textrm{idx} = \\textrm{rank}(y, k); \\;\\;\\;\\;
    \\bar X = (X \\odot \\textrm{tanh}(y))_{\\textrm{idx}}; \\;\\;\\;\\;
    \\bar A = A^2_{\\textrm{idx}, \\textrm{idx}}
    $$

    where \( \\textrm{rank}(y, k) \) returns the indices of the top k values of
    \( y \), and \( p \) is a learnable parameter vector of size \(F\).
    The gating operation \( \\textrm{tanh}(y) \) can be replaced with a sigmoid.

    Due to the lack of sparse-sparse matrix multiplication support, this layer
    temporarily makes the adjacency matrix dense in order to compute \(A^2\)
    (needed to preserve connectivity after pooling).
    **If memory is not an issue, considerable speedups can be achieved by using
    dense graphs directly.
    Converting a graph from dense to sparse and viceversa is a costly operation.**

    **Mode**: single, graph batch.

    **Input**

    - node features of shape `(n_nodes, n_features)`;
    - adjacency matrix of shape `(n_nodes, n_nodes)`;
    - (optional) graph IDs of shape `(n_nodes, )` (graph batch mode);

    **Output**

    - reduced node features of shape `(n_graphs * k, n_features)`;
    - reduced adjacency matrix of shape `(n_graphs * k, n_graphs * k)`;
    - reduced graph IDs with shape `(n_graphs * k, )` (graph batch mode);
    - If `return_mask=True`, the binary mask used for pooling, with shape
    `(n_graphs * k, )`.

    **Arguments**

    - `ratio`: float between 0 and 1, ratio of nodes to keep in each graph;
    - `return_mask`: boolean, whether to return the binary mask used for pooling;
    - `sigmoid_gating`: boolean, use a sigmoid gating activation instead of a
        tanh;
    - `kernel_initializer`: initializer for the kernel matrix;
    - `kernel_regularizer`: regularization applied to the kernel matrix;
    - `activity_regularizer`: regularization applied to the output;
    - `kernel_constraint`: constraint applied to the kernel matrix;
    """

    def __init__(self, ratio,
                 return_mask=False,
                 sigmoid_gating=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio  # Ratio of nodes to keep in each graph
        self.return_mask = return_mask
        self.sigmoid_gating = sigmoid_gating
        self.gating_op = K.sigmoid if self.sigmoid_gating else K.tanh
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.F = input_shape[0][-1]
        self.N = input_shape[0][0]
        self.kernel = self.add_weight(shape=(self.F, 1),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.top_k_var = tf.Variable(0.0, validate_shape=False)
        super().build(input_shape)

    def call(self, inputs):
        if len(inputs) == 3:
            X, A, I = inputs
            self.data_mode = 'graph'
        else:
            X, A = inputs
            I = tf.zeros(tf.shape(X)[:1], dtype=tf.int32)
            self.data_mode = 'single'
        if K.ndim(I) == 2:
            I = I[:, 0]

        A_is_sparse = K.is_sparse(A)

        # Get mask
        y = K.dot(X, self.kernel)
        y = filter_dot(A, y)
        N = K.shape(X)[-2]
        indices = top_k(y[:, 0], I, self.ratio, self.top_k_var)
        mask = tf.scatter_nd(tf.expand_dims(indices, 1), tf.ones_like(indices), (N,))

        # Multiply X and y to make layer differentiable
        features = X * self.gating_op(y)

        axis = 0 if len(K.int_shape(A)) == 2 else 1  # Cannot use negative axis in tf.boolean_mask
        # Reduce X
        X_pooled = tf.boolean_mask(features, mask, axis=axis)

        # Compute A^2
        if A_is_sparse:
            A_dense = tf.sparse.to_dense(A)
        else:
            A_dense = A
        A_squared = K.dot(A, A_dense)

        # Reduce A
        A_pooled = tf.boolean_mask(A_squared, mask, axis=axis)
        A_pooled = tf.boolean_mask(A_pooled, mask, axis=axis + 1)
        if A_is_sparse:
            A_pooled = tf.contrib.layers.dense_to_sparse(A_pooled)

        output = [X_pooled, A_pooled]

        # Reduce I
        if self.data_mode == 'graph':
            I_pooled = tf.boolean_mask(I[:, None], mask)[:, 0]
            output.append(I_pooled)

        if self.return_mask:
            output.append(mask)

        return output

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_mask:
            output_shape += [(input_shape[0][:-1])]
        return output_shape

    def get_config(self):
        config = {
            'ratio': self.ratio,
            'return_mask': self.return_mask,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def tf_repeat_1d(arr, repeats):
    arr = tf.expand_dims(arr, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    # If you want to support not just 1D arrays but any number of dimensions you can do:
    # tile_repeats = tf.concat(
    #     [[1], [max_repeats], tf.ones([tf.rank(arr) - 2], dtype=tf.int32)], axis=0)
    arr_tiled = tf.tile(arr, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def matrix_power(x, k):
    """
    Computes the k-th power of a square matrix.
    :param x: a square matrix (Tensor or SparseTensor)
    :param k: exponent
    :return: matrix of same type and dtype as the input
    """
    if K.is_sparse(x):
        sparse = True
        x_dense = tf.sparse.to_dense(x)
    else:
        sparse = False
        x_dense = x

    x_k = x_dense
    for _ in range(k - 1):
        x_k = filter_dot(x_k, x_dense)

    if sparse:
        return tf.contrib.layers.dense_to_sparse(x_k)
    else:
        return x_k


def top_k(scores, I, ratio, top_k_var):
    """
    Returns indices to get the top K values in `scores` segment-wise, with
    segments defined by I. K is not fixed, but it is defined as a ratio of the
    number of elements in each segment.
    :param scores: 1D tensor with scores
    :param I: segment IDs
    :param ratio: ratio of elements to keep for each segment
    :return:
    """
    num_nodes = tf.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.ceil(ratio * tf.cast(num_nodes, tf.float32))
    to_keep = tf.cast(to_keep, tf.int32)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(scores)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)  # subtract 1 to ensure that filler values do not get picked
    dense_y = tf.assign(top_k_var, dense_y,
                        validate_shape=False)  # top_k_var is a variable with unknown shape defined in the layer
    dense_y = tf.scatter_update(dense_y, index, scores)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = tf_repeat_1d(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm


def every_other_top(scores, I, top_k_var):
    """
    Returns index to get every other top value in `scores` segment-wise, with
    segments defined by I.
    :param scores: 1D tensor with scores
    :param I: segment IDs
    :return:
    """
    num_nodes = tf.segment_sum(tf.ones_like(I), I)  # Number of nodes in each graph
    cumsum = tf.cumsum(num_nodes)  # Cumulative number of nodes (A, A+B, A+B+C)
    cumsum_start = cumsum - num_nodes  # Start index of each graph
    n_graphs = tf.shape(num_nodes)[0]  # Number of graphs in batch
    max_n_nodes = tf.reduce_max(num_nodes)  # Order of biggest graph in batch
    batch_n_nodes = tf.shape(I)[0]  # Number of overall nodes in batch
    to_keep = tf.ceil(tf.cast(num_nodes, tf.float32) / 2)
    to_keep = tf.cast(to_keep, tf.int32)  # Nodes to keep in each graph

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(cumsum_start, I)) + (I * max_n_nodes)

    y_min = tf.reduce_min(scores)
    dense_y = tf.ones((n_graphs * max_n_nodes,))
    dense_y = dense_y * tf.cast(y_min - 1, tf.float32)  # subtract 1 to ensure that filler values do not get picked
    dense_y = tf.assign(top_k_var, dense_y,
                        validate_shape=False)  # top_k_var is a variable with unknown shape defined in the layer
    dense_y = tf.scatter_update(dense_y, index, scores)
    dense_y = tf.reshape(dense_y, (n_graphs, max_n_nodes))

    perm = tf.argsort(dense_y, direction='DESCENDING')
    perm = perm + cumsum_start[:, None]
    perm = tf.concat((perm[:, ::2], perm[:, 1::2]), axis=-1)
    perm = tf.reshape(perm, (-1,))

    to_rep = tf.tile(tf.constant([1., 0.]), (n_graphs,))
    rep_times = tf.reshape(tf.concat((to_keep[:, None], (max_n_nodes - to_keep)[:, None]), -1), (-1,))
    mask = tf_repeat_1d(to_rep, rep_times)

    perm = tf.boolean_mask(perm, mask)

    return perm


def sparse_bool_mask(x, mask, axis=0):
    # Only necessary if indices may have non-unique elements
    indices = tf.boolean_mask(tf.range(tf.shape(x)[axis]), mask)
    n_indices = tf.size(indices)
    # Get indices for the axis
    idx = x.indices[:, axis]
    # Find where indices match the selection
    eq = tf.equal(tf.expand_dims(idx, 1), tf.cast(indices, tf.int64))  # TODO this has quadratic cost
    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)
    # Selected values
    values_new = tf.boolean_mask(x.values, sel, axis=0)
    # New index value for selected elements
    n_indices = tf.cast(n_indices, tf.int64)
    idx_new = tf.reduce_sum(tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1)
    idx_new = tf.boolean_mask(idx_new, sel, axis=0)
    # New full indices tensor
    indices_new = tf.boolean_mask(x.indices, sel, axis=0)
    indices_new = tf.concat([indices_new[:, :axis],
                             tf.expand_dims(idx_new, 1),
                             indices_new[:, axis + 1:]], axis=1)
    # New shape
    shape_new = tf.concat([x.dense_shape[:axis],
                           [n_indices],
                           x.dense_shape[axis + 1:]], axis=0)
    return tf.SparseTensor(indices_new, values_new, shape_new)


def matmul_AT_B_A(A, B):
    """
    Computes A.T * B * A, deals with sparse A/B and batch mode automatically.
    TODO batch mode does not work for sparse tensors.
    :param A: Tensor with rank k = {2, 3} or SparseTensor with rank k = 2
    :param B: Tensor or SparseTensor with rank k.
    :return:
    """
    if K.ndim(A) == 3:
        return K.batch_dot(
            transpose(K.batch_dot(B, A), (0, 2, 1)), A
        )
    else:
        return K.dot(transpose(K.dot(B, A)), A)


def matmul_AT_B(A, B):
    """
    Computes A.T * B, deals with sparse A/B and batch mode automatically.
    TODO batch mode does not work for sparse tensors.
    :param A: Tensor with rank k = {2, 3} or SparseTensor with rank k = 2
    :param B: Tensor or SparseTensor with rank k.
    :return:
    """
    if K.ndim(A) == 3:
        return K.batch_dot(transpose(A, (0, 2, 1)), B)
    else:
        return K.dot(transpose(A), B)


def matmul_A_BT(A, B):
    """
    Computes A * B.T, deals with sparse A/B and batch mode automatically.
    TODO batch mode does not work for sparse tensors.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :param B: Tensor or SparseTensor with rank k.
    :return: SparseTensor of rank k.
    """
    if K.ndim(A) == 3:
        return K.batch_dot(
            A, transpose(B, (0, 2, 1))
        )
    else:
        return K.dot(A, transpose(B))


def normalize_A(A):
    """
    Computes symmetric normalization of A, deals with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: SparseTensor of rank k.
    """
    D = degrees(A)
    D = tf.sqrt(D)[:, None] + K.epsilon()
    if K.ndim(A) == 3:
        # Batch mode
        output = (A / D) / transpose(D, perm=(0, 2, 1))
    else:
        # Single mode
        output = (A / D) / transpose(D)

    return output


def degrees(A):
    """
    Computes the degrees of each node in A, deals with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :return: Tensor or SparseTensor of rank k - 1.
    """
    if K.is_sparse(A):
        D = tf.sparse.reduce_sum(A, axis=-1)
    else:
        D = tf.reduce_sum(A, axis=-1)

    return D


def degree_matrix(A, return_sparse_batch=False):
    """
    Computes the degree matrix of A, deals with sparse A and batch mode
    automatically.
    :param A: Tensor or SparseTensor with rank k = {2, 3}.
    :param return_sparse_batch: if operating in batch mode, return a
    SparseTensor. Note that the sparse degree tensor returned by this function
    cannot be used for sparse matrix multiplication afterwards.
    TODO why?
    :return: SparseTensor of rank k.
    """
    D = degrees(A)

    batch_mode = K.ndim(D) == 2
    N = tf.shape(D)[-1]
    batch_size = tf.shape(D)[0] if batch_mode else 1

    inner_index = tf.tile(tf.stack([tf.range(N)] * 2, axis=1), (batch_size, 1))
    if batch_mode:
        if return_sparse_batch:
            outer_index = tf_repeat_1d(
                tf.range(batch_size), tf.ones(batch_size) * tf.cast(N, tf.float32)
            )
            indices = tf.concat([outer_index[:, None], inner_index], 1)
            dense_shape = (batch_size, N, N)
        else:
            return tf.linalg.diag(D)
    else:
        indices = inner_index
        dense_shape = (N, N)

    indices = tf.cast(indices, tf.int64)
    values = tf.reshape(D, (-1, ))
    return tf.SparseTensor(indices, values, dense_shape)


def transpose(A, perm=None, name=None):
    """
    Transposes A according to perm, deals with sparse A automatically.
    :param A: Tensor or SparseTensor with rank k.
    :param perm: permutation indices of size k.
    :param name: name for the operation.
    :return: Tensor or SparseTensor with rank k.
    """
    if K.is_sparse(A):
        transpose_op = tf.sparse_transpose
    else:
        transpose_op = tf.transpose

    return transpose_op(A, perm=perm, name=name)
