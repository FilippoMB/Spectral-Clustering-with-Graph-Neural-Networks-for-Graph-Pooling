import itertools
import time
from collections import OrderedDict

import keras.backend as K
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from keras.layers import Input, Dense
from keras.metrics import categorical_accuracy
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from spektral.layers import GraphConv, GlobalAvgPool, ARMAConv, GraphConvSkip
from spektral.layers import MinCutPool, DiffPool, TopKPool, SAGPool
from spektral.layers.ops import sp_matrix_to_sp_tensor_value
from spektral.utils import batch_iterator, log, init_logging
from spektral.utils.convolution import normalized_adjacency

from utils.dataset_loader import get_graph_kernel_dataset


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    if len(keys) > 0:
        for instance in itertools.product(*vals):
            yield dict(zip(keys, instance))
    else:
        for _ in [dict(), ]:
            yield _


def create_batch(A_list, X_list):
    A_out = sp.block_diag(list(A_list))
    X_out = np.vstack(X_list)
    n_nodes = np.array([a_.shape[0] for a_ in A_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)

    return A_out, X_out, I_out


def evaluate(A_list, X_list, y_list, ops):
    batches_ = batch_iterator([A_list, X_list, y_list], batch_size=P['batch_size'])
    output_ = []
    for A__, X__, y__ in batches_:
        A__, X__, I__ = create_batch(A__, X__)
        feed_dict_ = {X_in: X__,
                      A_in: sp_matrix_to_sp_tensor_value(A__),
                      I_in: I__,
                      target: y__,
                      SW_KEY: np.ones((1,))}
        outs_ = sess.run(ops, feed_dict=feed_dict_)
        output_.append(outs_)
    return np.mean(output_, 0)


seed = 0
np.random.seed(seed)

# Parameters
P = OrderedDict(
    runs=10,             # Runs to repeat per config
    data_mode='bench',   # bench / synth
    GNN_type='GCS',      # Type of GNN {GCN, GCS, Cheb, ARMA}
    n_channels=32,       # Channels per layer
    activ='relu',        # Activation in GNN and mincut
    mincut_H=16,         # Dimension of hidden state in mincut
    GNN_l2=1e-4,         # l2 regularisation of GNN
    pool_l2=1e-4,        # l2 regularisation for mincut
    epochs=500,          # Number of training epochs
    es_patience=50,      # Patience for early stopping
    learning_rate=5e-4,  # Learning rate
)
log_dir = init_logging()  # Create log directory and file
log(P)

# Tunables
tunables = OrderedDict(
    dataset_ID=['PROTEINS'],
    method=['mincut_pool']  # 'flat', 'dense', 'diff_pool', 'top_k_pool', 'mincut_pool', 'sag_pool'
)
log(tunables)

df_out = None
for T in product_dict(**tunables):
    # Update params with current config
    P.update(T)
    print(T)

    # Custom parameters based on method
    SW_KEY = 'dense_{}_sample_weights:0'.format(4 if P['method'] == 'dense' else 1)
    if P['method'] == 'diff_pool' or P['method'] == 'mincut_pool':
        P['batch_size'] = 1
    else:
        P['batch_size'] = 8

    results = {'test_loss': [], 'test_acc': []}
    for _ in range(P['runs']):
        ########################################################################
        # LOAD DATA
        ########################################################################
        if P['data_mode'] == 'bench':
            A, X, y = get_graph_kernel_dataset(P['dataset_ID'], feat_norm='ohe')
            # Train/test split
            A_train, A_test, \
            X_train, X_test, \
            y_train, y_test = train_test_split(A, X, y, test_size=0.1, stratify=y)
            A_train, A_val, \
            X_train, X_val, \
            y_train, y_val = train_test_split(A_train, X_train, y_train, test_size=0.1, stratify=y_train)
        elif P['data_mode'] == 'synth':
            loaded = np.load('data/hard.npz', allow_pickle=True)
            X_train, A_train, y_train = loaded['tr_feat'], list(loaded['tr_adj']), loaded['tr_class']
            X_test, A_test, y_test = loaded['te_feat'], list(loaded['te_adj']), loaded['te_class']
            X_val, A_val, y_val = loaded['val_feat'], list(loaded['val_adj']), loaded['val_class']
        else:
            raise ValueError

        # Parameters
        F = X_train[0].shape[-1]  # Dimension of node features
        n_out = y_train[0].shape[-1]  # Dimension of the target
        average_N = np.ceil(np.mean([a.shape[-1] for a in A_train]))  # Average number of nodes in dataset

        # Preprocessing adjacency matrices for convolution
        if P['GNN_type'] == 'GCS' or P['GNN_type'] == 'ARMA':
            A_train = [normalized_adjacency(a) for a in A_train]
            A_val = [normalized_adjacency(a) for a in A_val]
            A_test = [normalized_adjacency(a) for a in A_test]
        elif P['GNN_type'] == 'GCN':
            A_train = [normalized_adjacency(a + sp.eye(a.shape[0])) for a in A_train]
            A_val = [normalized_adjacency(a + sp.eye(a.shape[0])) for a in A_val]
            A_test = [normalized_adjacency(a + sp.eye(a.shape[0])) for a in A_test]
        else:
            raise ValueError('Unknown GNN type: {}'.format(P['GNN_type']))

        ########################################################################
        # BUILD MODEL
        ########################################################################
        X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, F), name='X_in'))
        A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), sparse=True)
        I_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))
        target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_out), name='target'))

        if P['GNN_type'] == 'GCN':
            GNN = GraphConv
        elif P['GNN_type'] == 'ARMA':
            GNN = ARMAConv
        elif P['GNN_type'] == 'GCS':
            GNN = GraphConvSkip
        else:
            raise ValueError('Unknown GNN type: {}'.format(P['GNN_type']))

        # Block 1
        if P['method'] == 'diff_pool':
            X_1, A_1, I_1, M_1 = DiffPool(k=int(average_N // 2),
                                          channels=P['n_channels'],
                                          activation=P['activ'],
                                          kernel_regularizer=l2(P['GNN_l2']))([X_in, A_in, I_in])
        elif P['method'] == 'dense':
            X_1 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_in)
            A_1 = A_in
            I_1 = I_in
        else:
            gc1 = GNN(P['n_channels'],
                      activation=P['activ'],
                      kernel_regularizer=l2(P['GNN_l2']))([X_in, A_in])

            if P['method'] == 'top_k_pool':
                X_1, A_1, I_1, M_1 = TopKPool(0.5)([gc1, A_in, I_in])
            elif P['method'] == 'sag_pool':
                X_1, A_1, I_1 = SAGPool(0.5)([gc1, A_in, I_in])
            elif P['method'] == 'mincut_pool':
                X_1, A_1, I_1, M_1 = MinCutPool(k=int(average_N // 2),
                                                h=P['mincut_H'],
                                                activation=P['activ'],
                                                kernel_regularizer=l2(P['pool_l2']))([gc1, A_in, I_in])

            elif P['method'] == 'flat':
                X_1 = gc1
                A_1 = A_in
                I_1 = I_in
            else:
                raise ValueError

        # Block 2
        if P['method'] == 'diff_pool':
            X_2, A_2, I_2, M_2 = DiffPool(k=int(average_N // 4),
                                          channels=P['n_channels'],
                                          activation=P['activ'],
                                          kernel_regularizer=l2(P['GNN_l2']))([X_1, A_1, I_1])
        elif P['method'] == 'dense':
            X_2 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_1)
            A_2 = A_1
            I_2 = I_1
        else:
            gc2 = GNN(P['n_channels'],
                      activation=P['activ'],
                      kernel_regularizer=l2(P['GNN_l2']))([X_1, A_1])

            if P['method'] == 'top_k_pool':
                X_2, A_2, I_2, M_2 = TopKPool(0.5)([gc2, A_1, I_1])
            elif P['method'] == 'sag_pool':
                X_2, A_2, I_2 = SAGPool(0.5)([gc2, A_1, I_1])
            elif P['method'] == 'mincut_pool':
                X_2, A_2, I_2, M_2 = MinCutPool(k=int(average_N // 4),
                                                h=P['mincut_H'],
                                                activation=P['activ'],
                                                kernel_regularizer=l2(P['pool_l2']))([gc2, A_1, I_1])

            elif P['method'] == 'flat':
                X_2 = gc2
                A_2 = A_1
                I_2 = I_1
            else:
                raise ValueError

        # Block 3
        if P['method'] == 'dense':
            X_3 = Dense(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))(X_2)
        else:
            X_3 = GNN(P['n_channels'], activation=P['activ'], kernel_regularizer=l2(P['GNN_l2']))([X_2, A_2])

        # Output block
        avgpool = GlobalAvgPool()([X_3, I_2])
        output = Dense(n_out, activation='softmax')(avgpool)

        # Build model
        model = Model([X_in, A_in, I_in], output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', target_tensors=[target])
        model.summary()

        # Training setup
        sess = K.get_session()
        loss = model.total_loss
        acc = K.mean(categorical_accuracy(target, model.output))
        opt = tf.train.AdamOptimizer(learning_rate=P['learning_rate'])
        train_step = opt.minimize(loss)

        # Initialize all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Run training loop
        current_batch = 0
        model_loss = 0
        model_acc = 0
        best_val_loss = np.inf
        patience = P['es_patience']
        batches_in_epoch = 1 + y_train.shape[0] // P['batch_size']
        total_batches = batches_in_epoch * P['epochs']

        ########################################################################
        # FIT MODEL
        ########################################################################
        log('Fitting model')
        start_time = time.time()
        batches = batch_iterator([A_train, X_train, y_train],
                                 batch_size=P['batch_size'], epochs=P['epochs'])
        epoch_time = [0]
        for A_, X_, y_ in batches:
            A_, X_, I_ = create_batch(A_, X_)
            tr_feed_dict = {X_in: X_,
                            A_in: sp_matrix_to_sp_tensor_value(A_),
                            I_in: I_,
                            target: y_,
                            SW_KEY: np.ones((1,))}
            epoch_time[-1] -= time.time()
            outs = sess.run([train_step, loss, acc], feed_dict=tr_feed_dict)
            epoch_time[-1] += time.time()

            model_loss += outs[1]
            model_acc += outs[2]
            current_batch += 1
            if current_batch % batches_in_epoch == 0:
                model_loss /= batches_in_epoch
                model_acc /= batches_in_epoch

                val_loss, val_acc = evaluate(A_val, X_val, y_val, [loss, acc])
                ep = int(current_batch / batches_in_epoch)
                log('Ep: {:d} - Loss: {:.2f} - Acc: {:.2f} - Val loss: {:.2f} - Val acc: {:.2f}'
                    .format(ep, model_loss, model_acc, val_loss, val_acc))
                log('{} - Average epoch time: {} +- {}'
                    .format(P['method'], np.mean(epoch_time[1:]), np.std(epoch_time[1:])))
                epoch_time.append(0)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = P['es_patience']
                    log('New best val_loss {:.3f}'.format(val_loss))
                    model.save_weights(log_dir + 'best_model.h5')
                else:
                    patience -= 1
                    if patience == 0:
                        log('Early stopping (best val_loss: {})'.format(best_val_loss))
                        break
                model_loss = 0
                model_acc = 0
        avg_tr_time = (time.time() - start_time) / (current_batch / batches_in_epoch)
        log('Training time per epoch {:.3f}'.format(avg_tr_time))

        ########################################################################
        # EVALUATE MODEL
        ########################################################################
        # Load best model
        model.load_weights(log_dir + 'best_model.h5')

        # Test model
        log('Testing model')
        test_loss, test_acc = evaluate(A_test, X_test, y_test, [loss, acc])
        log('Done.\n'
            'Test loss: {:.2f}\n'
            'Test acc: {:.2f}'
            .format(test_loss, test_acc))
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        K.clear_session()

    P['test_loss_mean'] = np.mean(results['test_loss'])
    P['test_loss_std'] = np.std(results['test_loss'])
    P['test_acc_mean'] = np.mean(results['test_acc'])
    P['test_acc_std'] = np.std(results['test_acc'])

    if df_out is None:
        df_out = pd.DataFrame([P])
    else:
        df_out = pd.concat([df_out, pd.DataFrame([P])])
    df_out.to_csv(log_dir + 'results.csv')
