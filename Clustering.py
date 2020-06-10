from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from pygsp import graphs
from sklearn.cluster import spectral_clustering
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
from sklearn.neighbors import kneighbors_graph
from spektral.layers import MinCutPool, DiffPool
from spektral.layers.convolutional import GraphConvSkip
from spektral.utils import init_logging
from spektral.utils.convolution import normalized_adjacency
from tqdm import tqdm

from utils import citation
from utils.misc import sp_matrix_to_sp_tensor_value, product_dict

np.random.seed(0)  # for reproducibility

PLOTS_ON = True
ITER = 10000
plt.set_cmap('nipy_spectral')
VERBOSE = False

# Parameters
P = OrderedDict([
    ('apply_GNN', True),
    ('ACTIV', 'elu'),
    ('es_patience', ITER)
])
log_dir = init_logging()  # Create log directory and file

# Tunables
tunables = OrderedDict([
    ('dataset', ['cora']),  # 'cora', 'citeseer', 'pubmed', 'cloud', or 'synth'
    ('method', ['mincut_pool']),  # 'mincut_pool', 'diff_pool'
    ('H_', [None]),
    ('n_channels', [16]),
    ('learning_rate', [5e-4])
])

N_RUNS = 1
df_out = None
for T in product_dict(tunables):
    # Update params with current config
    P.update(T)
    print(T)

    ############################################################################
    # LOAD DATASET
    ############################################################################
    if P['dataset'] == 'synth':
        X, y = make_blobs(n_samples=100, centers=5, n_features=4, random_state=None)  # 6
        X = X.astype(np.float32)
        A = kneighbors_graph(X, n_neighbors=25, mode='distance').todense()
        A = np.asarray(A)
        A = np.maximum(A, A.T)
        A = sp.csr_matrix(A, dtype=np.float32)
        n_clust = y.max() + 1
    elif P['dataset'] == 'cloud':
        G = graphs.Grid2d(N1=15, N2=10)  # Community(N=150, seed=0) #SwissRoll(N=400, seed=0) #Ring(N=100) #TwoMoons()  #Cube(nb_pts=500)  #Bunny()
        X = G.coords.astype(np.float32)
        A = G.W
        y = np.ones(X.shape[0])  # X[:,0] + X[:,1]
        n_clust = 5
    else:
        A, X, _, _, _, _, _, _, y_ohe = citation.load_data(P['dataset'])
        y = np.argmax(y_ohe, axis=-1)
        X = X.todense()
        n_clust = y.max() + 1

    # Sort IDs
    if P['dataset'] != 'cloud':
        ids = np.argsort(y)
        y = y[ids]
        X = X[ids, :]
        A = A[ids, :][:, ids]
        A = sp.csr_matrix(A.todense())
    n_feat = X.shape[-1]

    homo_score_list = []
    complete_score_list = []
    v_score_list = []
    for run in range(N_RUNS):
        ############################################################################
        # MODEL
        ############################################################################
        X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_in'))
        A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), name='A_in', sparse=True)
        S_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))

        if P['apply_GNN'] and P['method'] != 'diff_pool':
            A_norm = normalized_adjacency(A)
            X_1 = GraphConvSkip(P['n_channels'],
                                kernel_initializer='he_normal',
                                activation=P['ACTIV'])([X_in, A_in])
        else:
            A_norm = A
            X_1 = X_in

        if P['method'] == 'mincut_pool':
            pool1, adj1, seg1, C = MinCutPool(k=n_clust,
                                              h=P['H_'],
                                              activation=P['ACTIV'])([X_1, A_in, S_in])

        elif P['method'] == 'diff_pool':
            pool1, adj1, seg1, C = DiffPool(k=n_clust,
                                            channels=P['n_channels'],
                                            activation=P['ACTIV'])([X_1, A_in, S_in])
        else:
            raise ValueError

        model = Model([X_in, A_in, S_in], [pool1, seg1, C])
        model.compile('adam', None)

        ############################################################################
        # TRAINING
        ############################################################################
        # Setup
        sess = K.get_session()
        loss = model.total_loss
        opt = tf.train.AdamOptimizer(learning_rate=P['learning_rate'])
        train_step = opt.minimize(loss)

        # Initialize all variables
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Fit layer
        tr_feed_dict = {X_in: X,
                        A_in: sp_matrix_to_sp_tensor_value(A_norm),
                        S_in: y}
        layer_out = []
        nmi_out = []
        best_loss = np.inf
        patience = P['es_patience']
        tol = 1e-5
        for _ in tqdm(range(ITER)):
            outs = sess.run([train_step, model.losses[0], model.losses[1], C], feed_dict=tr_feed_dict)
            layer_out.append((outs[1], outs[2], outs[1] + outs[2]))
            c = np.argmax(outs[3], axis=-1)
            v_score = v_measure_score(y, c)
            nmi_out.append(v_score)
            if outs[1] + outs[2] + tol < best_loss:
                best_loss = outs[1] + outs[2]
                patience = P['es_patience']
                if VERBOSE:
                    tqdm.write('New best loss {}'.format(best_loss))
            else:
                patience -= 1
                if VERBOSE:
                    tqdm.write('Patience {}'.format(patience))
                if patience == 0:
                    break
        layer_out = np.array(layer_out)

        ############################################################################
        # RESULTS
        ############################################################################
        C_ = sess.run([C], feed_dict=tr_feed_dict)[0]
        c = np.argmax(C_, axis=-1)
        hs = homogeneity_score(y, c)
        cs = completeness_score(y, c)
        nmis = v_measure_score(y, c)
        homo_score_list.append(hs)
        complete_score_list.append(cs)
        v_score_list.append(nmis)

        print('MinCutPool - HOMO: {}, CS: {}, NMI: {}'.format(hs, cs, nmis))
        np.savez(log_dir + 'loss+nmi_{}_{}_{}.npz'.format(P['dataset'], P['method'], run),
                 loss=layer_out, nmi=nmi_out)
        K.clear_session()

    P['homo_score_mean'] = np.mean(homo_score_list)
    P['homo_score_std'] = np.std(homo_score_list)
    P['complete_score_mean'] = np.mean(complete_score_list)
    P['complete_score_std'] = np.std(complete_score_list)
    P['v_score_mean'] = np.mean(v_score_list)
    P['v_score_std'] = np.std(v_score_list)

    if PLOTS_ON:
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plt.plot(layer_out[:, 2], label='Loss')
        plt.legend()
        plt.ylabel('Loss')
        plt.xlabel('Iteration')

        plt.subplot(122)
        plt.plot(nmi_out, label='NMI')
        plt.legend()
        plt.ylabel('NMI')
        plt.xlabel('Iteration')
        plt.tight_layout()
        plt.savefig(log_dir + 'loss+nmi_{}_{}.pdf'.format(P['dataset'], P['method']), bbox_inches='tight')

        if P['dataset'] == 'synth':
            plt.figure()
            plt.scatter(X[:, 0], X[:, 1], c=c)
            plt.title('GNN-pool clustering')
        if P['dataset'] == 'cloud':
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            G.plot_signal(c, vertex_size=30, plot_name='', colorbar=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig('logs/grid_mincut.pdf', bbox_inches='tight', pad_inches=0)
        plt.show()

    # Spectral clustering
    sc = spectral_clustering(A, n_clusters=n_clust, eigen_solver='arpack')
    P['homo_score_sc'] = homogeneity_score(y, sc)
    P['complete_score_sc'] = completeness_score(y, sc)
    P['v_score_sc'] = v_measure_score(y, sc)

    print('Spectral Clust - HOMO: {:.3f}, CS: {:.3f}, NMI: {:.3f}'
          .format(P['homo_score_sc'], P['complete_score_sc'], P['v_score_sc']))

    if df_out is None:
        df_out = pd.DataFrame([P])
    else:
        df_out = pd.concat([df_out, pd.DataFrame([P])])
    df_out.to_csv(log_dir + 'results.csv', index=False)

    if PLOTS_ON:
        if P['dataset'] == 'synth':
            plt.scatter(X[:, 0], X[:, 1], c=sc)
            plt.title('Spectral clustering')
            plt.show()
        if P['dataset'] == 'cloud':
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            G.plot_signal(sc, vertex_size=30, plot_name='', colorbar=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.tight_layout()
            plt.savefig('logs/grid_spectral.pdf', bbox_inches='tight', pad_inches=0)
    K.clear_session()
