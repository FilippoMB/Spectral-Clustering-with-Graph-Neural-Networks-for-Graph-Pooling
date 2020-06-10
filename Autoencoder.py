import matplotlib.pyplot as plt
import numpy as np
from keras import Input, Model
from keras import backend as K
from keras.backend import tf
from keras.layers import Lambda
from pygsp import graphs
from spektral.layers import GraphConvSkip
from spektral.layers import MinCutPool
from spektral.utils.convolution import normalized_adjacency
from tqdm import tqdm

from utils.misc import sp_matrix_to_sp_tensor_value, get_sw_key


def upsampling_from_mask(inputs):
    X_, A_, I_, M_ = inputs
    S_ = tf.eye(tf.shape(M_)[0])
    S_ = tf.boolean_mask(S_, M_)
    S_t_ = tf.transpose(S_)

    X_out_ = K.dot(S_t_, X_)
    A_out_ = K.dot(K.transpose(K.dot(A_, S_)), S_)
    I_out_ = K.dot(
        S_t_,
        K.cast(I_[:, None], tf.float32)
    )[:, 0]
    I_out_ = K.cast(I_out_, tf.int32)
    return [X_out_, A_out_, I_out_]


def upsampling_from_matrix(inputs):
    X_, A_, I_, S_ = inputs
    X_out_ = K.dot(S_, X_)
    A_out_ = K.dot(S_, K.transpose(K.dot(S_, A_, )))
    I_out_ = K.dot(
        S_,
        K.cast(I_[:, None], tf.float32)
    )[:, 0]
    I_out_ = K.cast(I_out_, tf.int32)
    return [X_out_, A_out_, I_out_]


upsampling_from_mask_op = Lambda(upsampling_from_mask)
upsampling_from_matrix_op = Lambda(upsampling_from_matrix)

# HYPERPARAMS
ITER = 10000
ACTIV = 'tanh'
dataset = 'grid'
gnn_channels = 32
es_patience = 1000

# LOAD DATASET
if dataset == 'ring':
    G = graphs.Ring(N=200)
elif dataset == 'grid':
    G = graphs.Grid2d(N1=30, N2=30)
X = G.coords.astype(np.float32)
A = G.W
y = np.zeros(X.shape[0])  # X[:,0] + X[:,1]
n_classes = np.unique(y).shape[0]
n_feat = X.shape[-1]
n_nodes = A.shape[0]

# MODEL DEFINITION
X_in = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_in'))
A_in = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), name='A_in')
I_in = Input(tensor=tf.placeholder(tf.int32, shape=(None,), name='segment_ids_in'))
X_target = Input(tensor=tf.placeholder(tf.float32, shape=(None, n_feat), name='X_target'))
A_target = Input(tensor=tf.sparse_placeholder(tf.float32, shape=(None, None)), name='A_target')
A = normalized_adjacency(A)
n_out = X.shape[-1]

# encoder
X1 = GraphConvSkip(gnn_channels, activation=ACTIV)([X_in, A_in])
X1 = GraphConvSkip(gnn_channels, activation=ACTIV)([X1, A_in])
# pooling
X2, A2, I2, M2 = MinCutPool(k=n_nodes // 4, h=gnn_channels)([X1, A_in, I_in])
# unpooling
X3, A3, I3 = upsampling_from_matrix_op([X2, A2, I2, M2])
# decoder
X3 = GraphConvSkip(gnn_channels, activation=ACTIV)([X3, A_in])
X3 = GraphConvSkip(gnn_channels, activation=ACTIV)([X3, A_in])
X3 = GraphConvSkip(n_out)([X3, A_in])

model = Model([X_in, A_in, I_in], [X3])
model.compile('adam', 'mse', target_tensors=[X_target])

# TRAINING
sess = K.get_session()
loss = model.total_loss
opt = tf.train.AdamOptimizer(learning_rate=5e-3)
train_step = opt.minimize(loss)

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Fit layer
tr_feed_dict = {X_in: X,
                A_in: sp_matrix_to_sp_tensor_value(A),
                I_in: y,
                X_target: X,
                get_sw_key(sess): np.ones(1)}
layer_out = []
patience = es_patience
best_loss = np.inf
tol = 1e-5
iterator = tqdm(range(ITER))
try:
    for _ in iterator:
        outs = sess.run([train_step, loss], feed_dict=tr_feed_dict)
        layer_out.append(outs[1])
        if outs[1] + tol < best_loss:
            best_loss = outs[1]
            patience = es_patience
            model.save_weights('best.h5')
        else:
            patience -= 1
            if patience == 0:
                iterator.close()
                break
except KeyboardInterrupt:
    print('training interrupted!')

# Evaluate
model.load_weights('best.h5')
pred = sess.run([model.output], feed_dict=tr_feed_dict)[0]
mask = sess.run([M2], feed_dict=tr_feed_dict)[0]
output = {'loss': layer_out, 'pred': pred, 'mask': mask}
lss_ = sess.run([model.total_loss], feed_dict=tr_feed_dict)[0]
print('MSE', lss_)
K.clear_session()

# PLOTS
plt.plot(output['loss'])
plt.title('Loss')
plt.figure(figsize=(8, 4))
pad = 0.1
x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
colors = X[:, 0] + X[:, 1]
plt.subplot(1, 2, 1)
plt.scatter(*X[:, :2].T, c=colors, s=8, zorder=2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Original')
plt.axvline(0, c='k', alpha=0.2)
plt.axhline(0, c='k', alpha=0.2)
plt.subplot(1, 2, 2)
plt.scatter(*output['pred'][:, :2].T, c=colors, s=8, zorder=2)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title('Reconstructed')
plt.axvline(0, c='k', alpha=0.2)
plt.axhline(0, c='k', alpha=0.2)
plt.tight_layout()
plt.show()
