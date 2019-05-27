import os, tempfile, shutil
import anndata
import numpy as np
from numpy import asarray
import tensorflow as tf
import pandas as pd
import pickle
import re, gc
import sys
from keras import backend as K
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator
from importlib import reload
sys.stdout.flush()

from pycodes.io import read_dataset, normalize, write_text_matrix
from pycodes.train import train
from pycodes.network import NBConstantDispAutoencoder, ZINBConstantDispAutoencoder, DecayModelAutoencoder
from pycodes.layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer, ElementwiseDense

PiAct = lambda x: tf.clip_by_value(x, 0, 0.95) 

mtx_file = "../data/10X_pbmc_filtered/SAVERX_temp.mtx"
curve_file_name = "../data/10X_pbmc_filtered/matrix_curve.txt"
pred_mtx_file = "../data/10X_pbmc_filtered/SAVERX_temp_test.mtx"
nonmissing_indicator = 1,
out_dir = '../data/10X_pbmc_filtered'
batch_size = 261
write_output_to_tsv = False

save_data = True
verbose = True
verbose_sum = True
verbose_fit = 1
seed = 1
data_name=""

curve = np.loadtxt(curve_file_name)
print(curve)

adata = anndata.read_mtx(mtx_file).transpose()

assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

# set seed for reproducibility
np.random.seed(seed)
tf.set_random_seed(seed)

adata = read_dataset(adata,
                      transpose=False,
                      test_split=False,
                      verbose = verbose,
                      copy=False)


pred_adata = anndata.read_mtx(pred_mtx_file).transpose()
pred_adata = read_dataset(pred_adata,transpose=False,test_split=False, verbose = verbose,copy=False)
 

tmpX = adata.X.A
tmpX = tf.convert_to_tensor(tmpX, dtype = np.float32)
curve = np.loadtxt(curve_file_name)
curve = tf.convert_to_tensor(curve, dtype = np.float32)
pi = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2])*tmpX))

net = DecayModelAutoencoder(curve = curve, pi = pi, input_size=adata.n_vars,nonmissing_indicator=nonmissing_indicator)
net.build()

# network = net
loss = net.loss
adata = adata[adata.obs.DCA_split == 'train']

optimizer = 'rmsprop'
learning_rate=None
train_on_full=False
epochs=300
reduce_lr=4
early_stop = 6
clip_grad=5
save_weights=False
nonmissing_indicator = None
tensorboard=False
verbose=True
verbose_sum=True
verbose_fit=1

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator

loss = net.loss
    
output_dir = out_dir
os.makedirs(output_dir, exist_ok=True)

optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)

net.model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
callbacks = []



checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
reduce_lr = 4
early_stop = 6

callbacks.append(checkpointer)
lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=True, verbose=verbose)
callbacks.append(lr_cb)

es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
callbacks.append(es_cb)



net.model.summary()

inputs = {'count': adata.X.A}

output = adata.X.A

validation_split = 0.1
epochs=300
loss = net.model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose_fit,
                     **kwargs)

net.load_weights("%s/weights.hdf5" % out_dir)

res = net.predict(adata)
output_mean = res['mean_norm']
output_dispersion = res['dispersion']
output_pi = res['pi']

del net,loss
gc.collect()

return output_mean, output_dispersion, output_pi
