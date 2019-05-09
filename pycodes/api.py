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
sys.stdout.flush()

from .io import read_dataset, normalize, write_text_matrix
from .train import train
from .network import NBConstantDispAutoencoder, ZINBConstantDispAutoencoder, DecayModelAutoencoder
from .layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer, ElementwiseDense

PiAct = lambda x: tf.clip_by_value(x, 0, 1) 

def autoencode(adata = None,
               curve_file_name = None,
               mtx_file = None,
               pred_adata=None, ## cross-validation purpose
               pred_mtx_file = None,
               out_dir=".",
               write_output_to_tsv = False,
               save_data = False,
               verbose = True, verbose_sum = True, verbose_fit = 1, 
               batch_size = 32,
               seed = 1,
               data_name = "",
               nonmissing_indicator = None): ###############
               
    print(out_dir)
               
    curve = np.loadtxt(curve_file_name)
    print(curve)

    if adata is None:
        if mtx_file is None:
            print('Either adata or mtx_file should be provided')
            return
        adata = anndata.read_mtx(mtx_file).transpose()
        if data_name == "":
            data_name = re.sub(r'.*/', '', mtx_file)
            data_name = data_name.replace('.mtx', '') + '_'

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

    adata.uns['data_type'] = 'UMI'
 

    # set seed for reproducibility
    np.random.seed(seed)
    tf.set_random_seed(seed)

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         verbose = verbose,
                         copy=False)

    adata = normalize(adata,
                      filter_min_counts = True,
                      size_factors=True,
                      logtrans_input=True)

    if pred_adata or pred_mtx_file:
        if pred_adata is None:
            pred_adata = anndata.read_mtx(pred_mtx_file).transpose()
        pred_adata.uns['data_type'] = 'UMI'
        pred_adata = read_dataset(pred_adata,
                transpose=False,
                test_split=False, 
                verbose = verbose,
                copy=False)
        pred_adata = normalize(pred_adata,
                size_factors=True,
                logtrans_input=True)
 

    tmpX = adata.X.A
    tmpX = tf.convert_to_tensor(tmpX, dtype = np.float32)
    curve = tf.cast(curve, tf.float32)
    pi = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2])*tmpX))
    
    net = DecayModelAutoencoder(curve = curve, pi = pi, input_size=adata.n_vars, nonmissing_indicator=nonmissing_indicator)
    
    net.build()

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

    optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    net.model.compile(loss=loss, optimizer=optimizer)

    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}
    output = adata.raw.X
    validation_split = 0.1
    callbacks=[]
    
    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose_sum:
        net.model.summary()

    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}

    output = adata.raw.X

    if train_on_full:
        validation_split = 0
    else:
        validation_split = 0.1

    loss = net.model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose_fit)

    net.load_weights("%s/weights.hdf5" % out_dir)
    
    if pred_adata or pred_mtx_file:
        del adata
        res = net.predict(pred_adata)
        
        output_mean = res['mean_norm']
        output_dispersion = res['dispersion']
        output_pi = res['pi']
        
        pred_adata.obsm['X_dca'] = res['mean_norm']
        del net,loss
        gc.collect()

        if write_output_to_tsv:
            print('Saving files ...')
            write_text_matrix(res['mean_norm'], 
                    os.path.join(out_dir, data_name + 'pred_mean_norm.tsv'))

        if save_data:
            with open(os.path.join(out_dir, data_name + 'pred_adata.pickle'), 'wb') as f:
                pickle.dump(pred_adata, f, protocol=4)
                f.close()
        return output_mean, output_dispersion, output_pi


    res = net.predict(adata)
    
    output_mean = res['mean_norm']
    output_dispersion = res['dispersion']
    output_pi = res['pi']
    
    if write_output_to_tsv:
        print('Saving files ...')
        write_text_matrix(res['mean_norm'], 
                    os.path.join(out_dir, data_name + 'mean_norm.tsv'))
        write_text_matrix(res['dispersion'], 
                    os.path.join(out_dir, data_name + 'dispersion.tsv'))
    if save_data:
            with open(os.path.join(out_dir, data_name + 'adata.pickle'), 'wb') as f:
                pickle.dump(adata, f, protocol=4)
                f.close()

    del net,loss
    gc.collect()
    
    return output_mean, output_dispersion, output_pi
