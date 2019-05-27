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
from .network import DecayModelAutoencoder
from .layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer, ElementwiseDense

PiAct = lambda x: tf.clip_by_value(x, 0, 0.95) 

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
                         
    adata.raw = adata
    if pred_adata or pred_mtx_file:
        if pred_adata is None:
            pred_adata = anndata.read_mtx(pred_mtx_file).transpose()
        pred_adata.uns['data_type'] = 'UMI'
        pred_adata = read_dataset(pred_adata,
                transpose=False,
                test_split=False, 
                verbose = verbose,
                copy=False)
 

    tmpX = adata.X.A
    tmpX = tf.convert_to_tensor(tmpX, dtype = np.float32)
    curve = tf.cast(curve, tf.float32)
    pi = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2])*tmpX))
    net = DecayModelAutoencoder(curve = curve, pi = pi, input_size=adata.n_vars, nonmissing_indicator=nonmissing_indicator)
    net.build()
    print("going into training..")
    
    loss = train(adata[adata.obs.DCA_split == 'train'], 
            net, 
            output_dir=out_dir, 
            batch_size = batch_size,
            save_weights = True, 
            nonmissing_indicator = nonmissing_indicator,
            verbose = verbose, verbose_sum = verbose_sum, verbose_fit = verbose_fit)
 
    net.load_weights("%s/weights.hdf5" % out_dir)
    
    if pred_adata or pred_mtx_file:
        del adata
        res = net.predict(pred_adata)
        output_dispersion = res['dispersion']
        output_mean = res['mean_norm']
        outputmean_tensor = tf.convert_to_tensor(output_mean)
        output_pi_tensor = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2])*outputmean_tensor))
        output_pi = (tf.Session().run(output_pi_tensor))
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
    output_dispersion = res['dispersion']
    output_mean = res['mean_norm']
    output_pi = res['pi']
    outputmean_tensor = tf.convert_to_tensor(output_mean)
    output_pi_tensor = PiAct(curve[1] * K.exp(curve[0]-K.exp(curve[2])*outputmean_tensor))
    output_pi = (tf.Session().run(output_pi_tensor))
    

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

