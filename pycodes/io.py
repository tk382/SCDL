import numpy as np
import scipy as sp
import pandas as pd
import scanpy.api as sc
from sklearn.model_selection import train_test_split


def read_dataset(adata, transpose=False, test_split = False, copy=False, verbose=True):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    # 
    # type_error = "Make sure that the dataset is of one of the two types: UMI, nonUMI(RPKM/TPM)"
    # assert adata.uns['data_type'] in ['UMI', 'nonUMI'], type_error
    
    if transpose: adata = adata.transpose()
    
    if test_split:
        train_idx, test_idx = train_test_split(np.arrange(adata.n_obs), test_size=0.1, random_state = 42)
        spl = pd.Series(['train']*adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
        adata.uns['train_idx'] = train_idx
        adata.uns['test_idx'] = test_idx
    else:
        adata.obs['DCA_split'] = 'train'
    
    adata.obs['DCA_split'] = 'train'
    
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    
    if verbose:
        print("### Autoencoder: Successfully preprocessed {} genes and {} cells.".format(adata.n_vars, adata.n_obs), flush=True)
    return adata


def normalize(adata, filter_min_counts=True, logtrans_input=True):
    
    # remove genes that are 0 everywhere
    if filter_min_counts:
        sc.pp.filter_cells(adata, min_counts=1)
    
    # if nothing's stored as raw data, consider the current data as raw data
    if adata.raw is None:
            adata.raw = adata
    
    if adata.uns['data_type'] != 'nonUMI': 
        #if UMI, n_counts is pre-specified
        n_counts = adata.obs.n_counts
    else: 
        #if not UMI (TPM, RPKM), n_counts is the column sum
        n_counts = adata.X.sum(axis = 1)
        
    # log transform
    if logtrans_input:
        sc.pp.log1p(adata)
        
    return adata

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                                                                                                                                                                                                                        sep='\t',
                                                                                                                                                                                                                                                                        index=(rownames is not None),
                                                                                                                                                                                                                                                                        header=(colnames is not None),
                                                                                                                                                                                                                                                                        float_format='%.3f')































    
