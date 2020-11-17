import numpy as np
from sklearn.cross_decomposition import CCA
import scipy.stats

def compose_feats_matrix(X1, X2, comp = "concat", th_corrcoef = 0., verbose = False):
    '''
    X1: first view of the data
    X2: second view of the same data
    X1 and X2 should have the same dimension in axis=1 (no validation is performed here)
    comp: ["concat" or "cca"], indicates what kind of composition we want to perform
    th_corrcoef: if a number bigger than zero is given, it behaves as a threshold for the CCA matrix and
                 only returns the dimensions with a higher correlation coefficient than this value [0,1]
    verbose: to print extra information

    returns:
        "concat": concatenated matrix
        "cca":
            X1_cca: first view transformed in the new shared space
            cca: model to transform other samples from the first space
            list_corrcoef: array with all the correlation-coefficients to further analysis
    '''
    if comp == "concat":
        return np.concatenate((X1, X2), axis = 1)
    
    elif comp == "cca":
        #We check all columns of X1 to avoid constant feats
        x_const = []
        for i in range(X1.shape[1]):
            x_col = list(X1[:,i])
            if (len(list(set(x_col))) < 2):
                x_const.append(i)        
        if verbose:
            print("-- CCA warning: we are deleting %d of %d features from X1 --" % (len(x_const),X1.shape[1]))
        X1 = np.delete(X1, x_const, axis=1)
    
        nc = min(X1.shape[1], X2.shape[1])
        #if verbose:
        #    print("n_components =", nc)
        cca = CCA(n_components = nc)
        try:
            cca.fit(X1, X2)
        except np.linalg.LinAlgError:
            print("SVD did not converge")
            return None, None, []
        X1_cca, X2_cca = cca.transform(X1, X2)
        
        if th_corrcoef > 0.:
            #We only retrieve dims with coef > th_corrcoef
            dim = 0
            dims_delete = []
            list_corrcoef = []
            for idx, (x_c, y_c) in enumerate(zip(X1_cca.T, X2_cca.T)):
                corrcoef, _ = scipy.stats.pearsonr(x_c, y_c)
                list_corrcoef.append(corrcoef)
                # to control dimensions with correlation "1.", but they are after shorter corr dims.
                #if idx > 0 and corrcoef == 1. and list_corrcoef[idx-1] < 1.:
                #    break
                if corrcoef >= th_corrcoef:
                    dim += 1
                if corrcoef < th_corrcoef or np.isnan(corrcoef):
                    dims_delete.append(idx)
            #X1_cca = X1_cca[:, :dim]
            X1_cca = np.delete(X1_cca,dims_delete,axis=1)
            for index in sorted(dims_delete, reverse=True):
                del list_corrcoef[index]
            #list_corrcoef = np.delete(list_corrcoef, dims_delete,axis=1)
            if verbose:
                print("    th_cc =", th_corrcoef, "; n_dim =", len(list_corrcoef), list_corrcoef)

        return X1_cca, cca, np.array(list_corrcoef), x_const