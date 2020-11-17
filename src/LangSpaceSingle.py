import numpy as np
from .utils import load_features
from .utils import svd_transform, trunc_svd_by_th
from .ClusterData import ClusterData
from .wit3langs import wit3_25_langs, wit3_emnlp_ids
from .tedlangs import ted_iso_langs
from statistics import mean, median, stdev
from scipy.stats import median_absolute_deviation
from random import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class LangSpaceSingle:
    def __init__(self, space_name, source_name, attributes=None, X_total=None, label_total=None, svd_transform=True, std_scaler=True, minmax_scaler=True, truth_value=False):
        '''
        space_name = ["uriel", "learn"]
            source_name : ["syntax_knn"]
            attributes: {"method": "average", "metric": "cosine", "svd": }
            source_name : ["Bib22", "Bib24", "Tan23", "Own23", "Own25", "Fact23", "Fact25"]
            attributes: {"dir": "en2x", "method": "average", "metric": "cosine", "svd": }
        '''
        self.space_name = space_name
        self.source_name = source_name
        self.attributes = attributes

        if X_total is not None and label_total is not None:
            self.X_total = X_total
            self.label_total = label_total
        else:
            #print("loading features...")
            self.X_total, self.label_total = load_features(self.space_name, self.source_name)
        self.num_langs = len(self.label_total)

        # truth values
        self.X_truth = None
        if space_name == "uriel" and truth_value:
            self.X_truth, _ = load_features(self.space_name, self.source_name.split("_")[0] + "_wals")
        
        # attributes for SVD ....
        self.X_svd_total = None
        self.svd_model = None
        if svd_transform:
            self.compute_SVD()
        
        self.scaler = None
        self.X_svd_total_scaled = None
        if std_scaler:
            self.compute_StandardScaler()
        self.mm_scaler = None
        self.X_svd_total_mm_scaled = None
        if minmax_scaler:
            self.compute_MinMaxScaler()
        

    '''
    def get_X_filter(self):
        if len(self.label_total) == self.num_langs:
            return self.X_total
        else:
            X_filter = []
            for l in self.label_filter:
                X_filter.append(self.X_svd_total[self.label_total.index(l)])
            return X_filter
    '''
    def compute_StandardScaler(self):
        self.scaler = StandardScaler().fit(self.X_svd_total)
        self.X_svd_total_scaled = self.scaler.transform(self.X_svd_total)
    def compute_MinMaxScaler(self):
        self.mm_scaler = MinMaxScaler().fit(self.X_svd_total)
        self.X_svd_total_mm_scaled = self.mm_scaler.transform(self.X_svd_total)

    def compute_SVD(self):
        X_svd, svd_model = svd_transform(self.X_total)
        self.X_svd_total = X_svd
        self.svd_model = svd_model
        '''
        # if there is nothing to filter....
        if len(self.label_total) == len(self.label_filter):
            self.X_svd_filter = self.X_svd_total
        else: # otherwise ...
            self.X_svd_filter = []
            for l in self.label_filter:
                self.X_svd_filter.append(self.X_svd_total[self.label_total.index(l)])
        '''

    def print_status(self):
        print("space = %s; source = %s" % (self.space_name, self.source_name))
        print("  #lang = %d; X.shape = (%d,%d)" % (len(self.label_total), self.X_total.shape[0], self.X_total.shape[1]))
        print("  if svd_th=0.75 -> X_svd.shape[1] = %d" % (trunc_svd_by_th(self.svd_model, 0.75)))

    def getLabels(self):
        return self.label_total
    
    def getSourceName(self):
        return self.source_name
    
    def get_X_truth(self, label_filter):
        X = self.X_truth
        X_filter = []
        for l in label_filter:
            X_filter.append(X[self.label_total.index(l)])
        return np.array(X_filter)

    def get_X_filter(self, label_filter, std_scaler=False, minmax_scaler=False, out_of_feats=[]):
        if (not std_scaler and not minmax_scaler) or self.space_name == "uriel":
            X = np.copy(self.X_total)
        elif std_scaler:
            X = StandardScaler().fit_transform(self.X_total)
        elif minmax_scaler:
            X = MinMaxScaler().fit_transform(self.X_total)
        if len(out_of_feats) > 0 :
            X = np.delete(X, out_of_feats, axis=1)
        X_filter = []
        for l in label_filter:
            X_filter.append(X[self.label_total.index(l)])
        return np.array(X_filter)
    
    def get_X_svd_filter(self, label_filter, svd_th=1.,std_scaler=False, minmax_scaler=False, out_of_feats=[]):
        # elemental case: no SVD, no Scaling (even if the parameter is True)
        if svd_th == 1.:
            return self.get_X_filter(label_filter, std_scaler=std_scaler, minmax_scaler=minmax_scaler, out_of_feats=out_of_feats)
        if len(out_of_feats) > 0:
            # analysing if we use an scaled version (zero mean)
            if std_scaler:
                X_total = self.X_svd_total_scaled
            elif minmax_scaler:
                X_total = self.X_svd_total_mm_scaled
            else:
                X_total = self.X_svd_total
            # obtaining the data
            X_filter = []
            for l in label_filter:
                X_filter.append(X_total[self.label_total.index(l)])
            X_filter = np.array(X_filter)
            if svd_th < 1.:
                dimensions = trunc_svd_by_th(self.svd_model, svd_th)
                X_filter = X_filter[:, :dimensions]
            return X_filter
        else:
            # we need to eliminate the features that are related
            X_total = np.copy(self.X_total)
            X_total = np.delete(X_total, out_of_feats, axis=1)
            X_svd_total, svd_model = svd_transform(X_total)
            X_filter = []
            for l in label_filter:
                X_filter.append(X_svd_total[self.label_total.index(l)])
            X_filter = np.array(X_filter)
            if svd_th < 1.:
                dimensions = trunc_svd_by_th(svd_model, svd_th)
                X_filter = X_filter[:, :dimensions]
            return X_filter

    def get_X_dict(self, label_filter, svd_th=1.,std_scaler=False, minmax_scaler=False, out_of_feats=[]):
        if label_filter is None:
            label_filter = self.label_total
        X = self.get_X_svd_filter(label_filter, svd_th=svd_th,std_scaler=std_scaler, minmax_scaler=minmax_scaler, out_of_feats=out_of_feats)
        X_dict = {}
        for i, l in enumerate(label_filter):
            X_dict[l] = X[i]
        return X_dict

    def compute_single_cluster(self, label_filter=None, svd_th=.75, method="average", metric="cosine", std_scaler=False, minmax_scaler=False):
        cluster = None
        if self.space_name == "uriel":
            name = self.space_name
        else:
            name = self.source_name
        if label_filter is not None:
            X_svd_filter = self.get_X_svd_filter(label_filter, svd_th=svd_th, std_scaler=std_scaler, minmax_scaler=minmax_scaler)
            #if std_scaler:
            #    X_svd_filter = StandardScaler().fit_transform(X_svd_filter)
            cluster = ClusterData(X_svd_filter, label_filter, name=name+ "_%.2f"%svd_th, max_clusters=len(label_filter)-1, method=method, metric=metric)
        else:
            X_svd_filter = self.get_X_svd_filter(self.label_total, svd_th=svd_th, std_scaler=std_scaler, minmax_scaler=minmax_scaler)
            cluster = ClusterData(X_svd_filter, self.label_total, name=name+"_all_%.2f"%svd_th, max_clusters=22, method=method, metric=metric)
        return cluster

    def compute_incremental(self, svd_th=.75, verbose=False, iter=30, metric="cosine", method="average", label_filter=wit3_emnlp_ids, std_scaler = False):

        #if self.space_name != "uriel":
        #    print("Only URIEL allows this incremental...")
        #    exit(1)
        #if metric=="hamming" and svd_th!=1.0:
        #    print("Hamming metric only works with binary values...")
        #    exit(1)
        
        if svd_th == 1.: #for both uriel and learn
            std_scaler = False

        #label_filter = wit3_emnlp_ids
        name = self.space_name + "_%.2f" % (svd_th)
        # We obtain the vectors for 25 languages
        X_uriel_filter = self.get_X_svd_filter(label_filter, svd_th)

        coph_corr = []
        max_sil_3 = []
        ncluster_3 = []
        max_sil_2 = []
        ncluster_2 = []
        
        #First, we get the languages out of the filter list...
        label_total = ted_iso_langs.copy()
        if self.source_name == "Tan23":
            label_total = wit3_emnlp_ids.copy()
        if self.source_name == "Bible":
            lst_remove = ["aze", "mon", "sqi", "bos", "glg", "kat", "kaz", "kur", "mar", "zlm"]
            for l in lst_remove:
                label_total.remove(l)
        label_extra = sorted(list(set(label_total) - set(label_filter)))
        num_lang_extra = len(label_extra)
        #print(len(label_filter), len(label_extra), label_extra)

        #Second, we extract those languages from the Uriel View with svd_th
        X_uriel_extra = self.get_X_svd_filter(label_extra, svd_th)
    
        idx_original = list(np.arange(num_lang_extra))
        for it in range(iter):
            if verbose:
                    print("--iteration %02d/%02d..." % (it+1,iter) )
            if it > 0:
                shuffle(idx_original)
                X_uriel_shuffle = []
                label_shuffle = []
                for idx in idx_original:
                    X_uriel_shuffle.append(X_uriel_extra[idx])
                    label_shuffle.append(label_extra[idx])
                X_uriel_shuffle = np.array(X_uriel_shuffle)
            else:
                X_uriel_shuffle = X_uriel_extra
                label_shuffle = label_extra
                
            #Adding CCA data:
            # Perform normalization (zero mean), with exception of URIEL_1.0
            if std_scaler:
                cluster_uriel = ClusterData(StandardScaler().fit_transform(X_uriel_filter), label_filter, name="_%2d"%len(label_filter)+name, max_clusters=22, metric=metric, method=method)
            else:
                cluster_uriel = ClusterData(X_uriel_filter, label_filter, name="_%2d"%len(label_filter)+name, max_clusters=22, metric=metric, method=method)
            coph_corr.append([cluster_uriel.cophenetic_corr()])
            max_sil_3.append([cluster_uriel.max_silhouette(begin=3)])
            ncluster_3.append([cluster_uriel.num_clusters_by_sil(begin=3)])
            max_sil_2.append([cluster_uriel.max_silhouette()])
            ncluster_2.append([cluster_uriel.num_clusters_by_sil()]) 
            
            for i in range(num_lang_extra):
                X_uriel_inc = np.concatenate((X_uriel_filter, X_uriel_shuffle[:i+1, :]), axis=0)
                if std_scaler:
                    X_uriel_inc = StandardScaler().fit_transform(X_uriel_inc)
                cl_uriel_inc = ClusterData(X_uriel_inc, label_filter + label_shuffle[:i+1], name="_%2d"%(len(label_filter)+i+1)+name, max_clusters=22, metric=metric, method=method)
                coph_corr[-1].append(cl_uriel_inc.cophenetic_corr())
                max_sil_3[-1].append(cl_uriel_inc.max_silhouette(begin=3))
                ncluster_3[-1].append(cl_uriel_inc.num_clusters_by_sil(begin=3))
                max_sil_2[-1].append(cl_uriel_inc.max_silhouette())
                ncluster_2[-1].append(cl_uriel_inc.num_clusters_by_sil()) 
        
        #return (coph_corr, max_sil_3, ncluster_3, max_sil_2, ncluster_2)
        # transform to np.array and transpose to shape: (langs, iter)
        coph_corr = np.array(coph_corr).T
        max_sil_3 = np.array(max_sil_3).T
        ncluster_3 = np.array(ncluster_3).T
        max_sil_2 = np.array(max_sil_2).T
        ncluster_2 = np.array(ncluster_2).T

        #print(coph_corr.shape, max_sil_3.shape, ncluster_3.shape, max_sil_2.shape, ncluster_2.shape)
        scores_inc = []
        for i in range(len(label_extra)+1):
            '''
            try:
                #s1 = mean(coph_corr[i])
                s2 = mean(max_sil_3[i])
                #s3 = mean(max_sil_2[i])
            except: 
                print(i, max_sil_3[i])
                s2 = np.mean(max_sil_3[i])
                print(s2)
                #exit(1)
            '''
            scores_inc.append((np.mean(coph_corr[i]),np.std(coph_corr[i]), np.mean(max_sil_3[i]), np.std(max_sil_3[i]), 
                                                        median(ncluster_3[i]), median_absolute_deviation(ncluster_3[i]), 
                                                        np.mean(max_sil_2[i]), np.std(max_sil_2[i]), 
                                                        median(ncluster_2[i]), median_absolute_deviation(ncluster_2[i])) )    
        comp_dims = X_uriel_filter.shape[1]
        return scores_inc, comp_dims