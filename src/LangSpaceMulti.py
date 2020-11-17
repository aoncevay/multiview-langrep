import numpy as np
from .ClusterData import ClusterData
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .compose_matrix import compose_feats_matrix
from .wit3langs import wit3_25_langs, wit3_emnlp_ids
from .tedlangs import ted_iso_langs, all_iso_langs, ted_family2tag, ted_tag2iso
from statistics import mean, median, stdev
from scipy.stats import median_absolute_deviation
from random import shuffle
from .utils import get_related_features, load_families
list_baseline_langs = ['eng', 'swe', 'dan', 'deu', 'nld', 'ron', 'fra', 'ita', 'spa', 'por', 'lav', 'lit', 'pol', 'slk', 'ces', 'slv', 'bul']

class LangSpaceMulti:
    def __init__(self, name, space_1, space_2, label_filter=None, label_both=None, family_tags=False):
        '''
        name = any name for the object
        composition = CAT or CCA
        space_1 = LangSpaceSingle obj. from URIEL
        space_2 = LangSpaceSingle obj. from Learn
        label_filter = languages to extract from both views
        '''
        self.name = name
        self.spaceUriel = space_1
        self.spaceLearn = space_2

        if label_both is None:
            self.label_both = sorted(self.compute_label_both())
        else:
            self.label_both = sorted(label_both)

        if label_filter is None:
            #self.label_filter = sorted(self.spaceLearn.getLabels())
            self.label_filter = self.label_both
        else:
            self.label_filter = sorted(label_filter)
        
        if len(self.label_filter) > len(self.label_both):
            print("Warning: there are not enough lang. entries in one view...")
        
        self.num_langs = len(self.label_filter)
        self.no_filter = " ".join(self.label_filter) == " ".join(self.label_both)

        self.dict_families = None
        if family_tags:
            self.dict_families = load_families(self.label_both)
        
        # we identify the languages in label_both that have TRUTH values
        self.label_both_truth = None
        if self.spaceUriel.X_truth is not None:
            self.label_both_truth = self.compute_label_both_truth()


    def compute_label_both(self):
        return list(set(self.spaceUriel.getLabels()).intersection(self.spaceLearn.getLabels()))

    def getLabelsBoth(self):
        return self.label_both
    
    def getLabelsFilter(self):
        return self.label_filter
    
    def compute_label_both_truth(self):
        label_both_truth = []
        X_truth_both = self.spaceUriel.get_X_truth(self.label_both)
        for lang, truth_lang in zip(self.label_both, X_truth_both):
            # we look if all the row is empty
            values = list(set(truth_lang))
            if len(values) == 1 and values[0] == "--":
                continue
            else:
                label_both_truth.append(lang)
        return sorted(label_both_truth)

    def print_status(self):
        print("Multi-name = %s" % (self.name))
        print("  #lang_both = %d; #lang_filter = %d; both==filter? %s" % (len(self.label_both), len(self.label_filter), str(self.no_filter)))

    def get_X_svd_filter_both(self, th=1., space="uriel"):
        if space=="uriel":
            return self.spaceUriel.get_X_svd_filter(self.label_both, th)
        elif space=="learn":
            return self.spaceLearn.get_X_svd_filter(self.label_both, th)

    def get_X_cca_any(self, label_langs, svd_th=[.9, .9],  post_std_scaler=True, source="uriel", pre_std_scaler=False, pre_mm_scaler=False, candidates=None):
        # We pass a list of languages, and we are going to compute the CCA matrix, and project as many languages as we can
        X_uriel_both = self.spaceUriel.get_X_svd_filter(self.label_both, svd_th[0], pre_std_scaler)
        X_learn_both = self.spaceLearn.get_X_svd_filter(self.label_both, svd_th[1], pre_std_scaler)

        X_CCA_both, cca_model, corrcoefs_, col_const = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "cca", th_corrcoef=0.5)

        if label_langs is None:
            label_langs = list.copy(self.label_both)
        if candidates != "" and candidates is not None:
            candidates = candidates.split(",")
            label_langs.extend(candidates)
            label_langs = list(set(label_langs))
        # 1. We identify which languages are in self.label_both
        X_CCA_dict = {}
        for idx, l in enumerate(self.label_both):
            if l in label_langs:
                X_CCA_dict[l] = X_CCA_both[idx]

        print("Languages found in both views: %s" % (",".join(X_CCA_dict.keys())))
        # 2. From the rest, we identify which languages are in self.spaceUriel.label_total
        extra_langs = []
        invalid_langs = []
        for l in label_langs:
            if l not in X_CCA_dict.keys():
                if l in self.spaceUriel.label_total:
                    extra_langs.append(l)
                else:
                    invalid_langs.append(l)

        # 3. We build a dict using (1) and concatenate the extra entries for (2)
        # We extract those languages from the Uriel View with svd_th
        if len(extra_langs) > 0:
            if source == "uriel":
                X_uriel_extra = self.spaceUriel.get_X_svd_filter(extra_langs, svd_th[0], std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)
            elif source == "learn":
                X_uriel_extra = self.spaceLearn.get_X_svd_filter(extra_langs, svd_th[1], std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)
            X_uriel_extra = np.array(X_uriel_extra)
            # We need to delete the constant columns we detect in the CCA composition....
            if len(col_const) > 0:
                X_uriel_extra = np.delete(X_uriel_extra, col_const, axis=1)
            # We transform the vectors using the cca_model
            X_CCA_extra = cca_model.transform(X_uriel_extra)#.reshape(-1,1)
            # We append the vectors in the CCA
            if X_CCA_extra.shape[1] > X_CCA_both.shape[1]:
                X_CCA_extra = X_CCA_extra[:, :X_CCA_both.shape[1]]
            
            for i, l in enumerate(extra_langs):
                X_CCA_dict[l] = X_CCA_extra[i]

        # 4. Print a message for all the languages not found
            print("Languages projected from KB: %s" % (",".join(extra_langs)))
        #print(len(X_CCA_dict.keys()), len(X_CCA_dict.values()[0]))
        if len(invalid_langs) > 0:
            print("Languages not found: %s" % (",".join(invalid_langs)))

        return X_CCA_dict



    def get_X_cca_filter(self, label_langs, svd_th=[.9, .9], corpus_extra="all", verbose=False, pre_std_scaler=False, post_std_scaler=False, post_mm_scaler=False):
        # We are going to build the CAT, CCA_filter and CCA+25
        X_uriel_both = self.spaceUriel.get_X_svd_filter(self.label_both, svd_th[0], pre_std_scaler)
        X_learn_both = self.spaceLearn.get_X_svd_filter(self.label_both, svd_th[1], pre_std_scaler)

        X_CCA_both, cca_, corrcoefs_, col_const = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "cca", th_corrcoef=0.5)
        if verbose:
            print("  CCA (shape, corrcoefs):", X_CCA_both.shape[1], corrcoefs_)
        # Analyse if there are more languages in BOTH, otherwise we need to filter them
        if self.no_filter:
            X_CCA_filter = X_CCA_both
        else:
            X_CCA_filter = []
            for l in self.label_filter:
                X_CCA_filter.append(X_CCA_both[self.label_both.index(l)])
        X_CCA_filter = np.array(X_CCA_filter)
        
        # check if we need to scale the uriel features to extract the extra entries
        uriel_scaler = None
        if pre_std_scaler and svd_th[0] != 1.:
            uriel_scaler = self.spaceUriel.scaler

        label_extra = []    
        if corpus_extra in ["wit3", "ted", "gold", "all"]:
            X_CCA_extra, label_extra = self.compute_extra_cca_entries(X_CCA_filter, cca_, col_const, svd_th[0], uriel_scaler, None, corpus=corpus_extra)

        X_cca_all = []
        for l in label_langs:
            if l in self.label_filter:
                X_cca_all.append(X_CCA_filter[self.label_filter.index(l)])
            elif l in label_extra:
                X_cca_all.append(X_CCA_extra[label_extra.index(l)])
            else:
                print(" error ! ", l)
        
        # Compute the standard scaler if required:
        #scaler = None
        if post_std_scaler:
            X_cca_all = StandardScaler().fit_transform(X_cca_all)
        elif post_mm_scaler:
            X_cca_all = MinMaxScaler().fit_transform(X_cca_all)

        return np.array(X_cca_all)

    def compute_CCA_space(self, svd_th=[.75, .75], verbose=False):
        # We are going to build the CAT, CCA_filter and CCA+25
        X_uriel_both = self.spaceUriel.get_X_svd_filter(self.label_both, svd_th[0])
        X_learn_both = self.spaceLearn.get_X_svd_filter(self.label_both, svd_th[1])

        X_CAT_both = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "concat")
        X_CCA_both, cca_, corrcoefs_ = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "cca", th_corrcoef=0.5)
        if verbose:
            print("  CCA (shape, corrcoefs):", X_CCA_both.shape[1], corrcoefs_)
        if self.no_filter:
            X_CAT_filter = X_CAT_both
            X_CCA_filter = X_CCA_both
        else:
            X_CAT_filter = []
            X_CCA_filter = []
            for l in self.label_filter:
                X_CAT_filter.append(X_CAT_both[self.label_both.index(l)])
                X_CCA_filter.append(X_CCA_both[self.label_both.index(l)])
        
        X_CAT_filter = np.array(X_CAT_filter)
        X_CCA_filter = np.array(X_CCA_filter)
        return X_CAT_filter, X_CCA_filter

    def compute_compositional_clusters(self, svd_th=[.75, .75], corpus_extra="", verbose=False, method="average", pre_std_scaler=False, pre_mm_scaler=False, post_std_scaler=False, post_mm_scaler=False):
        # We are going to build the CAT, CCA_filter and CCA+25
        X_uriel_both = self.spaceUriel.get_X_svd_filter(self.label_both, svd_th[0], std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)
        X_learn_both = self.spaceLearn.get_X_svd_filter(self.label_both, svd_th[1], std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)

        # Pre standardscaler --> we normalize URIEL and LEARN with zero-mean
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        
        # we do not need to do this, as we are scaling the features in the single space...
        #if (pre_std_scaler or pre_mm_scaler) and svd_th[0] == 1.:
            ##X_learn_both = MinMaxScaler().fit_transform(X_learn_both)
            #X_CAT_both = compose_feats_matrix(X_uriel_both, MinMaxScaler().fit_transform(X_learn_both), comp = "concat")
        #else:
        X_CAT_both = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "concat")
        
        X_CCA_both, cca_, corrcoefs_, col_const = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "cca", th_corrcoef=0.5)
        if verbose:
            print("  CCA (shape, corrcoefs):", X_CCA_both.shape[1], corrcoefs_)
        if self.no_filter:
            X_CAT_filter = X_CAT_both
            X_CCA_filter = X_CCA_both
        else:
            X_CAT_filter = []
            X_CCA_filter = []
            for l in self.label_filter:
                X_CAT_filter.append(X_CAT_both[self.label_both.index(l)])
                X_CCA_filter.append(X_CCA_both[self.label_both.index(l)])
        
        X_CAT_filter = np.array(X_CAT_filter)
        X_CCA_filter = np.array(X_CCA_filter)
        
        name = self.name + "_%.2f-%.2f" % (svd_th[0], svd_th[1])
        cluster_CAT = ClusterData(X_CAT_filter, self.label_filter, name="CAT_"+name, max_clusters=len(self.label_filter)-1)
        cluster_CCA_extra = None
        
        #if X_CCA_filter.shape[0] < 25:
        if corpus_extra != "":
            if corpus_extra in ["wit3", "ted", "gold", "all"]:
                X_CCA_extra, label_extra = self.compute_extra_cca_entries(X_CCA_filter, cca_, col_const, svd_th[0], pre_std_scaler, pre_mm_scaler, corpus=corpus_extra)
                n_extra = len(label_extra)
                name_scale = ""
                # scale the spaces
                if post_std_scaler:
                    X_CCA_extra = StandardScaler().fit_transform(X_CCA_extra)
                    name_scale = "_ss"
                elif post_mm_scaler:
                    X_CCA_extra = MinMaxScaler().fit_transform(X_CCA_extra)
                    name_scale = "_mms"
                cluster_CCA_extra = ClusterData(X_CCA_extra, label_extra, name= "CCA%d%s_" % (n_extra, name_scale) + name, max_clusters=n_extra-3, method=method)
            elif corpus_extra == "both":
                X_CCA_wit3, label_wit3 = self.compute_extra_cca_entries(X_CCA_filter, cca_, col_const, svd_th[0], pre_std_scaler, pre_mm_scaler, corpus="wit3")
                X_CCA_ted,  label_ted  = self.compute_extra_cca_entries(X_CCA_filter, cca_, col_const, svd_th[0], pre_std_scaler, pre_mm_scaler, corpus="ted" )
                cluster_CCA_extra_wit = ClusterData(X_CCA_wit3, label_wit3, name= "CCA%d_"%len(label_wit3)+ name, max_clusters=len(label_wit3)-3, method=method)
                cluster_CCA_extra_ted = ClusterData(X_CCA_ted , label_ted , name= "CCA%d_"%len(label_ted) + name, max_clusters=len(label_ted)-3, method=method)
                cluster_CCA_extra = (cluster_CCA_extra_wit, cluster_CCA_extra_ted)
        
        name_scale = ""
        if post_std_scaler:
            X_CCA_filter = StandardScaler().fit_transform(X_CCA_filter)
            name_scale = "_ss"
        elif post_mm_scaler:
            X_CCA_filter = MinMaxScaler().fit_transform(X_CCA_filter)
            name_scale = "_mms"
        cluster_CCA = ClusterData(X_CCA_filter, self.label_filter, name="CCA_"+name, max_clusters=len(self.label_filter)-1, method=method)

        return cluster_CAT, cluster_CCA, cluster_CCA_extra

    def compute_extra_cca_entries(self, X_CCA_filter, cca_model, col_const, svd_th, pre_std_scaler, pre_mm_scaler, corpus="wit3", source="uriel"):
        if corpus == "wit3": # WIT3
            total_langs = wit3_25_langs
        elif corpus == "ted": # TED (50?)
            total_langs = ted_iso_langs
        elif corpus == "gold": #Gold Standard
            total_langs = list_baseline_langs
        elif corpus == "all":
            total_langs = all_iso_langs
        else: # corpus has a list of languages: "lang1 lang2 lang3"
            extra_langs = corpus.split()
            total_langs = self.label_filter.copy()
            total_langs.extend(extra_langs)
        num_langs = len(total_langs)

        #First, we get the languages out of the filter list...
        label_extra = sorted(list(set(total_langs) - set(self.label_filter)))
        if len(label_extra) + len(self.label_filter) != num_langs:
            print("Warning: (CCA extra entries) check carefully the filter list...")
        label_25 = self.label_filter.copy()
        label_25.extend(label_extra)
        
        #Second, we extract those languages from the Uriel View with svd_th
        if source == "uriel":
            X_uriel_extra = self.spaceUriel.get_X_svd_filter(label_extra, svd_th, std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)
        elif source == "learn":
            X_uriel_extra = self.spaceLearn.get_X_svd_filter(label_extra, svd_th, std_scaler=pre_std_scaler, minmax_scaler=pre_mm_scaler)
        X_uriel_extra = np.array(X_uriel_extra)
        # if there was a normalization (std_scaler) we need to perform it...
        #if uriel_scaler != None:
        #    X_uriel_extra = uriel_scaler.transform(X_uriel_extra)

        #2.5: We need to delete the constant columns we detect in the CCA composition....
        if len(col_const) > 0:
            X_uriel_extra = np.delete(X_uriel_extra, col_const, axis=1)
        
        #Third, we transform the vectors using the cca_model
        X_CCA_extra = cca_model.transform(X_uriel_extra)#.reshape(-1,1)
        
        #Fourth, we append the vectors in the CCA
        #print(X_CCA_filter.shape, X_CCA_extra.shape)
        if X_CCA_extra.shape[1] > X_CCA_filter.shape[1]:
            X_CCA_extra = X_CCA_extra[:, :X_CCA_filter.shape[1]]
        X_CCA_25 = np.concatenate((X_CCA_filter, X_CCA_extra), axis=0)
        return X_CCA_25, label_25

    def compute_all_unique_clusters(self, svd_begin=0.5, svd_end=1.0, svd_step=0.05, corpus="wit3", verbose=False):
        '''
        This method computes all possible clusters in max_silhouette given a svd_th for the 2 views.
        '''
        svd_th = [svd_begin]
        while svd_begin < svd_end:
            svd_begin += svd_step
            svd_th.append(svd_begin)
        
        total_members_CAT = []
        total_members_CCA = []
        for th1 in svd_th:
            for th2 in svd_th:
                cl_CAT, cl_CCA, _ = self.compute_compositional_clusters(svd_th=[th1, th2], corpus_extra="")
                members_CAT, nc_CAT = cl_CAT.get_cluster_members()
                members_CCA, nc_CCA = cl_CCA.get_cluster_members()
                total_members_CAT.extend(members_CAT)
                total_members_CCA.extend(members_CCA)
                if verbose:
                    print("  %.2f-%.2f: CAT = %02d; CCA = %02d" % (th1, th2, nc_CAT, nc_CCA) )
        if verbose:
            print("  Total CAT = %02d;  Total CCA = %02d" % (len(total_members_CAT), len(total_members_CCA)) )
        total_members_CAT = sorted(list(set(total_members_CAT)))
        total_members_CCA = sorted(list(set(total_members_CCA)))
        # Sorting for the len of the string -> the number of members per cluster
        total_members_CAT.sort(key = lambda s: len(s))
        total_members_CCA.sort(key = lambda s: len(s))
        if verbose:
            print("  Unique CAT= %02d;  Unique CCA= %02d" % (len(total_members_CAT), len(total_members_CCA)) )
            print("  CAT: %s" % (",".join(total_members_CAT)) )
            print("  CCA: %s" % (",".join(total_members_CCA)) )
        return total_members_CAT, total_members_CCA

    
    def analyse_cluster_variability(self, svd_begin=0.5, svd_end=1.0, svd_step=0.05):
        '''
        This method plots different analysis of the svd_th variability.
        (It is expected to analyse the case when new languages are projected too...)
        '''
        svd_th = [svd_begin]
        while svd_begin < svd_end:
            svd_begin += svd_step
            svd_th.append(svd_begin)
        
        cophen_values = []
        for th1 in svd_th:
            for th2 in svd_th:
                cl_CAT, cl_CCA, cl_CCA_extras = self.compute_compositional_clusters(svd_th=[th1, th2], corpus_extra="both")
                cl_CCA_wit, cl_CCA_ted = cl_CCA_extras
                cophen_values.append((th1, th2, cl_CAT.cophenetic_corr(), cl_CCA.cophenetic_corr(), 
                                      cl_CCA_wit.cophenetic_corr(), cl_CCA_ted.cophenetic_corr()))
                print((" %.2f-%.2f : %.4f %.4f  %.4f %.4f") % (cophen_values[-1]))
        #return

    def compute_CCA_incremental(self, svd_th=[.75, .75], verbose=False, iter=30, std_scaler=False):
        # We are going to build the CCA_filter and CCA+ ...
        X_uriel_both = self.spaceUriel.get_X_svd_filter(self.label_both, svd_th[0])
        X_learn_both = self.spaceLearn.get_X_svd_filter(self.label_both, svd_th[1])

        coph_corr = []
        max_sil_3 = []
        ncluster_3 = []
        max_sil_2 = []
        ncluster_2 = []
        name = self.name + "_%.2f-%.2f" % (svd_th[0], svd_th[1])
        
         #..., we compute the CCA shared space
        X_CCA_both, cca_model, corrcoefs_, x_const = compose_feats_matrix(X_uriel_both, X_learn_both, comp = "cca", th_corrcoef=0.5)
        if verbose:
            print("  CCA (shape, corrcoefs): %2d, [%s]" % (X_CCA_both.shape[1], " ".join(["%.4f"%c for c in corrcoefs_])) )
        #..., we check if we need to filter the CCA space
        if self.no_filter:
            X_CCA_filter = X_CCA_both
        else:
            X_CCA_filter = []
            for l in self.label_filter:
                X_CCA_filter.append(X_CCA_both[self.label_both.index(l)])
        X_CCA_filter = np.array(X_CCA_filter)
        
        #First, we get the languages out of the filter list...
        label_total = ted_iso_langs
        label_extra = sorted(list(set(label_total) - set(self.label_filter)))
        num_lang_extra = len(label_extra)

        if num_lang_extra == 0:
            #this is the case when we are analysing the ted53-53l:
            label_extra = sorted(list(set(label_total) - set(wit3_emnlp_ids)))
            num_lang_extra = len(label_extra)
            X_CCA_filter_23 = []
            for l in wit3_emnlp_ids:
                X_CCA_filter_23.append(X_CCA_both[self.label_both.index(l)])
            X_CCA_filter = np.array(X_CCA_filter_23)

        #Second, we extract those languages from the Uriel View with svd_th
        X_uriel_extra = self.spaceUriel.get_X_svd_filter(label_extra, svd_th[0])
        X_uriel_extra = np.array(X_uriel_extra)
            
        #Third, we transform the vectors using the cca_model
        if len(x_const) > 0:
            X_uriel_extra = np.delete(X_uriel_extra, x_const, axis=1)
        X_CCA_extra = cca_model.transform(X_uriel_extra)#.reshape(-1,1)
        
        #Fourth, we retain the columns similar to the original CCA matrix
        if X_CCA_extra.shape[1] > X_CCA_filter.shape[1]:
            X_CCA_extra = X_CCA_extra[:, :X_CCA_filter.shape[1]]

        idx_original = list(np.arange(num_lang_extra))
        for it in range(iter):
            if verbose:
                    print("--iteration %02d/%02d..." % (it+1,iter) )
        
            if it > 0:
                shuffle(idx_original)
                X_CCA_shuffle = []
                label_shuffle = []
                for idx in idx_original:
                    X_CCA_shuffle.append(X_CCA_extra[idx])
                    label_shuffle.append(label_extra[idx])
                X_CCA_shuffle = np.array(X_CCA_shuffle)
            else:
                X_CCA_shuffle = X_CCA_extra
                label_shuffle = label_extra
                
            #Adding CCA data:
            # Performing normalization post-transformation
            if std_scaler:
                cluster_CCA = ClusterData(StandardScaler().fit_transform(X_CCA_filter), self.label_filter, name="CCA_"+name, max_clusters=22)
            else:
                cluster_CCA = ClusterData(X_CCA_filter, self.label_filter, name="CCA_"+name, max_clusters=22)
            coph_corr.append([cluster_CCA.cophenetic_corr()])
            max_sil_3.append([cluster_CCA.max_silhouette(begin=3)])
            ncluster_3.append([cluster_CCA.num_clusters_by_sil(begin=3)])
            max_sil_2.append([cluster_CCA.max_silhouette()])
            ncluster_2.append([cluster_CCA.num_clusters_by_sil()]) 
            
            #Fourth, we append the vectors in the CCA
            for i in range(num_lang_extra):
                X_CCA_inc = np.concatenate((X_CCA_filter, X_CCA_shuffle[:i+1, :]), axis=0)
                if std_scaler:
                    X_CCA_inc = StandardScaler().fit_transform(X_CCA_inc)
                cl_CCA_inc = ClusterData(X_CCA_inc, self.label_filter + label_shuffle[:i+1], name="CCA_"+name, max_clusters=22)
                coph_corr[-1].append(cl_CCA_inc.cophenetic_corr())
                max_sil_3[-1].append(cl_CCA_inc.max_silhouette(begin=3))
                ncluster_3[-1].append(cl_CCA_inc.num_clusters_by_sil(begin=3))
                max_sil_2[-1].append(cl_CCA_inc.max_silhouette())
                ncluster_2[-1].append(cl_CCA_inc.num_clusters_by_sil()) 
        
        #return (coph_corr, max_sil_3, ncluster_3, max_sil_2, ncluster_2)
        # transform to np.array and transpose to shape: (langs, iter)
        coph_corr = np.array(coph_corr).T
        max_sil_3 = np.array(max_sil_3).T
        ncluster_3 = np.array(ncluster_3).T
        max_sil_2 = np.array(max_sil_2).T
        ncluster_2 = np.array(ncluster_2).T
        scores_inc = []
        for i in range(len(label_extra)+1):
            scores_inc.append((mean(coph_corr[i]),stdev(coph_corr[i]), 
                            mean(max_sil_3[i]), stdev(max_sil_3[i]), 
                            median(ncluster_3[i]), median_absolute_deviation(ncluster_3[i]), # idx: 4 and 5
                            mean(max_sil_2[i]), stdev(max_sil_2[i]), 
                            median(ncluster_2[i]), median_absolute_deviation(ncluster_2[i])) )    
        comp_dims = (X_uriel_both.shape[1], X_learn_both.shape[1], X_CCA_both.shape[1])
        return scores_inc, comp_dims


    def typological_feat_prediction_per_lang(self, th1=0.65, th2=0.6, out_of_feats=False, family_iter=False, std_scaler=False, minmax_scaler=False, solver_param="liblinear", verbose=False):
        '''
        Method: compare Learned and SVCCA predictions for typ feat prediction
                with an out-of-language setting
        std_scaler -> if True, we StandardScale the Learned and SVCCA
        minmax_scaler -> if True, we MinMaxScale ....
        '''
        from sklearn.linear_model import LogisticRegression
        total_langs = self.label_both

        score_total_learn = []
        score_total_svcca = []
        
        if not family_iter:
            # we iterate one language out of the set
            iteration_dict = {}
            for lang in total_langs:
                iteration_dict[lang] = [lang]
        else:
            # we iterate using the language family dict
            iteration_dict = self.dict_families

        list_weights = []
        if verbose:
            print(" INIT: # langs (w/T) = %d / %d ;  # groups = %d" % (len(total_langs), len(self.label_both_truth), len(iteration_dict.items())))
        empty_groups_to_predict = 0

        for key, list_langs in iteration_dict.items():
            # we extract the language(s) from the list
            langs_one_out = total_langs.copy()
            for lang in list_langs:
                langs_one_out.remove(lang)
            
            # we drop entries that do not have a truth value:
            for lang in langs_one_out.copy():
                if lang not in self.label_both_truth:
                    langs_one_out.remove(lang)
            # if all the languages to predict do not have a truth value, we skip...
            count = 0
            for lang in list_langs:
                if lang not in self.label_both_truth:
                    count += 1
            if count == len(list_langs):
                empty_groups_to_predict += 1
                continue

            # we update the label_filter to compute the extra entries without issues
            self.label_filter = langs_one_out
            
            # extracting the training and test entries from Learn
            X_learn_train = self.spaceLearn.get_X_svd_filter(langs_one_out, th2)
            X_learn_test  = self.spaceLearn.get_X_svd_filter(list_langs, th2)

            # compute SVCCA by removing the KB entry we don't need:
            X_svcca_train = []
            X_svcca_test = []

            for idx_feat in range(103):
                # extract the URIEL-svd with one-feature-out
                idx_out_of_feats = []
                if out_of_feats:
                    idx_out_of_feats = get_related_features(idx_feat, feat_class="syntax")
                # if out_of_feats == False, X_uriel_train will extract the original matrix
                X_uriel_train = self.spaceUriel.get_X_svd_filter(langs_one_out, th1, out_of_feats=idx_out_of_feats)
                #X_uriel_train = X_uriel_featsout[:len(langs_one_out)]
                #X_uriel_test  = X_uriel_featsout[len(langs_one_out):]
                #print(X_uriel_featsout.shape, X_uriel_train.shape, X_uriel_test.shape)

                # computing the training and test entries for SVCCA (using LEARN as the first view)
                X_svcca_train_per_feat, cca_, corrcoefs_, col_const = compose_feats_matrix(X_learn_train, X_uriel_train, comp = "cca", th_corrcoef=0.5)
                # compute the extra entry(es)
                X_learn_test_per_feat = np.copy(X_learn_test)
                if len(col_const) > 0:
                    X_learn_test_per_feat = np.delete(X_learn_test_per_feat, col_const, axis=1)
                # we transform the vectors using the cca_model
                X_svcca_test_per_feat = cca_.transform(X_learn_test_per_feat)
                if X_svcca_test_per_feat.shape[1] > X_svcca_train_per_feat.shape[1]:
                    X_svcca_test_per_feat = X_svcca_test_per_feat[:, :X_svcca_train_per_feat.shape[1]]
                
                #X_svcca_test, label_extra = self.compute_extra_cca_entries(X_svcca_train, cca_, col_const, th2, False, False, corpus=lang, source="learn")
                # we take the last entry of X_svcca_test, that includes the extra language.
                #X_svcca_test = X_svcca_test[-1].reshape(1, -1)
                #if label_extra[-1] != lang:
                #    print("ERROR: there is not a coincidence for language %s and list [%s]" % (lang, " ".join(label_extra)))
                #    print(X_svcca_test.shape)
                #    exit(1)
                
                # scale SVCCA if required...
                scaler = None
                if std_scaler:
                    scaler = StandardScaler().fit(X_svcca_train_per_feat)
                elif minmax_scaler:
                    scaler = MinMaxScaler().fit(X_svcca_train_per_feat)
                if scaler is not None:
                    X_svcca_train_per_feat = scaler.transform(X_svcca_train_per_feat)
                    X_svcca_test_per_feat = scaler.transform(X_svcca_test_per_feat)
                # adding the matrix to the full list
                X_svcca_train.append(X_svcca_train_per_feat)
                X_svcca_test.append(X_svcca_test_per_feat)

                # if we do not perform the out-of-feat setting, we break (first iteration)
                if not out_of_feats:
                    break

            X_svcca_test = np.array(X_svcca_test)
            X_svcca_train = np.array(X_svcca_train)
            #print(X_svcca_train.shape, X_svcca_test.shape)

            # we train LogRegression models for Learned and SVCCA per "truth" feature
            Y_targets = self.spaceUriel.get_X_svd_filter(langs_one_out, 1.0)
            #Y_targets = self.spaceUriel.get_X_truth(langs_one_out)
            learn_clfs = []
            svcca_clfs = []
            for idx_feat, y_target in enumerate(Y_targets.T):
                if not out_of_feats:
                    idx_feat = 0 # there is only one model
                if (len(list(set(y_target))) >= 2):
                    clf = LogisticRegression(solver=solver_param, random_state=2020).fit(X_learn_train, y_target)
                    learn_clfs.append(clf)
                    clf = LogisticRegression(solver=solver_param, random_state=2020).fit(X_svcca_train[idx_feat], y_target)
                    svcca_clfs.append(clf)
                else: # there are not at least 2 target classes
                    learn_clfs.append(None)
                    svcca_clfs.append(None)

            list_langs_score_learn = 0
            list_langs_score_svcca = 0
            lang_group_is_empty = True
            empty_langs = 0
            #test_targets = self.spaceUriel.get_X_svd_filter([lang], 1.0)
            test_targets = self.spaceUriel.get_X_truth(list_langs)
            #for x_learn_test, x_svcca_test, y_lang_test in zip(X_learn_test, X_svcca_test, test_targets):
            for idx_lang, (x_learn_test, y_lang_test) in enumerate(zip(X_learn_test, test_targets)):
                x_svcca_test_all_feats = X_svcca_test[:, idx_lang, :]
                sum_score_learn = 0
                sum_score_svcca = 0
                n_feats = 0
                for idx_feat, (learn_clf, svcca_clf, y_truth) in enumerate(zip(learn_clfs, svcca_clfs, y_lang_test)):
                    if not out_of_feats:
                        idx_feat = 0 # there is only one model
                    if y_truth == "--":
                        continue
                    else:
                        y_truth = int(float(y_truth))
                    if learn_clf is not None and svcca_clf is not None:
                        # we predict each feat of the out-of-language with Learned
                        value_learn = learn_clf.predict(x_learn_test.reshape(1, -1))
                        sum_score_learn += int(y_truth == value_learn)
                        # we project the out-of-language with Learned and SVCCA, and predict each feat.
                        value_svcca = svcca_clf.predict(x_svcca_test_all_feats[idx_feat].reshape(1, -1))
                        sum_score_svcca += int(y_truth == value_svcca)
                        n_feats += 1
                if n_feats == 0:
                    empty_langs += 1
                    continue
                else:
                    lang_group_is_empty = False
                    list_langs_score_learn += sum_score_learn / n_feats
                    list_langs_score_svcca += sum_score_svcca / n_feats
            
            if not lang_group_is_empty:
                list_langs_score_learn /= len(list_langs) - empty_langs
                list_langs_score_svcca /= len(list_langs) - empty_langs
                list_weights.append(len(list_langs) - empty_langs)
            if verbose:
                if not lang_group_is_empty:
                    print("%20s [%02d]: learn = %.2f ; svcca = %.2f" % (key[:20], len(list_langs), list_langs_score_learn, list_langs_score_svcca))
                else:
                    print("%20s [%02d]: learn = -.-- ; svcca = -.--" % (key[:20], len(list_langs)))
            
            # we calculate the prediction results and accumulate.
            if not lang_group_is_empty:
                score_total_learn.append(list_langs_score_learn)
                score_total_svcca.append(list_langs_score_svcca)
            #score_total_learn += lang_score_learn
            #score_total_svcca += lang_score_svcca
        
        #score_total_learn /= len(total_langs)
        #score_total_svcca /= len(total_langs)
        if verbose:
            print("    Normal AVG.  learn = %.2f +/-%.1f ;  svcca = %.2f +/-%.1f" % (100*np.mean(np.array(score_total_learn)), 100*np.std(np.array(score_total_learn)), 100*np.mean(np.array(score_total_svcca)), 100*np.std(np.array(score_total_svcca))))
            if family_iter:
                w = list_weights
                print("    Weight AVG.  learn = %.2f  ; svcca = %.2f" % (100*np.average(np.array(score_total_learn), weights=w), 100*np.average(np.array(score_total_svcca), weights=w)))
            print(" END: # predicted groups = %d | %d " % (len(list_weights), len(iteration_dict.items())-empty_groups_to_predict ))
        return np.array(score_total_learn), np.array(score_total_svcca), list_weights