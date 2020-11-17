import numpy as np

from sklearn.metrics import pairwise_distances, silhouette_score

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use('TkAgg')
#plt = matplotlib.pyplot
from matplotlib import pyplot as plt
import seaborn as sns

from .wit3langs import wit3_map
from .load_datasets import get_list_languages, PATH_FEATS
PATH_FIGS = "./figs_obj.v3/"


# computing hierarchical stuf....

def inertia_score(X, a, metric="euclidean"):
    '''
    a: assignments (predictions)
    X: dataset
    '''
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)

def linkage_matrix(X, method="ward", metric="euclidean"):
    y = pdist(X, metric = metric)
    if method == "ward":
        #y = pdist(X, metric = metric)
        Z = ward(y)
    else:
        #Z = linkage(X, method = method, metric = metric)
        Z = linkage(y, method = method, metric = metric)
    return Z, y

# transformation ...
from sklearn.decomposition import TruncatedSVD, PCA

def svd_transform(X, mode="truncated"):
    n = X.shape[1] - 1
    if mode == "truncated":
        #svd = TruncatedSVD(random_state=42, n_components=n, algorithm="arpack")
        svd = TruncatedSVD(random_state=42, n_iter=15, n_components=n)
    elif mode == "full":
        svd = PCA(random_state=42, n_components=n, svd_solver="full")
    X_svd = svd.fit_transform(X)
    '''
    if th > 0:
        p = 0
        sum_variance = 0
        for i in range(n):
            sum_variance += svd.explained_variance_ratio_[i]
            p += 1
            if sum_variance >= th:
                break
        X_svd = X_svd[:, :p]
    '''
    return X_svd, svd

def get_X_svd_filter(X, X_svd, svd_model, svd_th=1.):
    if svd_th == 1.:
        return X
    elif svd_th < 1.:
        dimensions = trunc_svd_by_th(svd_model, svd_th)
        X_filter = X_svd[:, :dimensions]
        return X_filter

def trunc_svd_by_th(svd_model, th):
    p = 0
    sum_variance = 0
    n = svd_model.n_components
    for i in range(n):
        sum_variance += svd_model.explained_variance_ratio_[i]
        p += 1
        if sum_variance >= th:
            break
    return p

# features & plotting ....
wit3_bib24_ids = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ind', 'ita', 'jpn', 
                  'kor', 'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'zho']
wit3_bib22_ids = ['arb', 'bul', 'ces', 'deu', 'ell', 'spa', 'pes', 'fra', 'hun', 'ita', 'jpn', 
                  'nld', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'tha', 'tur', 'vie', 'zho']

def load_families(list_langs):
    import lang2vec.lang2vec as l2v
    # extracting the one-hot-vectors of families
    X_families = l2v.get_features(list_langs, "fam", header=True)
    # family names
    list_fam_names = X_families['CODE']
    # one-hot-codes
    X_fam_data = []
    for lang in list_langs:
        X_fam_data.append(X_families[lang])
    X_fam_data = np.array(X_fam_data)
    # extracting the families with at least one entry
    X_dict_fam = {}
    for idx_fam, fam in enumerate(X_fam_data.T):
        # if it has at least one entry with 1 ...
        if max(fam) == 1:
            fam_name = list_fam_names[idx_fam]
            X_dict_fam[fam_name] = []
            # identify the languages that belongs to that family group
            for idx_lang, lang_value in enumerate(fam):
                if lang_value == 1:
                    X_dict_fam[fam_name].append(list_langs[idx_lang])
    
    # reducing the "duplicated" family groups: with the same members
    family_values = [" ".join(sorted(group)) for group in X_dict_fam.values()]
    # we sort the groups by LEN for further processing
    lang_groups = sorted(list(set(family_values)), key=len)
    # converting back to a list, for easier processing
    lang_groups_lists = [group.split() for group in lang_groups]
    # if the last entry has more than half of the entries, we truncate it: (Indo-European)
    if len(lang_groups_lists[-1]) > len(list_langs) / 2:
        lang_groups_lists = lang_groups_lists[:-1]
    '''
    reverse_idx = -1
    while len(lang_groups_lists[reverse_idx]) > 50:
        reverse_idx -= 1
    reverse_idx += 1
    if reverse_idx < 0:
        lang_groups_lists = lang_groups_lists[:reverse_idx]
    '''
    # obtaining the final list of family groups that are not included in bigger groups
    final_list = []
    for idx, group in enumerate(lang_groups_lists):
        # flag to identify if a group is included in a posterior one (pre-sorting is a must)
        flag_included = False
        for other_group in lang_groups_lists[idx+1:]:
            # if all elements of other_group includes elements from group
            if all(elem in other_group for elem in group):
                flag_included = True
                break
        # if the group is not included, we should keep it
        if not flag_included:
            final_list.append(group)
    # obtaining the final names
    final_dict = {}
    for g in final_list:
        # we asume that the family names are sorted in priority/inclusivity (e.g. Italic before Romance)
        for key, val in X_dict_fam.items():
            # we compare at string level...
            if " ".join(g) == " ".join(sorted(val)):
                final_dict[key] = g
                break
    #for key, val in final_dict.items():
    #    print(key, len(val))

    return final_dict

def load_features(space_name, source_name):

    # source_name : ["Bib22", "Bib24", "Tan23", "Own23", "Own25", "Fact23", "Fact25"]
    label_total = None
    X_total = []
    #print(space_name, source_name)
    if space_name == "uriel":
        feat_set_uriel = source_name + "<" + space_name # syntax_knn<uriel
        X_npdict = np.load(PATH_FEATS + feat_set_uriel + ".npy")
        label_total = list(get_list_languages(source = "uriel"))
        #print(len(label_total))

    elif space_name == "learn": 
        if source_name in ["Bib22", "Bib24", "Bible"]:
            feat_set_learn = "learned<learned"
            X_npdict = np.load(PATH_FEATS + feat_set_learn + ".npy")
            if source_name == "Bible":
                label_total = list(get_list_languages(source = "learned"))
            elif source_name == "Bib22":
                label_total = wit3_bib22_ids
            elif source_name == "Bib24":
                label_total = wit3_bib24_ids
            #label_total[label_total.index("zho")] = "cmn" # unifying Chinese (zho, cmn)
    
        #elif source_name in ["Tan23", "Own23", "Fact23", "Own25", "Fact25", "Fact23-A", "Fact23-B"] or source_name.startswith("ted"):
        else:
            learn_emb_file = "data/embeddings/%s.npy" % source_name.split("-")[0]
            X_npdict = np.load(learn_emb_file)
            label_total = sorted(list(X_npdict.item().keys()))
    
    for l in label_total:
    #    if source_name == "Fact23-A":
    #        X_total.append(X_npdict.item().get(l)[:256])
    #    elif source_name == "Fact23-B":
    #       X_total.append(X_npdict.item().get(l)[256:])
    #    else:
        X_total.append(X_npdict.item().get(l))
    
    # tag 2 iso codes
    #if source_name in ["Own23", "Fact23", "Own25", "Fact25", "Fact23-A", "Fact23-B"]:
    #    label_total = [wit3_map[l] for l in label_total]
    
    if "zho" in label_total:
        label_total[label_total.index("zho")] = "cmn"
    if "hbo" in label_total:
        label_total[label_total.index("hbo")] = "heb"

    # adding extra iso codes in URIEL (they are there!!!!!)
    if space_name == "uriel":
        list_extra_uriel = ["aze", "mon", "sqi"]
        import lang2vec.lang2vec as l2v
        X_extra_dict = l2v.get_features(list_extra_uriel, source_name)
        X_extra = []
        for l in list_extra_uriel:
            label_total.append(l)
            X_extra.append(X_extra_dict[l])
        X_total = np.concatenate((X_total, np.array(X_extra)), axis = 0)
    
    return np.array(X_total), label_total

new_colors = ['#1f77b4', '#17becf', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#ff7f0e']

def save_numc_and_dend_plot(Z, langs_names, maxclusters, X_inertia, X_silhouette, name, path, color_th=0, ang_rot=90, orient="top"):
    if path is None:
        path = PATH_FIGS
    f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15,2), gridspec_kw={'width_ratios': [1, 1, 3]})

    x_items = min(20,maxclusters+1)
    if x_items < len(X_inertia):
        X_inertia = X_inertia[:x_items-2]
        X_silhouette = X_silhouette[:x_items-2]
    #First subplot: Elbow method (Inertia)
    a0.title.set_text("Elbow method")
    a0.grid(True, axis='x', linewidth=0.5)
    a0.plot(range(2, x_items), X_inertia,'-') #, label='data')
    a0.set_xlabel('# clusters')
    #a0.set_ylabel('Inertia score')
    a0.set_xticks(np.arange(2, x_items, step=2))
    a0.tick_params(axis='x', which='minor', labelsize=2)
    
    #Second subplot: Silhouette analysis
    a1.title.set_text("Silhouette analysis")
    #a1.yaxis.tick_right()
    #a1.yaxis.set_label_position("right")
    a1.grid(True, axis='x', linewidth=0.5)
    a1.plot(range(2, x_items), X_silhouette, '-')
    #a1.set_ylabel('Silhouetthe score')
    a1.set_xlabel('# clusters')
    a1.set_xticks(np.arange(2, x_items, step=2))
    
    #Third subplot: Dendrogram
    # analyse color_th:
    if color_th == 0: color_th = np.median(Z[:,2])

    a2.title.set_text("Dendrogram: " + name)
    a2.yaxis.tick_right()
    a2.yaxis.set_label_position("right")
    a2.grid(False)
    hierarchy.set_link_color_palette(new_colors[0:7] + new_colors[8:])
    hierarchy.dendrogram( Z = Z,  
                orientation=orient,
                labels=langs_names,
                distance_sort='descending',
                color_threshold=color_th,
                show_leaf_counts=True,
                leaf_rotation=ang_rot,
                above_threshold_color=new_colors[7]
            )
    f.savefig(path + name + ".pdf", bbox_inches='tight', transparent=True)
    plt.close()

syntax_feat_areas = [0,12,13,14,16,18,20,22,24,26,30,32,35,37,49,52,54,56,58,60,61,65,66,67,68,69,70,71,72,74,76,81,84,87,89,90,92,101,103]
def get_related_features(idx, feat_class="syntax"):
    #looking for the interval...
    if feat_class == "syntax":
        feat_areas = syntax_feat_areas
        for i, boundary in enumerate(feat_areas):
            if idx >= boundary: 
                ini_idx = boundary
                fin_idx = feat_areas[i+1]
            break
        array_to_delete = []
        for i in range(ini_idx, fin_idx):
            array_to_delete.append(i)
        return np.array(array_to_delete)
    else:
        return np.array([idx])