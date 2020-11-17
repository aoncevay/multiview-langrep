import numpy as np
import sys
from statistics import mean, median, stdev

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, ward
from scipy.spatial.distance import pdist

from sklearn.metrics import silhouette_score
from src.utils import inertia_score, linkage_matrix, svd_transform, trunc_svd_by_th
from src.utils import save_numc_and_dend_plot, load_features
from src.compose_matrix import compose_feats_matrix

from src.wit3langs import wit3_25_langs, wit3_uriel_ids, wit3_emnlp_ids, wit3_map_inv, wit3_tag2name
from src.tedlangs import ted_iso_langs, ted_iso2tag

from src.ClusterData import ClusterData
from src.LangSpaceMulti import LangSpaceMulti
from src.LangSpaceSingle import LangSpaceSingle

def analyse():
    #objFact53 = LangSpaceSingle(space_name="learn", source_name="ted53-360k")
    objFact53 = LangSpaceSingle(space_name="learn", source_name="ted53marian")
    cls_Fact53 = objFact53.compute_single_cluster(label_filter=ted_iso_langs, svd_th=1.0)
    cls_Fact53.plot_cluster_analysis(corpus="ted", name="ddg+sil-ted53marian")

    objBible = LangSpaceSingle(space_name="learn", source_name="Bible")
    langs_filter = ted_iso_langs.copy()
    lst_remove = ["aze", "mon", "sqi", "bos", "glg", "kat", "kaz", "kur", "mar", "zlm"]
    for l in lst_remove:
        langs_filter.remove(l)
    cls_Bible = objBible.compute_single_cluster(label_filter=langs_filter, svd_th=1.0)
    cls_Bible.plot_cluster_analysis(corpus="ted", name="ddg+sil-Bible43-l2v")
    '''
    cls_Fact53 = objFact53.compute_single_cluster(label_filter=wit3_emnlp_ids, svd_th=1.0)
    cls_Fact53.plot_cluster_analysis(name="_23l",corpus="wit")

    objFact23 = LangSpaceSingle(space_name="learn", source_name="ted23")
    cls_Fact23 = objFact23.compute_single_cluster(label_filter=wit3_emnlp_ids, svd_th=1.0)
    cls_Fact23.plot_cluster_analysis(corpus="wit")
    '''

def compute_incremental_cluster_single(list_ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.], name="_050-100_+05", std_scaler=False, label_filter=wit3_emnlp_ids, method="average", metric="cosine"):
    objSyntax = LangSpaceSingle(space_name="uriel", source_name="syntax_knn")
    objPhonology = LangSpaceSingle(space_name="uriel", source_name="phonology_knn")
    objInventory = LangSpaceSingle(space_name="uriel", source_name="inventory_knn")
    objLTed53 = LangSpaceSingle(space_name="learn", source_name="ted53")
    objLBible = LangSpaceSingle(space_name="learn", source_name="Bible")
    objLTan23 = LangSpaceSingle(space_name="learn", source_name="Tan23")
    results_syntax = []
    results_phonology = []
    results_inventory = []
    results_Lted53 = []
    results_Lbible = []
    results_Ltan23 = []
    for th1 in list_ths:
        print("th = %.2f " % th1)
        scores_syntax, dim_us = objSyntax.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter, metric=metric, method=method) 
        scores_phonology, dim_up = objPhonology.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter)
        scores_inventory, dim_ui = objInventory.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter)
        scores_Lted53, dim_t53 = objLTed53.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter, metric=metric, method=method)
        scores_Lbible, dim_bib = objLBible.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter)
        scores_Ltan23, dim_t23 = objLTan23.compute_incremental(svd_th=th1, std_scaler=std_scaler, label_filter=label_filter, metric=metric, method=method)
        results_syntax.append((th1, scores_syntax, dim_us))
        results_phonology.append((th1, scores_phonology, dim_up))
        results_inventory.append((th1, scores_inventory, dim_ui))
        results_Lted53.append((th1, scores_Lted53, dim_t53))
        results_Lbible.append((th1, scores_Lbible, dim_bib))
        results_Ltan23.append((th1, scores_Ltan23, dim_t23))
    np.save("res_uriel-syntax" + name, np.array(results_syntax))
    np.save("res_uriel-phonol" + name, np.array(results_phonology))
    np.save("res_uriel-invent" + name, np.array(results_inventory))
    np.save("res_learn-ted53" + name, np.array(results_Lted53))
    np.save("res_learn-bible" + name, np.array(results_Lbible))
    np.save("res_learn-tan23" + name, np.array(results_Ltan23))
        

def compute_incremental_cluster_results(list_ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.], name="res_CCA_ted53-23l_050-100_ss+05", label_both=None, learn_source="ted53"):
    #list_ths = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    objUriel = LangSpaceSingle(space_name="uriel", source_name="syntax_knn")
    objTed53 = LangSpaceSingle(space_name="learn", source_name=learn_source)

    #objMulti1 = LangSpaceMulti(name=objTed53.getSourceName()+"_53l", space_1=objUriel, space_2=objTed53) #, label_filter=ted_iso_langs)
    if label_both is not None:
        objMulti2 = LangSpaceMulti(name=objTed53.getSourceName()+"_23l", space_1=objUriel, space_2=objTed53, label_both=label_both)
    else:
        objMulti2 = LangSpaceMulti(name=objTed53.getSourceName()+"_53l", space_1=objUriel, space_2=objTed53)

    results = []

    #tuples_ths = generate_list(list_ths)
    #print(len(tuples_ths), tuples_ths)
    
    #for th1, th2 in tuples_ths:
    for th1 in list_ths:
        for th2 in list_ths:
            print("th1 = %.2f ; th2 = %.2f" % (th1, th2))
            scores_inc, dims = objMulti2.compute_CCA_incremental(svd_th=[th1,th2], std_scaler=True)
            #print(scores_inc, comp_dims)
            #for idx, s in enumerate(scores_inc):
            #    print("  %d => %d +/- %.2f" % (idx+23, s[4], s[5]) )
            results.append((th1, th2, scores_inc, dims))

        #np.save(name, np.array(results))
    np.save(name, np.array(results))


def retrieve_clusters(th1 = 0.8, th2 = 0.6, label_both=None, learn_source="ted53"):
    objUriel = LangSpaceSingle(space_name="uriel", source_name="syntax_knn")
    cls_Uriel_1_0 = objUriel.compute_single_cluster(label_filter=ted_iso_langs, svd_th=1.0, std_scaler=False)
    cls_Uriel_th = objUriel.compute_single_cluster(label_filter=ted_iso_langs, svd_th=th1, std_scaler=True)
    #cls_Uriel_1_0.print_status()
    cls_Uriel_1_0.plot_cluster_analysis(corpus="ted")
    #cls_Uriel_th.print_status()
    cls_Uriel_th.plot_cluster_analysis(corpus="ted")
    print()
    
    langs_filter = ted_iso_langs.copy()
    if learn_source == "Bible":
        lst_remove = ["aze", "mon", "sqi", "bos", "glg", "kat", "kaz", "kur", "mar", "zlm"]
        for l in lst_remove:
            langs_filter.remove(l)
    if learn_source == "Tan23":
        langs_filter = wit3_emnlp_ids
    objTed53 = LangSpaceSingle(space_name="learn", source_name=learn_source)
    if learn_source == "ted53":
        cls_Ted53_1_0 = objTed53.compute_single_cluster(label_filter=langs_filter, svd_th=1.0, std_scaler=False)
        cls_Ted53_th = objTed53.compute_single_cluster(label_filter=langs_filter, svd_th=th2, std_scaler=True)
        #cls_Ted53_1_0.print_status()
        cls_Ted53_1_0.plot_cluster_analysis(corpus="ted")
        #cls_Ted53_th.print_status()
        cls_Ted53_th.plot_cluster_analysis(corpus="ted")
        print()
    
    objMulti = LangSpaceMulti(name=objTed53.getSourceName(), space_1=objUriel, space_2=objTed53)
    if learn_source == "ted53":
        cls_CAT_1_0, c_CCA_1_0, _ = objMulti.compute_compositional_clusters(svd_th=[1.0, 1.0], corpus_extra="", pre_std_scaler=True, post_std_scaler=False)
        cls_CAT_th, c_CCA_th, _ = objMulti.compute_compositional_clusters(svd_th=[th1, th2], corpus_extra="", pre_std_scaler=True, post_std_scaler=True)
        #cls_CAT_1_0.print_status()
        cls_CAT_1_0.plot_cluster_analysis(corpus="ted")
        #cls_CAT_th.print_status()
        cls_CAT_th.plot_cluster_analysis(corpus="ted")
        print()
    
    if label_both is not None:
        objMulti2 = LangSpaceMulti(name=objTed53.getSourceName()+"_23l", space_1=objUriel, space_2=objTed53, label_both=label_both)
        _, _, c_CCA_1_0 = objMulti2.compute_compositional_clusters(svd_th=[1.0, 1.0], corpus_extra="ted", post_std_scaler=True)
        _, c_CCA_th_23l, c_CCA_th = objMulti2.compute_compositional_clusters(svd_th=[th1, th2], corpus_extra="ted", post_std_scaler=True)
    c_CCA_1_0.plot_cluster_analysis(corpus="ted")
    c_CCA_th.plot_cluster_analysis(corpus="ted")
    #c_CCA_1_0.print_status()
    #c_CCA_th.print_status()
    #if label_both is not None:
    #    c_CCA_th_23l.print_status()
    print()
    
    if label_both is None:
        clusters = [cls_Uriel_1_0, cls_Uriel_th, cls_Ted53_1_0, cls_Ted53_th, cls_CAT_1_0, cls_CAT_th, c_CCA_th] # c_CCA_1_0,
    else:
        clusters = [c_CCA_th]
    total_members = []
    for cluster in clusters:
        cls_members, cls_num = cluster.get_cluster_members(allow_two=False)
        for cm in cls_members:
            print(cluster.name + "\t" + " ".join([ted_iso2tag[l] for l in cm.split()]))
        total_members.extend(cls_members)
    unique_members = list(set(total_members))
    
    #print(len(total_members), len(unique_members))
    unique_members.sort(key = lambda s: len(s), reverse=True)
    #for cluster in unique_members:
    #    print(" ".join([ted_iso2tag[l] for l in cluster.split()]))
    

def generate_list(ths=[.5]):
    list_tuples = []
    # base case
    list_tuples.append((ths[0],ths[0]))
    # next case
    for i in range(1,len(ths)):
        # base case
        list_tuples.append((ths[i],ths[i]))
        for j in range(0, i):
            list_tuples.append((ths[i],ths[j]))
            list_tuples.append((ths[j],ths[i]))
    #print(list_tuples)
    return list_tuples
    

if __name__ == "__main__":
    list_ths = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
    label_filter= ["arb", "ces", "deu", "ell", "spa", "pes", "jpn", "tha", "tur", "vie", "cmn"]
    compute_incremental_cluster_single(list_ths = list_ths, name="_050-100_ward+05", std_scaler=False, label_filter=label_filter, method="ward", metric="cosine")
    #print(generate_list([.5, .6, .7, .8]))
    #compute_incremental_cluster_results(list_ths = list_ths, name="res_CCA_ted53-53l_050-100_ss+05", label_both=None)
    #retrieve_clusters(th1=0.65, th2=0.5 , label_both=wit3_emnlp_ids)
    #retrieve_clusters(th1=0.7 , th2=0.55, label_both=wit3_emnlp_ids)
    
    #retrieve_clusters(th1=0.65, th2=0.5)
    #retrieve_clusters(th1=0.65, th2=0.55)
    
    #### CHOSEN ONE: 0.65, 0.6
    #retrieve_clusters(th1=0.65, th2=0.6 , learn_source="ted53")
    #retrieve_clusters(th1=0.65, th2=0.6 , learn_source="Tan23", label_both=wit3_emnlp_ids)
    #analyse()

    #compute_incremental_cluster_single(list_ths = list_ths, name="_050-100_ss+05_11l", std_scaler=True, label_filter=["arb", "ces", "deu", "ell", "spa", "pes", "jpn", "tha", "tur", "vie", "cmn"])
    #compute_incremental_cluster_single(list_ths = list_ths, name="_050-100_+05_11l", std_scaler=False, label_filter=["arb", "ces", "deu", "ell", "spa", "pes", "jpn", "tha", "tur", "vie", "cmn"])