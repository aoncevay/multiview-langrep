import sys
import numpy as np 
import pandas as pd

from src.LangSpaceSingle import LangSpaceSingle
from src.LangSpaceMulti import LangSpaceMulti
from src.tedlangs import all_iso2name, all_iso_langs, ted_iso_langs, ted_iso2tag, ted_tag2size
from src.wit3langs import wit3_emnlp_ids
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform


def get_distance_wrt_lang(pairwise_matrix, label_langs, lang, rank=5, verbose=False):
    dist2lang = [] 
    idx_lang = label_langs.index(lang)
    for i_l, dist_l in enumerate(pairwise_matrix[idx_lang]):
        if i_l == idx_lang: continue
        l_iso = label_langs[i_l]
        dist2lang.append((dist_l, l_iso, all_iso2name[l_iso])) #, iso2code[l_iso]))
    dist2lang = sorted(dist2lang, key=lambda x: x[0], reverse=True)
    
    if rank is None:
        rank = len(label_langs)
    rank_list = []
    for item in dist2lang:
        if verbose:
            print("%s(%4.1f) ; " % (item[1], item[0]) , end="")
        rank_list.append(item[1])
        rank -= 1
        if rank == 0: break
    if verbose: print()
    return rank_list

def get_langs_by_size(lang, rank_list, size, proportion=True):
    rank_list = [ted_iso2tag[l] for l in rank_list]
    cls_topk = []
    total_size = ted_tag2size[ted_iso2tag[lang]]/1000
    i = 0
    while total_size < size:
        if not proportion and total_size + ted_tag2size[rank_list[i]]/1000 > size:
            break
        total_size += ted_tag2size[rank_list[i]]/1000
        cls_topk.append(rank_list[i])
        i += 1
    if proportion:
        extra_size = total_size - size
        last_size = ted_tag2size[cls_topk[-1]]/1000
        ratio = (last_size - extra_size)/last_size
        cls_topk[-1] += "*%.2f" % (ratio)
    return cls_topk, total_size

def my_langrank(lang_tgt, size_tgt, learn="ted53", ths=[1.,1.], num_langs=5, group_size=-1, filter_size=-1, rank=None, proportion=True, metric="cosine", std_scaler=False, corpus_extra=None):
    '''
    lang_tgt can be a single language or a list of languages separated by whitespace
    size_tgt are integers separated by whitespace
    learn = {ted53, Tan23}
    '''
    langs = lang_tgt.split()

    #if group_size != -1:
    #    sizes = [group_size] * len(langs)
    if size_tgt is not None:
        sizes = [int(i) for i in size_tgt.split()]
    
    objUriel = LangSpaceSingle(space_name="uriel", source_name="syntax_knn")
    objLearn = LangSpaceSingle(space_name="learn", source_name=learn)
    objMulti = LangSpaceMulti(name="syntax+"+learn, space_1 = objUriel, space_2 = objLearn)
    
    #corpus_extra = "ted" if learn != "ted53" else None
    X_cca = objMulti.get_X_cca_filter(label_langs=ted_iso_langs, svd_th=[th1,th2], corpus_extra=corpus_extra, post_std_scaler=std_scaler)
    
    #cosine_noS = 1. - squareform(pdist(X_cca_noS, metric="cosine"))
    #correlation_noS = 1. - squareform(pdist(X_cca_noS, metric="correlation"))
    pairwise_dist_matrix = 1. - squareform(pdist(X_cca, metric=metric))
    
    for i, lang in enumerate(langs):
        rank_list = get_distance_wrt_lang(pairwise_dist_matrix, ted_iso_langs, lang, rank)
        rank_list = [ted_iso2tag[l] for l in rank_list]
        cls_top5 = rank_list.copy()
        cls_top5.append(ted_iso2tag[lang])
        cls_top3 = rank_list[:3].copy()
        cls_top3.append(ted_iso2tag[lang])
        #print("%s,%s,%s,%s" % ("SVCCA+"+learn, "corr-top5", lang, " ".join(sorted(cls_top5)) ))
        #print("%s,%s,%s,%s" % ("SVCCA+"+learn, "top3", lang, " ".join(sorted(cls_top3)) ))

        if size_tgt is not None:
            cls_topk, cls_size = get_langs_by_size(lang, rank_list, sizes[i], proportion=proportion)
            cls_topk.append(ted_iso2tag[lang])
            print("%s,%s,%s,%s,%d" % ("SVCCA+"+learn, "topk", lang, " ".join(sorted(cls_topk)), int(cls_size) ))


def my_langrank_tool(X_cca, label_langs, lang_tgt, num_langs=5, group_size=-1, filter_size=-1, proportion=True, metric="cosine", source_learn="ted53"):
    '''
    lang_tgt can be a single language or a list of languages separated by whitespace
    size_tgt are integers separated by whitespace
    learn = {ted53, Tan23}
    '''
    langs = lang_tgt.split(",")
    if group_size != -1:
        sizes = [group_size] * len(langs)
    
    pairwise_dist_matrix = 1. - squareform(pdist(X_cca, metric=metric))
    
    for i, lang in enumerate(langs):
        if group_size == -1:
            rank_list = get_distance_wrt_lang(pairwise_dist_matrix, label_langs, lang, rank=num_langs)
            cls_topK = rank_list.copy()
            #cls_topK.append(lang)
            print("SVCCA+%s; top-%s; tgt_lang= %s ; rank= %s" % (source_learn, str(num_langs), lang, " ".join(cls_topK) ))
        else:
            rank_list = get_distance_wrt_lang(pairwise_dist_matrix, label_langs, lang, rank=None)
            cls_topk, cls_size = get_langs_by_size(lang, rank_list, sizes[i], proportion=proportion)
            cls_topk.append(lang)
            print("SVCCA+%s; size= %d; tgt_lang= %s ; rank= %s" % (source_learn, int(cls_size), lang, " ".join(cls_topk) ))


# python3 my_langrank.py "bos glg zlm est kat eus" "434 443 463 533 499 491"
# python3 my_langrank.py "kaz aze ben urd kur mon" "477 506 478 479 510 499"
# python3 my_langrank.py "bel tam mar" "451 503 508"

if __name__ == "__main__":
    argv = sys.argv
    lang = argv[1]
    sizes = argv[2]
    try:
        th1 = float(argv[3])
        th2 = float(argv[4])
    except:
        #sizes = None
        th1 = 0.65
        th2 = 0.6
    print("---TED53---")
    my_langrank(lang_tgt=lang, size_tgt=sizes ,learn="ted53", ths=[th1,th2], proportion=True, metric="cosine", std_scaler=True)
    print("---WIT23---")
    my_langrank(lang_tgt=lang, size_tgt=sizes ,learn="Tan23", ths=[th1,th2], proportion=True, metric="cosine", std_scaler=True)

    #print(" ".join(["[\'%s\']=\'%s\'" % (ted_iso2tag[l], l) for l in lang.split()]))