import argparse
import numpy as np

def phylogeny_comparison(list_langs=None, source_uriel="syntax_knn", source_learn="ted53", path_learn=None, ths=[0.65, 0.6], out_name="phylogeny_output", metric="cosine", method="ward"):
    from phylogeny_comparison import phylogeny_comparison    
    # only works with external embeddings
    X_dict = multiview_langrep(list_langs=list_langs, source_uriel=source_uriel, source_learn=source_learn, path_learn=path_learn, ths=ths, out_name=None)
    phylogeny_comparison(X_dict, out_name=out_name, metric=metric, method=method)

def clustering_languages(list_langs=None, source_uriel="syntax_knn", source_learn="ted53", path_learn=None, ths=[0.65, 0.6], out_name="clustering_output"):
    #print("Clustering ...")
    from src.ClusterData import ClusterData

    X_cca_dict = multiview_langrep(list_langs=list_langs, source_uriel=source_uriel, source_learn=source_learn, path_learn=path_learn, ths=ths, out_name=None)
    X_cca = []
    for l in X_cca_dict.keys():
        X_cca.append(X_cca_dict[l])
    X_cca = np.array(X_cca)
    cluster = ClusterData(X = X_cca, labels=list(X_cca_dict.keys()), name=out_name, max_clusters=len(X_cca_dict.keys())-1)
    cluster.print_status()
    if out_name is not None:
        cluster.plot_cluster_analysis(name=out_name, corpus="ted", title=False, elbow=True)


def ranking_languages(list_langs=None, source_uriel="syntax_knn", source_learn="ted53", path_learn=None, ths=[0.65, 0.6], tgt_lang="eng", num_langs=5, group_size=-1, filter_size=-1, candidates_extra=""):
    from langrank_comparison import my_langrank_tool
    X_cca_dict = multiview_langrep(list_langs="all", source_uriel=source_uriel, source_learn=source_learn, path_learn=path_learn, ths=ths, out_name=None)
    X_cca = []
    for l in X_cca_dict.keys():
        X_cca.append(X_cca_dict[l])
    X_cca = np.array(X_cca)
    print()
    print("Ranking ...")
    my_langrank_tool(X_cca, list(X_cca_dict.keys()), lang_tgt=tgt_lang, num_langs=num_langs, group_size=group_size, filter_size=filter_size, proportion=False, metric="cosine", source_learn=source_learn)

def multiview_langrep(list_langs=None, source_uriel="syntax_knn", source_learn="ted53", path_learn=None, ths=[0.65, 0.6], out_name="langrep_output"):
    from src.LangSpaceMulti import LangSpaceMulti
    from src.LangSpaceSingle import LangSpaceSingle

    if list_langs != "all":
        list_langs = list_langs.split(",")
    else:
        list_langs = None
    
    if source_uriel != "None":
        objUriel = LangSpaceSingle(space_name="uriel", source_name=source_uriel)
    if source_learn != "None":
        if source_learn == "own":
            if path_learn != "":
                source_learn = path_learn
            else:
                print("You did not enter a value for --own_source, changing to learn=ted53")
                source_learn = "ted53"
        objLearn = LangSpaceSingle(space_name="learn", source_name=source_learn)
    
    if source_uriel == "None":
        return objLearn.get_X_dict(list_langs, svd_th=ths[1])
    elif source_learn == "None": 
        return objUriel.get_X_dict(list_langs, svd_th=ths[0])
    else:
        objMulti = LangSpaceMulti(name=objLearn.getSourceName()+"_53l", space_1=objUriel, space_2=objLearn)
        X_cca_dict = objMulti.get_X_cca_any(list_langs, svd_th=ths, candidates="eng,lav")
        if out_name is not None:
            print("Saving embedding file.... %s.npy" % out_name)
            np.save(out_name + ".npy", X_cca_dict)
        return X_cca_dict



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-view language representations.')
    parser.add_argument('action', type=str, default="langrep", help='langrep = get language embeddings; cluster = cluster a list languages; rank = rank related languages')
    # arguments for embeddings
    parser.add_argument('--KB_source', type=str, default="syntax_knn", help='from lang2vec: syntax_knn, phonology_knn, inventory_knn')
    parser.add_argument('--learn_source', type=str, default="ted53", help='ted53 = factored language embeddings from TED; own = provide your own embeddings')
    parser.add_argument('--own_source', type=str, default="", help='name of the file with language embeddings')
    # arguments for langrep-parameters
    parser.add_argument('--KB_svd_th', type=float, default=0.65, help='SVD threshold for the KB source (default 0.65)')
    parser.add_argument('--learn_svd_th', type=float, default=0.6, help='SVD threshold for the Learn source (default 0.6)')
    # arguments for all actions
    parser.add_argument('--list_langs', type=str, default="all", help='list of languages separated by a comma: spa,ita,ron,fra (default: \"all\" or all languages with entries in the two views')
    parser.add_argument('--out_name', type=str, default="", help='name for the output file (embeddings, clustering plot)')
    # arguments for ranking
    parser.add_argument('--target_lang', type=str, help='target language for ranking. Could be more than one: \"lang1,lang2\"')
    parser.add_argument('--num_langs', type=int, default=5, help='number of languages for ranking')
    parser.add_argument('--group_size', type=int, default=-1, help='rank languages given an amount of training data to agglomerate (disabled by default: -1)')
    parser.add_argument('--filter_size', type=int, default=-1, help='disregard languages with less than the given size (default: -1, no filter at all)')
    parser.add_argument('--candidates_extra', type=str, default="", help='list of extra candidates languages for ranking')
    
    args = parser.parse_args()
    print(args)
    if args.out_name == "":
        args.out_name = args.action + "_output"
    if args.action == "langrep":
        multiview_langrep(args.list_langs, args.KB_source, args.learn_source, args.own_source, ths=[args.KB_svd_th, args.learn_svd_th], out_name=args.out_name)
    elif args.action == "cluster":
        clustering_languages(args.list_langs, args.KB_source, args.learn_source, args.own_source, ths=[args.KB_svd_th, args.learn_svd_th], out_name=args.out_name)
    elif args.action == "rank":
        ranking_languages(args.list_langs, args.KB_source, args.learn_source, args.own_source, ths=[args.KB_svd_th, args.learn_svd_th], tgt_lang=args.target_lang, num_langs=args.num_langs, group_size=args.group_size, filter_size=args.filter_size, candidates_extra=args.candidates_extra)
    elif args.action == "phylogeny":
        phylogeny_comparison(args.list_langs, args.KB_source, args.learn_source, args.own_source, ths=[args.KB_svd_th, args.learn_svd_th], out_name=args.out_name)