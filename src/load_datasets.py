import lang2vec.lang2vec as l2v
import numpy as np
from .wit3langs import wit3_tag2name, wit3_map_inv, wit3_map, langs_map, langs_others

PATH_FEATS = "data/cache/"

databases_sets = [  "syntax_wals",
                    "phonology_wals",
                    "syntax_sswl",
                    "syntax_ethnologue",
                    "phonology_ethnologue",
                    "inventory_ethnologue",
                    "inventory_phoible_aa",
                    "inventory_phoible_gm",
                    "inventory_phoible_saphon",
                    "inventory_phoible_spa",
                    "inventory_phoible_ph",
                    "inventory_phoible_ra",
                    "inventory_phoible_upsid"]
average_sets = ["syntax_average",
                "phonology_average",
                "inventory_average"]
knn_predictions = [ "syntax_knn",
                    "phonology_knn",
                    "inventory_knn"]
learned = ["learned"]
comb_sets = ["inventory", "phonology", "syntax"]
features_sets = {"db": databases_sets, "avg": average_sets, "knn": knn_predictions, "l": learned}
all_sets_names = databases_sets + average_sets + knn_predictions + learned + comb_sets


def get_kNN_langs(k=10, geodesic=True, genetic=True, source="uriel", geo_or_gen=False, save_all=True):
    if geodesic and not genetic:
        try:
            dict_geodesic = np.load(PATH_FEATS + str(k) + "_geodesic_" + source + ".npy")
            return dict_geodesic
        except:
            print("File to be processed...")
    if genetic and not geodesic:
        try:
            dict_genetic = np.load(PATH_FEATS + str(k) + "_genetic_" + source + ".npy")
            return dict_genetic
        except:
            print("File to be processed...")
    if genetic and geodesic:
        try:
            dict_geo_gen = np.load(PATH_FEATS + str(k) + "_geo&gen_" + source + ".npy")
            return dict_geo_gen
        except:
            print("File to be processed...")
    
    list_lang = get_list_languages(source = source)
    #test purposes
    #list_lang = list_lang[:10]
    
    geodesic_feats = {}
    genetic_feats = {}

    geodesic_dist = {}
    genetic_dist = {}
    geo_gen_dist = {}

    num_lang = len(list_lang)
    print("Getting features for languages....")
    i = 0
    for l in list_lang:
        if geodesic: 
            geodesic_feats[l] = l2v.get_features(l, "geo").get(l)
            if not genetic: geodesic_dist[l] = {}
        if genetic:
            genetic_feats[l] = l2v.get_features(l,"fam").get(l)
            if not geodesic: genetic_dist[l] = {}
        if geodesic and genetic:
            geo_gen_dist[l] = {}
        i += 1
        if (i % 100 == 0): print("  %04d/%d" % (i,num_lang))

    if save_all:
        if (geodesic and not genetic) or geo_or_gen:
            np.save(PATH_FEATS + "feats_geodesic_" + source + ".npy", geodesic_feats)
        if (genetic and not geodesic) or geo_or_gen:
            np.save(PATH_FEATS + "feats_genetic_" + source + ".npy", genetic_feats)
    
    print("Computing distances....")
    i = 0
    for l1 in list_lang:
        if geodesic: 
            l1_geo = geodesic_feats.get(l1)
        if genetic: 
            l1_gen = genetic_feats.get(l1)
        for l2 in list_lang:
            if l2 != l1:
                if (geodesic and not genetic) or geo_or_gen:
                    l2_in_dist1 = l2 in geodesic_dist.get(l1)
                    l1_in_dist2 = l1 in geodesic_dist.get(l2)
                    if not l2_in_dist1 and not l1_in_dist2:
                        dist = np.linalg.norm( l1_geo - geodesic_feats.get(l2) )
                        geodesic_dist[l1][l2] = dist
                        geodesic_dist[l2][l1] = dist
                    elif not l2_in_dist1:
                        geodesic_dist[l1][l2] = geodesic_dist.get(l2).get(l1)
                    elif not l1_in_dist2:
                        geodesic_dist[l2][l1] = geodesic_dist.get(l1).get(l2)
                
                if (genetic and not geodesic) or geo_or_gen:
                    l2_in_dist1 = l2 in genetic_dist.get(l1)
                    l1_in_dist2 = l1 in genetic_dist.get(l2)
                    if not l2_in_dist1 and not l1_in_dist2:
                        dist = np.linalg.norm( l1_gen - genetic_feats.get(l2) )
                        genetic_dist[l1][l2] = dist
                        genetic_dist[l2][l1] = dist
                    elif not l2_in_dist1:
                        genetic_dist[l1][l2] = genetic_dist.get(l2).get(l1)
                    elif not l1_in_dist2:
                        genetic_dist[l2][l1] = genetic_dist.get(l1).get(l2)

                if geodesic and genetic:
                    l2_in_dist1 = l2 in geo_gen_dist.get(l1)
                    l1_in_dist2 = l1 in geo_gen_dist.get(l2)
                    if not l2_in_dist1 and not l1_in_dist2:
                        dist = np.linalg.norm( np.concatenate((l1_geo, l1_gen)) - 
                                               np.concatenate((geodesic_feats.get(l2), genetic_feats.get(l2))) )
                        geo_gen_dist[l1][l2] = dist
                        geo_gen_dist[l2][l1] = dist
                    elif not l2_in_dist1:
                        geo_gen_dist[l1][l2] = geo_gen_dist.get(l2).get(l1)
                    elif not l1_in_dist2:
                        geo_gen_dist[l2][l1] = geo_gen_dist.get(l1).get(l2)
        i += 1
        if (i % 100 == 0): print("  %04d/%d" % (i,num_lang))
    
    if save_all:
        if (geodesic and not genetic) or geo_or_gen:
            np.save(PATH_FEATS + "dist_geodesic_" + source + ".npy", geodesic_dist)
        if (genetic and not geodesic) or geo_or_gen:
            np.save(PATH_FEATS + "dist_genetic_" + source + ".npy", genetic_dist)
        if genetic and geodesic:
            np.save(PATH_FEATS + "dist_geo&gen_" + source + ".npy", geo_gen_dist)

    geodesic_knn = {}
    genetic_knn = {}
    geo_gen_knn = {}

    print("Ordering by distance....")
    i = 0
    for l1 in list_lang:
        if (geodesic and not genetic) or geo_or_gen:
            d = geodesic_dist[l1]
            sorted_dist = [ (ks, d[ks]) for ks in sorted(d, key=d.__getitem__) ]
            i = 0
            geodesic_knn[l1] = []
            for ks, vs in sorted_dist:
                geodesic_knn[l1].append((ks,vs))
                i += 1
                if (i==k): break
        if (genetic and not geodesic) or geo_or_gen:
            d = genetic_dist[l1]
            sorted_dist = [ (ks, d[ks]) for ks in sorted(d, key=d.__getitem__) ]
            i = 0
            genetic_knn[l1] = []
            for ks, vs in sorted_dist:
                genetic_knn[l1].append((ks,vs))
                i += 1
                if (i==k): break
        if genetic and geodesic:
            d = geo_gen_dist[l1]
            sorted_dist = [ (ks, d[ks]) for ks in sorted(d, key=d.__getitem__) ]
            i = 0
            geo_gen_knn[l1] = []
            for ks, vs in sorted_dist:
                geo_gen_knn[l1].append((ks,vs))
                i += 1
                if (i==k): break
        i += 1
        if (i % 100 == 0): print("  %04d/%d" % (i,num_lang))
    

    if (geodesic and not genetic) or geo_or_gen:        
        np.save(PATH_FEATS + str(k) + "_geodesic_" + source + ".npy", geodesic_knn)
        return np.load(PATH_FEATS + str(k) + "_geodesic_" + source + ".npy")
    if (genetic and not geodesic) or geo_or_gen:
        np.save(PATH_FEATS + str(k) + "_genetic_" + source + ".npy", geodesic_knn)
        return np.load(PATH_FEATS + str(k) + "_genetic_" + source + ".npy")
    np.save(PATH_FEATS + str(k) + "_geo&gen_" + source + ".npy", geo_gen_knn)
    return np.load(PATH_FEATS + str(k) + "_geo&gen_" + source + ".npy")


def get_list_languages(source = "l&u"):
    '''
    param = {"learned": languages from Malaviya et al (2017),
             "uriel":   languages from URIEL,
             "l&u": languages in URIEL and Malaviya et al (2017),
             "l+u": all languages}
    '''
    if source == "uriel":
        try:
            return np.load(PATH_FEATS + "languages_uriel.npy")
        except:
            list_lang = np.array(sorted(list(l2v.available_uriel_languages())))
            np.save(PATH_FEATS + "languages_uriel.npy", list_lang)
            return list_lang
    if source == "learned":
        try:
            return np.load(PATH_FEATS + "languages_learned.npy")
        except:
            list_lang = np.array(sorted(list(l2v.available_learned_languages())))
            #Removing "alb", see issue: https://github.com/antonisa/lang2vec/issues/2
            idx = np.argwhere(list_lang == 'alb')
            list_lang = np.delete(list_lang, idx)
            np.save(PATH_FEATS + "languages_learned.npy", list_lang)
            return list_lang
    if source in ["l+u", "l&u", "l-u", "u-l"]:
        try:
            list_uriel = np.load(PATH_FEATS + "languages_uriel.npy")
        except:
            list_uriel = get_list_languages(source = "uriel")
        try:
            list_learned = np.load(PATH_FEATS + "languages_learned.npy")
        except:
            list_learned = get_list_languages(source = "learned")
        if source == "l+u":
            return np.array(sorted(list(set(list(list_uriel)+list(list_learned)))))
        elif source == "l&u":
            return np.array(sorted([l for l in list_uriel if l in list_learned]))
        elif source == "l-u":
            return np.array(sorted([l for l in list_learned if l not in list_uriel]))
        elif source == "u-l":
            return np.array(sorted([l for l in list_uriel if l not in list_learned]))

    

def load_features_l2v(features="learned", comb_op="+", source="", verbose=False):
    '''
    '''
    if features not in all_sets_names:
        print("Error: invalid feature set")
        return
    
    #Building the expected name
    if features in comb_sets:
        features = comb_op.join([s for s in databases_sets if features in s])
    
    print("Feature:", features)
    if source == "": source = "learned" if features=="learned" else "uriel"
    print("Source:", source)
    
    try:
        return np.load(PATH_FEATS + features + "<"+ source + ".npy")
    except:
        
        list_lang = get_list_languages(source)
        dict_feats = {}
        for i,l in enumerate(list_lang):
            if i % 100 == 0: print(" lang.", i, "of", len(list_lang))
            dict_feats[l] = l2v.get_features(l, features)[l]
        np.save(PATH_FEATS + features + "<"+ source + ".npy", dict_feats)
        return dict_feats


if __name__ == '__main__':
    #print(len(get_list_languages("l-u")))
    #print(len(get_list_languages("u-l")))
    #langs_l_and_u = get_list_languages("l&u")
    langs_learned = get_list_languages("learned")
    langs_uriel   = get_list_languages("uriel")
    #print(len(langs_l_and_u))
    langs_wit3 = wit3_map_inv.keys()
    #print(langs_wit3)
    for l in langs_wit3:
        #print(l, wit3_map_inv[l], end=" ")
        if l not in langs_learned or l not in langs_uriel:
            if l not in langs_map:
                print(l, str(0), "L:", l in langs_learned, "U:", l in langs_uriel)
            else:
                print(l, len(langs_map[l]), "L:", l in langs_learned, "U:", l in langs_uriel)
                for sub_l in langs_map[l]:
                    print("  ", sub_l, "L:", sub_l in langs_learned, "U:", sub_l in langs_uriel)

    wit3_uriel_ids = wit3_map_inv.keys()
    #wit3_learned_ids = wit3_map_inv.keys()
    print(", ".join(["'" + k + "'" for k in wit3_uriel_ids]))


'''
Decisions:
    Replace:
    'ara' -> 'arb' (Standard Arabic) for Learned and URIEL
    'fas' -> 'prs' or 'pes' (Western Persian+) for Learned and URIEL (consider the second)
    'zho' for Learned; 'cmn' for URIEL (Mandarin Chinese)
    Special case:
    'heb' -> It does not have an entry in Learned (Hebrew: religious reasons?)
    *'heb' will be one of the held-out languages*
'''