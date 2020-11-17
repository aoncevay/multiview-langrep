import numpy as np
from scipy.cluster.hierarchy import fcluster, cophenet
from sklearn.metrics import silhouette_score
from .utils import inertia_score, linkage_matrix
from .utils import save_numc_and_dend_plot
from .wit3langs import wit3_map_inv, wit3_tag2name
from .tedlangs import ted_iso2name, Baseline_Langs


class ClusterData:
    def __init__(self, X, labels, name, max_clusters=None, method="average", metric="cosine", Z=None): 
        self.X = X # data matrix
        self.Z = Z # linkage matrix
        self.Y = None # pairwise distance matrix
        self.labels = labels # language labels
        self.max_clusters = max_clusters # max. number of clusters
        self.method = method
        self.metric = metric
        try:
            self.name = name + "-d%02d" % X.shape[1]
        except:
            self.name = name
        self.silhouette = [] # silhouette scores
        self.inertia = [] # inertia scores
        self.cluster_members = [] #list of members in each cluster partition
        if Z is None:
            self.compute_LinkageMatrix()
            self.compute_scores_and_clusters()

    def shuffle_entries(self, shuffle_labels, method="", metric=""):
        # check if the content is equal to self.labels....
        X_shuffle = []
        for l in shuffle_labels:
            X_shuffle.append(self.X[self.labels.index(l)])
        
        #we need to restart some variables....
        self.X = np.array(X_shuffle)
        self.labels = shuffle_labels
        self.silhouette = [] # silhouette scores
        self.inertia = [] # inertia scores
        self.cluster_members = [] #list of members in each cluster partition
        if method != "": self.method = method
        if metric != "": self.metric = metric
        self.compute_LinkageMatrix()
        self.compute_scores_and_clusters()

    
    def compute_LinkageMatrix(self):
        self.Z, self.Y = linkage_matrix(self.X, method=self.method, metric=self.metric)

    #def getZ(self): return np.copy(self.Z)
    #def getY(self): return np.copy(self.Y)

    def cophenetic_corr(self, correlation=True):
        if correlation:
            # returns the cophenetic correlation
            return cophenet(Z = self.Z, Y = self.Y)[0]
        else:
            # return the cophenetic condensed matrix
            return cophenet(Z = self.Z)


    def print_status(self, best=1):
        print("ClusterData_name = %s" % self.name)
        print("  #lang = %d; X.shape[1] = %d" % (len(self.labels), self.X.shape[1]))
        print("  sil[2,] = [%s]" % ", ".join(["%.3f" % s for s in self.silhouette]))
        members, nc = self.get_cluster_members()
        print("  best cluster: %d (%.3f) - %s" % (nc, self.silhouette[nc-2], " ; ".join(members)))
        if best > 1:
            nc = self.silhouette.index(sorted(self.silhouette)[-2]) + 2
            members, _ = self.get_cluster_members(num_clusters=nc)
            print("  2ndB cluster: %d (%.3f) - %s" % (nc, self.silhouette[nc-2], " ; ".join(members)))


    def compute_scores_and_clusters(self):
        # We iterate towards the different number of clusters
        for k in range(2, self.max_clusters + 1):
            assignments = fcluster(self.Z, t=k, criterion='maxclust')
            num_labels = len(set(list(assignments)))
            
            #There are special cases when there is no exact cluster division at "k"
            if num_labels <= 1 or num_labels != k:
                #print("  num_labels=", num_labels, "; k=", k)
                self.inertia.append(0)
                self.silhouette.append(0)
                continue
            
            # we compute inertia (to disregard Elbow) and sil
            _inertia = inertia_score(self.X, assignments, metric=self.metric)
            _silhouette = silhouette_score(self.X, assignments, metric=self.metric)
            self.silhouette.append(_silhouette)
            self.inertia.append(_inertia)
            # we extract the cluster members with the names
            self.cluster_members.append(self.extract_cluster_members(k, assignments))

    def extract_cluster_members(self, num_clusters, assignments, start_in_one=True):
        # return a list of the cluster members: ["l1 l2", "l3", "l4 l5 l6"]
        predictions = np.array(assignments)
        cluster_members = []
        for i in range(num_clusters):        
            k_members = np.where(predictions == i+1*start_in_one)[0]
            #print("  ", i+1, k_members, predictions)
            label_k_members = []
            if len(k_members) > 0 :
                label_k_members = np.take(self.labels, k_members)
            cluster_members.append(" ".join(sorted(label_k_members)))
        return cluster_members

    def get_cluster_members(self, num_clusters=None, allow_two=False):
        # Return the cluster members of the specific desired number of clusters
        # if number is none, it looks for the max silohuette
        if num_clusters == None:
            num_clusters = self.silhouette.index(max(self.silhouette)) + 2
            if not allow_two and num_clusters == 2:
                _, num_clusters = self.silhouette_and_nClusters(pos=2)
        # return ["l1 l2", "l3", ...], int
        return self.cluster_members[num_clusters - 2], num_clusters
    
    #Methods for returning silhouette and number of clusters
    def max_silhouette(self, begin=2):
        if begin == 2:
            return max(self.silhouette)
        else:
            return max(self.silhouette[begin-2:])
    def num_clusters_by_sil(self, begin=2):
        if begin == 2:
            return self.silhouette.index(max(self.silhouette)) + 2
        else:
            return self.silhouette.index(self.max_silhouette(begin=begin)) + 2
    def silhouette_and_nClusters(self, pos=1):
        # Return the cluster members from the first, second... silhouette.
        if pos==1:
            return self.max_silhouette(), self.num_clusters_by_sil()
        else: # pos > 2
            copy_sil = self.silhouette.copy()
            while pos > 1:
                copy_sil.remove(max(copy_sil))
                pos -= 1
            max_sil = max(copy_sil)
            nc = self.silhouette.index(max_sil) + 2
            return max_sil, nc

    def plot_cluster_analysis(self, name="", lang_names=True, corpus="wit", path="", elbow=True, title=True):
        if name == "":
            name = self.name + "_" + corpus + name
        name = name.replace("_", "-")
        
        if lang_names:
            lang_labels = []
            for l in self.labels:
                if corpus == "wit23" or corpus== "wit25" or corpus == "wit":
                    lang_labels.append(wit3_tag2name[wit3_map_inv[l]])
                elif corpus == "ted":
                    if l in ted_iso2name.keys():
                        lang_labels.append(ted_iso2name[l])
                    else:
                        lang_labels.append(l)
                elif corpus == "gold":
                    lang_labels.append(Baseline_Langs[l])
        else:  
            lang_labels = self.labels
        self.save_numc_and_dend_plot(lang_labels, name, path, elbow=elbow, title=title)
    
    def save_numc_and_dend_plot(self, langs_names, name, path, elbow=True, title=True, color_th=0, ang_rot=90, orient="top", width=13, height=1.25):
        from scipy.cluster.hierarchy import dendrogram
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        #plt = matplotlib.pyplot
        import seaborn as sns#; sns.set()
        sns.set_color_codes()

        #with sns.axes_style("white"):
        if elbow:
            f, (a0, a1, a2) = plt.subplots(1, 3, figsize=(width,height), gridspec_kw={'width_ratios': [1, 1, 5.5]})
        else:
            f, (a1, a2) = plt.subplots(1, 2, figsize=(height*8.5,height), gridspec_kw={'width_ratios': [1, 5.5], 'wspace': 0.05})

        X_inertia = self.inertia
        X_silhouette = self.silhouette
        x_items = min(20, self.max_clusters + 1)
        if x_items < len(self.inertia):
            X_inertia = self.inertia[:x_items-2]
            X_silhouette = self.silhouette[:x_items-2]
        
        if elbow:
            #First subplot: Elbow method (Inertia)
            if title:
                a0.title.set_text("Elbow method")
            else:
                a0.set_ylabel("inertia")
            a0.grid(True, axis='x', linewidth=0.5)
            a0.plot(range(2, x_items), X_inertia,'-') #, label='data')
            a0.set_xlabel('# clusters\n(Elbow method)')
            a0.set_xticks(np.arange(2, x_items, step=4))
            a0.tick_params(axis='x', which='minor', labelsize=2)
        
        #Second subplot: Silhouette analysis
        if title:
            a1.title.set_text("Silhouette analysis")
        else:
            a1.set_ylabel("silhouette")
        a1.grid(True, axis='x', linewidth=0.5)
        a1.plot(range(2, x_items), X_silhouette, '-')
        #a1.set_ylim(ymin=0, ymax=0.75)
        #a1.set_ylim((0.1, 0.6))
        a1.set_xlabel('# clusters\n(Silhouette analysis)')
        a1.set_xticks(np.arange(2, x_items, step=4))
        
        #Third subplot: Dendrogram
        if color_th == 0: 
            num_clusters = self.num_clusters_by_sil(begin=3)
            color_th = (self.Z[-num_clusters,2] + self.Z[-(num_clusters-1),2])/2

        if title:
            a2.title.set_text("Dendrogram: " + name)
        if elbow:
            a2.yaxis.set_visible(False)
        else:
            a2.yaxis.tick_right()
            a2.yaxis.set_label_position("right")
        a2.grid(False)
        #hierarchy.set_link_color_palette(new_colors[0:7] + new_colors[8:])
        dendrogram( Z = self.Z,  
                    orientation=orient,
                    labels=langs_names,
                    distance_sort='descending',
                    color_threshold=color_th,
                    show_leaf_counts=True,
                    leaf_rotation=ang_rot,
                    leaf_font_size=9
                    #above_threshold_color=new_colors[7]
                )
        f.savefig(path + name + ".pdf", bbox_inches='tight', transparent=True)
        plt.close()
    
    def plot_dendrogram(self, name="", lang_names=True, corpus="gold", path="figs_phytree/", w=5, h=1.5, ang_rot=90, orient="top", color_th=0, distance_sort='descending', count_sort=False, log_scale=False):
        if name == "":
            name = self.name + "_" + corpus + name
        
        #if corpus != "wit":
        #    name = "_" + corpus + "_" + name
        if lang_names:
            lang_labels = []
            for l in self.labels:
                if corpus == "wit23" or corpus== "wit25" or corpus == "wit":
                    lang_labels.append(wit3_tag2name[wit3_map_inv[l]])
                elif corpus == "ted":
                    lang_labels.append(ted_iso2name[l])
                elif corpus == "gold":
                    lang_labels.append(Baseline_Langs[l])
        else:  
            lang_labels = self.labels
        
        from scipy.cluster.hierarchy import dendrogram
        from matplotlib import pyplot as plt
        import seaborn as sns#; sns.set()
        sns.set_color_codes()
        if color_th == 0: 
            num_clusters = self.num_clusters_by_sil(begin=3)
        else:
            num_clusters = color_th
        color_th = (self.Z[-num_clusters,2] + self.Z[-(num_clusters-1),2])/2

        #with sns.axes_style("white"):
        fig, ax = plt.subplots(1,1,figsize=(w, h)) 
        dendrogram( Z = self.Z,  
                    orientation=orient,
                    labels=lang_labels,
                    distance_sort=distance_sort,
                    count_sort=count_sort,
                    color_threshold=color_th,
                    show_leaf_counts=True,
                    #leaf_font_size=12, 
                    leaf_rotation=ang_rot
                )
        if orient == "top":
            #ax.get_yaxis().set_visible(False)
            if log_scale: ax.set_yscale('log')
        elif orient == "left" or orient == "right":
            #ax.get_xaxis().set_visible(False)
            if log_scale: ax.set_xscale('log')
        plt.savefig(path + name + ".pdf", bbox_inches='tight')

