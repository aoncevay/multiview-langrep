# Bridging linguistic typology and multilingual machine translation with multi-view language representations

Sparse language vectors from linguistic typology databases and learned embeddings from tasks like multilingual machine translation have been investigated in isolation, without analysing how they could benefit from each other's language characterisation. We propose to fuse both views using singular vector canonical correlation analysis and study what kind of information is induced from each source. By inferring typological features and language phylogenies, we observe that our representations embed typology and strengthen correlations with language relationships. We then take advantage of our multi-view language vector space for multilingual machine translation, where we achieve competitive overall translation accuracy in tasks that require information about language similarities, such as language clustering and ranking candidates for multilingual transfer. With our method, we can easily project and assess new languages without expensive retraining of massive multilingual or ranking models, which are major disadvantages of related approaches.

## How to project your language-level representations

Once the paper is published, code will be released here, but contact the authors if you want to give it a (preliminary) try.

## Configurate your environment
```
# create a new environment, e.g. with conda, and activate it
conda create -n langrep python=3.6
conda activate langrep
# install the requirements
pip install -r requirements.txt
# install lang2vec independently
pip install --index-url https://test.pypi.org/simple/ --no-deps lang2vec
```

## Functionalities

### Computing multi-view language representations
Executing the default action ("langrep") will compute SVCCA representations and store them in a ```langrep_out.npy``` file that contains a Python dictionary saved with Numpy and with a key-value pair like: ```"eng" : [0.1, 0.2, 0.3, ...., 0.7]```
```
python multiview_langrep.py # OR
python multiview_langrep.py langrep 
```
Default options include:
- KB-source: syntax_knn from [lang2vec](https://pypi.org/project/lang2vec/)
- Learned-source: ted53 (our own factored-embeddings trained from the dataset of [Qi et al. (2018)](https://github.com/neulab/word-embeddings-for-nmt))

Other arguments are:
```
python multiview_langrep.py langrep --KB_source phonology_knn --list_langs "spa,ita,ron,fra" --out_name "my_langreps"
```
which uses Phonology vectors from lang2vec, filters only the listed languages, and saves a file with the specified output name. 

If there are languages that are not included in the KB or learned-source, there will be a message about them. If nothing is introduced, ```list_langs``` will consider all the languages that have representations in the two sources.

There are other parameters for SVCCA like ```--KB_svd_th``` or ```--learn_svd_th```, which are tuned at 0.65 and 0.6, respectively (see the Appendix).

### Using your own language embeddings
In the folder ```data/embeddings/``` you can store a file like ```ted53.npy``` (our own embeddings), which contains a Python dictionary as described previously. Then, use the following command:
```
python multiview_langrep.py langrep --learned_source own --own_source ted53  
# write the name of the file without the extension
```
Future feature: include your own KB-vectors as well.

### Clustering languages
Uses the same parameters as ```langrep```. The function will print a short report, and will plot a PDF with the Elbow method, Silhouette analysis and the Dendrogram of the hierarchy.
```
# Default settings (like the Figure 2 in the paper):
python multiview_langrep.py cluster
# To cluster only Romance languages that are included in ted53:
python multiview_langrep.py cluster --list_langs "por,spa,ita,fra,ron,glg" --out_name "romance_langs"
# We can cluster Romance languages that are not included in ted53 (they are going to be projected from the KB-source):
python multiview_langrep.py cluster --list_langs "por,spa,ita,fra,ron,glg,ast,arg,scn" --out_name "romance_langs_extended"
```

### Ranking related languages
Uses the same parameters as ```langrep```, but with some specific ones:
- ```--target_lang``` : target language for ranking (obligatory)
- ```--list_langs``` : list of candidate languages to consider (optional). By default, it will only analyse the languages included in the two sources.
- ```--num_langs``` : rank an specific number of languages. Default = 5 (int)
- ```--group_size``` : rank languages given an amount of training data to agglomerate (e.g. 500000). Default = -1 (disabled). Currently, it only works with ted53. If enabled, it will disable ```--num_langs```
- ```--filter_size``` : the selection does not consider languages with a training size below this value (e.g. 100000). Default = -1 (disabled). Currently, it only works with ted53.

```
python multiview_langrep.py rank --target_lang glg --num_langs 5 --filter_size 40000
python multiview_langrep.py rank --target_lang glg --group_size 500000
python multiview_langrep.py rank --target_lang glg --num_langs 2 --list_langs "por,spa,ita,fra,ron,glg,ast,arg,scn"
```

## Cite
```
@article{oncevay-etal-2020-bridging,
  title={Bridging linguistic typology and multilingual machine translation with multi-view language representations},
  author={Arturo Oncevay, Barry Haddow and Alexandra Birch},
  journal={arXiv preprint},
  year={2020},
  url={https://arxiv.org/abs/2004.14923}
}
```
