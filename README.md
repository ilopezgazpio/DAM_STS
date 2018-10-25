# DAM_STS
Reimplementation of the Decomposable Attention Model (DAM) for STS

If you use this software for academic research please cite the described paper:

```
TBA
```

# Requirements
- Python 2
- Python 3
- pyTorch (tested on 0.2)
- The following python modules: numpy, h5py and nltk.corpus.stopwords (optional)
  

# Usage

1. Download and extract the STS Benchmark dataset from http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz

wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
tar xvzf Stsbenchmark.tar.gz

2. Process the dataset

```
python2 preprocess_datasets/process-STSBenchmark.py \
	--data_folder stsbenchmark \
	--out_folder stsbenchmark
```

3. Download and extract Glove word embeddings in the stsbenchmark folder from http://nlp.stanford.edu/data/glove.840B.300d.zip

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.txt embeddings.glove.txt
```

4. Run Evaluate.sh script

```
./Evaluate.sh script
```

# Acknowledgements
The project is motivated by the following papers and github repositories:

* A. Parikh, O. Täckström, D. Das, J. Uszkoreit, A decomposable attention model for natural language inference
  in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Proceing, Association for Computational Linguistics, Austin, Texas, 2016, pp. 2249-2255.
  URL https://aclweb.org/anthology/D16-1244

* Decomposable Attention Model for Sentence Pair Classification
  https://github.com/harvardnlp/decomp-attn

* SNLI-decomposable-attention
  https://github.com/libowen2121/SNLI-decomposable-attention

* decomposable_attention
  https://github.com/shuuki4/decomposable_attention
