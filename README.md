# Spectral Clustering with Graph Neural Networks for Graph Pooling

<img src="./figs/mincutpool.png" width="400" height="200">

[![arXiv](https://img.shields.io/badge/arXiv-1907.00481-b31b1b.svg?)](https://arxiv.org/abs/1907.00481)
[![ICML](https://img.shields.io/badge/ICML-2020-blue.svg?)](https://proceedings.mlr.press/v119/bianchi20a.html)
[![slides](https://custom-icon-badges.demolab.com/badge/slides-pdf-orange.svg?logo=note&logoSource=feather&logoColor=white)](https://danielegrattarola.github.io/files/talks/2020-ICML-mincut.pdf)
[![blog](https://custom-icon-badges.demolab.com/badge/blog-html-green.svg?logo=message-circle&logoSource=feather&logoColor=white)](https://danielegrattarola.github.io/posts/2019-07-25/mincut-pooling.html)


This repository contains the code to reproduce the results obtained with the MinCutPool layer as presented in the ICML 2020 paper [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481)  by [F. M. Bianchi](https://sites.google.com/view/filippombianchi/home), [D. Grattarola](https://danielegrattarola.github.io/), and C. Alippi

- The official Tensorflow implementation of the MinCutPool layer is in 
[Spektral](https://graphneural.network/layers/pooling/#mincutpool). 
- The PyTorch implementation of MinCutPool is in 
[Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.dense.dense_mincut_pool.html#torch_geometric.nn.dense.dense_mincut_pool).

## Setup

The code is based on Python 3.5, TensorFlow 1.15, and Spektral 0.1.2. 
All required libraries are listed in `requirements.txt` and can be installed with

```bash
pip install -r requirements.txt
``` 

## Image segmentation

<img src="./figs/overseg_and_rag.png" width="700" height="150">

Run [Segmentation.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Segmentation.py) 
to perform hyper-segmentation, generate a Region Adjacency Graph from the 
resulting segments, and then cluster the nodes of the RAG graph with the 
MinCutPool layer.

## Clustering

<img src="./figs/clustering_stats.png" width="600" height="250">

Run [Clustering.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Clustering.py) 
to cluster the nodes of a citation network. The datasets `cora`, `citeseer`, and 
`pubmed` can be selected.
Results are provided in terms of homogeneity score, completeness score, and 
normalized mutual information (v-score).

#### Pytorch
[Clustering_pytorch.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Clustering_pytorch.py) contains a basic implementation in Pytorch based on [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.dense_mincut_pool).

## Autoencoder

<img src="./figs/ae_ring.png" width="400" height="200">
<img src="./figs/ae_grid.png" width="400" height="200">

Run [Autoencoder.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Autoencoder.py) 
to train an autoencoder with a bottleneck and compute the reconstructed graph. It 
is possible to switch between the `ring` and `grid` graphs, but also any other 
[point clouds](https://pygsp.readthedocs.io/en/stable/reference/graphs.html?highlight=bunny#graph-models) 
from the [PyGSP](https://pygsp.readthedocs.io/en/stable/index.html) library 
are supported. Results are provided in terms of the Mean Squared Error.

## Graph Classification

Run [Graph_Classification.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Graph_Classification.py) to train a graph classifier. Additional classification datasets are available [here](https://chrsmrrs.github.io/datasets/) (drop them in ````data/classification/````) and [here](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification) (drop them in ````data/````).
Results are provided in terms of classification accuracy averaged over 10 runs.

#### Pytorch
A basic Pytorch implementation of the graph classification task can be found in this [example](https://github.com/pyg-team/pytorch_geometric/blob/a238110ff5ac772656c967f135fa138add6dabb4/examples/proteins_mincut_pool.py) from Pytorch Geometric.

## Citation

Please, cite the original paper if you are using MinCutPool in your research

```bibtex
@inproceedings{bianchi2020mincutpool,
  title={Spectral Clustering with Graph Neural Networks for Graph Pooling},
  author={Bianchi, Filippo Maria and Grattarola, Daniele and Alippi, Cesare},
  booktitle={Proceedings of the 37th international conference on Machine learning},
  pages={2729-2738},
  year={2020},
  organization={ACM}
}
```
    
## License

The code is released under the MIT License. See the attached LICENSE file.
