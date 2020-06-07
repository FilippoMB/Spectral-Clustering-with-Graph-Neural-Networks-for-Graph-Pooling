# Spectral Clustering with Graph Neural Networks for Graph Pooling
Experimental results obtained with the MinCutPool layer as presented in the 2020 ICML paper [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481)

<img src="./figs/mincutpool.png" width="400" height="200">

This repository is based on [MinCutPool implementation](https://graphneural.network/layers/pooling/#mincutpool) provided by [Spektral](https://graphneural.network/), the Keras/TensorFlow library for Graph Neural Networks. A Pytorch implementation of MinCutPool is also available in the [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.mincut_pool.dense_mincut_pool) library.

**Required libraries**

- spektral (tested on v.1.2)
- tensorflow (tested on v.1.14)
- keras (tested on v.2.2.4)
- scikit-learn (tested on v.0.22.1)
- scikit-image (tested on v.0.16.2)
- networkx (tested on v.2.4)
- pygsp (tested on v.0.5.1 )
- pandas 
- scipy 
- tqdm
- numpy
- matplotlib 

## Image segmentation

<img src="./figs/overseg_and_rag.png" width="700" height="150">

Run [Segmentation.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Segmentation.py) to perform hyper-segmentation, generate a Region Adjacency Graph from the resulting segments, and then cluster the nodes of the RAG graph by means of a GNN equipped with the MinCutPool layer.

## Clustering

<img src="./figs/clustering_stats.png" width="600" height="250">

Run [Clustering.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Clustering.py) to cluster the nodes of a citation network. The datasets ````cora````, ````citeseer````, and ````pubmed```` can be selected.
Resutls are provided in terms of homogeneity score, completeness score, and normalized mutual information (v-score).

## Autoencoder

<img src="./figs/ae_ring.png" width="400" height="200">
<img src="./figs/ae_grid.png" width="400" height="200">

Run [Autoencoder.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Autoencoder.py) to compute the reconstruction from an Autoencoder with bottleneck. It is possible to switch between the ````ring```` and ````grid````, but also the other [point-clouds datasets](https://pygsp.readthedocs.io/en/stable/reference/graphs.html?highlight=bunny#graph-models) from the [PyGSP](https://pygsp.readthedocs.io/en/stable/index.html) library are supported. Results are provided in terms of Mean Squared Error.

## Graph Classification

Run [Classification.py](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling/blob/master/Classification.py) to train a graph classifier. Additional classification datasets are available [here](https://chrsmrrs.github.io/datasets/) (drop them in ````data/classification/````) and [here](https://github.com/FilippoMB/Benchmark_dataset_for_graph_classification) (drop them in ````data/````).
Results are provided in terms of classification accuray averaged over 10 folds.

## Citation

Please, cite the original paper if you are using MinCutPool in your reasearch

	@inproceedings{bianchi2020mincutpool,
        title={Spectral Clustering with Graph Neural Networks for Graph Pooling},
        author={Bianchi, Filippo Maria and Grattarola, Daniele and Alippi, Cesare},
        booktitle={Proceedings of the 37th international conference on Machine learning},
        pages={},
        year={2020},
        organization={ACM}
    }
    

## License
The code is released under the MIT License. See the attached LICENSE file.
