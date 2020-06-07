# Spectral Clustering with Graph Neural Networks for Graph Pooling
Experimental results obtained with the MinCutPool layer as presented in the 2020 ICML paper [Spectral Clustering with Graph Neural Networks for Graph Pooling](https://arxiv.org/abs/1907.00481)

<img src="./figs/mincutpool.png" width="350" height="150">

This repository is based on [MinCutPool implementation](https://graphneural.network/layers/pooling/#mincutpool) provided by [Spektral](https://graphneural.network/). A Pytorch implementation of MinCutPool is also available in the [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.dense.mincut_pool.dense_mincut_pool) library.

**Required libraries**

- spektral
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

## Clustering

## Autoencoder

## Graph Classification

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
