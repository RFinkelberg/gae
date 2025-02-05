Graph Auto-Encoders
============

This is a TensorFlow implementation of the (Variational) Graph Auto-Encoder model as described in our paper:
 
T. N. Kipf, M. Welling, [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308), NIPS Workshop on Bayesian Deep Learning (2016)

Graph Auto-Encoders (GAEs) are end-to-end trainable neural network models for unsupervised learning, clustering and link prediction on graphs. 

![(Variational) Graph Auto-Encoder](figure.png)

GAEs have successfully been used for:
* Link prediction in large-scale relational data: M. Schlichtkrull & T. N. Kipf et al., [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103) (2017),
* Matrix completion / recommendation with side information: R. Berg et al., [Graph Convolutional Matrix Completion](https://arxiv.org/abs/1706.02263) (2017).


GAEs are based on Graph Convolutional Networks (GCNs), a recent class of models for end-to-end (semi-)supervised learning on graphs:

T. N. Kipf, M. Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907), ICLR (2017). 

A high-level introduction is given in our blog post:

Thomas Kipf, [Graph Convolutional Networks](http://tkipf.github.io/graph-convolutional-networks/) (2016)



## Installation

```bash
python setup.py install
```

## Requirements
* TensorFlow (1.0 or later)
* python 2.7
* networkx
* scikit-learn
* scipy

## Run the demo

```bash
python train.py
```

## Data

In order to use your own data, you have to provide 
* an N by N adjacency matrix (N is the number of nodes), and
* an N by D feature matrix (D is the number of features per node) -- optional

Have a look at the `load_data()` function in `input_data.py` for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/ and here (in a different format): https://github.com/kimiyoung/planetoid

You can specify a dataset as follows:

```bash
python train.py --dataset citeseer
```

(or by editing `train.py`)

## Cyber Data
This fork provides support for a number of cybersecurity network datasets. These datasets can be specified with

```bash
python train.py --data_dir <path> --dataset <dataset_str>
```
Where `--data_dir <path>` specifies the directory containing the datasets, and `--dataset <dataset_str>` specifies which dataset to use. Currently the following are supported:

Dataset | File expected inside `data_dir`
--- | ---
[email][1] | `email-Eu-core-temporal.txt` and `email_Eu-core-department-labels.txt`
[alamos_flows][2] | `flows.txt`
[UNSW][3] | `UNSW-NB15_1.csv`
[gnutella][4] | `p2p-Gnutella08.txt`
[bitcoin][5] | `soc-sign-bitcoinalpha.csv`

[1]: http://snap.stanford.edu/data/email-Eu-core.html
[2]: https://csr.lanl.gov/data/cyber1/
[3]: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/
[4]: http://snap.stanford.edu/data/p2p-Gnutella08.html
[5]: http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html

 These datasets are not packaged with the repository and should be downloaded from the links provided

## Models

You can choose between the following models: 
* `gcn_ae`: Graph Auto-Encoder (with GCN encoder)
* ~~`gcn_vae`: Variational Graph Auto-Encoder (with GCN encoder)~~ (not supported with cyber)

## Cite

Please cite our paper if you use this code in your own work:

```
@article{kipf2016variational,
  title={Variational Graph Auto-Encoders},
  author={Kipf, Thomas N and Welling, Max},
  journal={NIPS Workshop on Bayesian Deep Learning},
  year={2016}
}
```
