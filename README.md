# SACN

Paper: "[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/pdf/1811.04441.pdf)" 

Published in the Thirty-Third AAAI Conference on Artificial Intelligence ([AAAI-19](https://aaai.org/Conferences/AAAI-19/)). 

--- PyTorch Version ---

## Overview
The end-to-end Structure-Aware Convolutional Network (SACN) model takes the benefit of GCN and ConvE together for knowledge base completion. SACN consists of an encoder of a weighted graph convolutional network (WGCN), and a decoder of a convolutional network called Conv-TransE. WGCN utilizes knowledge graph node structure, node attributes and
edge relation types. The decoder Conv-TransE enables the state-of-the-art ConvE to be translational between entities and relations while keeps the same link prediction performance as ConvE. 

## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch 0.4.1 ](https://github.com/pytorch/pytorch) using [official website](https://pytorch.org/) or [Anaconda](https://www.continuum.io/downloads). (For example: conda install pytorch=0.4.1 cuda90 -c pytorch)

2. Install the requirements: `pip install -r requirements.txt`

3. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`.

## Data Preprocessing

Run the preprocessing script for FB15k-237, WN18RR, FB15k-237-attr and kinship: `sh preprocess.sh`.

## Run a model

To run a model, you first need to preprocess the data. This can be done by specifying the `process` parameter:
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237 process True
```

Parameters need to be specified by white-space tuples for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237 batch_size 128 dropout_rate 0.2 channels 100 kernel_size 1 lr 0.001 process True
```

Here a list of parameters for the available datasets:
```
FB15k-237
WN18RR
FB15k-237-attr
kinship
```

The following models/methods are provided in the code:
```
SACN
ConvTransE
ConvE
DistMult
ComplEx
```

After the dataset is preprocessed it will be saved to disk and the parameter "process True" can be omitted or changed to 'process False'.
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237
```

In addition, you can added your own dataset into "data" folder. Your dataset should have the splited files: train.txt, valid.txt, and test.txt. Then you need to add 'mkdir data/DATA_NAME' and 'python wrangle_KG.py DATA_NAME' into the 'preprocess.sh` file and run it.


## Citation

```
@article{shang2018end,
  title={End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion},
  author={Shang, Chao and Tang, Yun and Huang, Jing and Bi, Jinbo and He, Xiaodong and Zhou, Bowen},
  journal={arXiv preprint arXiv:1811.04441},
  year={2018}
}
```

## Acknowledgements

Code is inspired by [ConvE](https://github.com/TimDettmers/ConvE). 

