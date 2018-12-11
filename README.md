# SACN

Paper: "[End-to-end Structure-Aware Convolutional Networks for Knowledge Base Completion](https://arxiv.org/pdf/1811.04441.pdf)" 

Published in the Thirty-Third AAAI Conference on Artificial Intelligence ([AAAI-19](https://aaai.org/Conferences/AAAI-19/)). 

--- PyTorch Version ---

## Overview


## Installation

This repo supports Linux and Python installation via Anaconda. 

1. Install [PyTorch](https://github.com/pytorch/pytorch) using [Anaconda](https://www.continuum.io/downloads). If you compiled PyTorch from source, please checkout the [v0.5 branch](https://github.com/TimDettmers/ConvE/tree/pytorch_v0.5): `git checkout pytorch_0.5`.
2. Download the default English model used by [spaCy](https://github.com/explosion/spaCy), which is installed in the previous step `python -m spacy download en`.
3. If you don't have the "spodernet" and "bashmagic" folder in "src", install the requirements `pip install -r requirements.txt`.

## Data Preprocessing

Run the preprocessing script for WN18RR, FB15k-237, YAGO3-10, UMLS, Kinship, and Nations: `sh preprocess.sh`.

## Run a model

To run a model, you first need to preprocess the data. This can be done by specifying the `process` parameter:
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237 process True
```

Parameters need to be specified by white-space tuples for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py model SACN dataset FB15k-237 \
                                      input_drop 0.2 hidden_drop 0.3 feat_drop 0.2 \
                                      lr 0.003 dataset FB15k-237 process True
```

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

Code is inspired by ConvE.

