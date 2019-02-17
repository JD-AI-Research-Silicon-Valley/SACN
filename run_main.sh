#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $(readlink -f $0))

TRAIN_PATH=$SCRIPT_DIR"/main.py"
LOG="logs/main"

source activate ENV_NAME

/.../script/queue/queue.pl -q g.q -l gpu=1 $LOG bash /.../script/run_gpu.sh /.../anaconda3/envs/ENV_NAME/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237 process True &

source  deactivate