#!/usr/bin/env bash

SCRIPT_DIR=$(dirname $(readlink -f $0))

TRAIN_PATH=$SCRIPT_DIR"/main_1.py"
LOG="logs/main_1-fb"

source activate chao_env

/mnt/cephfs2/asr/users/chao.shang/script/queue/queue.pl -q g.q -l gpu=1 $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237 process True &

source  deactivate

# lr 0.003
# dropout 0.2
# nn.Conv1d(2, 100)
# init_emb_size = 200