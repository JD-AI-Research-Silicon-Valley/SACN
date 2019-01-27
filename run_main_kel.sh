#!/usr/bin/env bash
. ./path.sh
SCRIPT_DIR=$(dirname $(readlink -f $0))
#queue_cmd="/mnt/cephfs2/asr/users/chao.shang/script/queue/queue.pl -q g.q@GPU_10_252_192_6* -l gpu=1" 
#queue_cmd="/mnt/cephfs2/asr/users/chao.shang/script/queue/queue.pl -q g.q@GPU_10_252_192_8* -l gpu=1" #GPU_10_252_192_80 is broken 
queue_cmd="/mnt/cephfs2/asr/users/chao.shang/script/queue/queue.pl -q g.q@GPU_10_252_192_7* -l gpu=1" 
TRAIN_PATH=$SCRIPT_DIR"/main_kel.py"
source activate chao_env
#for convs in "100.1" "80.1_20.3" "50.1_50.3"
#do
#    key=main-fb.$convs
#    LOG="logs/$key"
#    mkdir -p output/$key
#    if [ ! -e $LOG ]; then
#    $queue_cmd $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237 process False convs $convs save_model_dir output/$key&
#    sleep 30
#    fi
#done    




##for convs in "300.1" "200.1_100.3" "200.1_50.3_50.5" "100.1_100.3_100.5" "300.1_100.3_100.5"
##for convs in "300.3" "300.5" "100.1_200.3" 
#do
#    key=main-fb.$convs
#    LOG="logs/$key"
#    mkdir -p output/$key
#    if [ ! -e $LOG ]; then
#    $queue_cmd $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237 process False convs $convs save_model_dir output/$key  &
#    sleep 30
#    fi
#
#    key=main-wn.$convs
#    LOG="logs/$key"
#    mkdir -p output/$key
#    if [ ! -e $LOG ]; then
#    $queue_cmd $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset WN18RR process False  convs $convs save_model_dir output/$key  &
#    sleep 30
#    fi
#    
#    key=main-fb-attr.$convs
#    LOG="logs/$key"
#    mkdir -p output/$key
#    if [ ! -e $LOG ]; then
#    $queue_cmd  $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237-attr process False convs $convs save_model_dir output/$key   &
#    sleep 30
#    fi
#done





#for convs in "300.1" "200.1_100.3" "200.1_50.3_50.5" "100.1_100.3_100.5" "300.1_100.3_100.5" "300.3" "300.5" "100.1_200.3"
for convs in "100.1_200.3"
do
    opt="dropout 0.3 feature_map_dropout 0.3"
    key=main-fb.dp0.3.$convs
    LOG="logs/$key"
    mkdir -p output/$key
    if [ ! -e $LOG ]; then
    $queue_cmd $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237 process False convs $convs save_model_dir output/$key $opt &
    sleep 30
    fi

    key=main-wn.dp0.3.$convs
    LOG="logs/$key"
    mkdir -p output/$key
    if [ ! -e $LOG ]; then
    $queue_cmd $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset WN18RR process False  convs $convs save_model_dir output/$key $opt &
    sleep 30
    fi
    
    #key=main-fb-attr.dp0.3.$convs
    #LOG="logs/$key"
    #mkdir -p output/$key
    #if [ ! -e $LOG ]; then
    #$queue_cmd  $LOG bash /mnt/cephfs2/asr/users/chao.shang/script/run_gpu.sh /opt/cephfs1/asr/users/chao.shang/anaconda3/envs/chao_env/bin/python -u $TRAIN_PATH model ConvE dataset FB15k-237-attr process False convs $convs save_model_dir output/$key $opt  &
    #sleep 30
    #fi
done
source  deactivate

##
### lr 0.003
### dropout 0.2
### nn.Conv1d(2, 100)
### init_emb_size = 200
