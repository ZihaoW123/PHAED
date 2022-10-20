#!/bin/bash
# Author: GMFTBY
# Time: 2020.2.8

mode=$1     # graph/stat/train/translate/eval/curve
dataset=$2
model=$3
CUDA=$4

# try catch
if [ ! $model ]; then
    model='none'
    CUDA=0
fi

if [ $dataset = 'PersonaChat' ]; then
    knowlege_length=5
else
    knowlege_length=0
fi

# maxlen and batch_size 
maxlen=70
tgtmaxlen=70
batch_size=32

# ========== Ready Perfectly ========== #
echo "========== $mode begin ==========" 

if [ $mode = 'vocab' ]; then
    # Generate the src and tgt vocabulary
    echo "[!] Begin to generate the vocab"
    
    if [ ! -d "./processed/$dataset" ]; then
        mkdir -p ./processed/$dataset
        echo "[!] cannot find the folder, create ./processed/$dataset"
    else
        echo "[!] ./processed/$dataset: already exists"
    fi
        
    # generate the whole vocab for PHAED
    python utils.py \
        --mode vocab \
        --cutoff 50000 \
        --vocab ./processed/$dataset/vocab.pkl \
        --file ./data/$dataset/tgt-train.txt ./data/$dataset/src-train.txt
        
elif [ $mode = 'train' ]; then
    # cp -r ./ckpt/$dataset/$model ./bak/ckpt    # too big, stop back up it
    rm -rf ./ckpt/$dataset/$model
    mkdir -p ./ckpt/$dataset/$model
    
    # create the training folder
    if [ ! -d "./processed/$dataset/$model" ]; then
        mkdir -p ./processed/$dataset/$model
    else
        echo "[!] ./processed/$dataset/$model: already exists"
    fi
    
    # delete traninglog.txt
    if [ ! -f "./processed/$dataset/$model/trainlog.txt" ]; then
        echo "[!] ./processed/$dataset/$model/trainlog.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/trainlog.txt
    fi
    
    # delete metadata.txt
    if [ ! -f "./processed/$dataset/$model/metadata.txt" ]; then
        echo "[!] ./processed/$dataset/$model/metadata.txt doesn't exist"
    else
        rm ./processed/$dataset/$model/metadata.txt
    fi
    
    cp -r tblogs/$dataset/ ./bak/tblogs
    rm tblogs/$dataset/$model/*
    
    
    src_vocab="./processed/$dataset/vocab.pkl"
    tgt_vocab="./processed/$dataset/vocab.pkl"
        
    
    dropout=0.1
    lr=2e-4
    lr_mini=1e-6 
    
    echo "[!] back up finished"
    
    # Train
    echo "[!] Begin to train the model"
    
    # dim_feedforward = 1024 or 2048
    CUDA_VISIBLE_DEVICES="$CUDA" python -u train.py \
        --src_train ./data/$dataset/src-train.txt \
        --tgt_train ./data/$dataset/tgt-train.txt \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --src_dev ./data/$dataset/src-dev.txt \
        --tgt_dev ./data/$dataset/tgt-dev.txt \
        --src_vocab $src_vocab \
        --tgt_vocab $tgt_vocab \
        --pred ./processed/${dataset}/${model}/pure-pred.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --seed 30 \
        --epochs 1000 \
        --lr $lr \
        --batch_size $batch_size \
        --model $model \
        --teach_force 1 \
        --patience 1000 \
        --dataset $dataset \
        --grad_clip 10.0 \
        --dropout $dropout \
        --embed_size 512 \
        --d_model 512 \
        --n_head 8 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --num_turn_embeddings 30 \
        --dim_feedforward 1024\
        --maxlen $maxlen \
        --tgt_maxlen $tgtmaxlen \
        --position_embed_size 102 \
        --knowlege_length $knowlege_length \
        --no-debug \
        --lr_mini $lr_mini \
        --lr_gamma 0.5 \
    
elif [ $mode = 'translate' ]; then
    rm ./processed/$dataset/$model/pertub-ppl.txt
    rm ./processed/$dataset/$model/pred.txt
    
    dropout=0.1
    lr=1e-6
    lr_mini=1e-8 
    
    src_vocab="./processed/$dataset/vocab.pkl"
    tgt_vocab="./processed/$dataset/vocab.pkl"
    
    batch_size=1
    CUDA_VISIBLE_DEVICES="$CUDA" python translate.py \
        --src_train ./data/$dataset/src-train.txt \
        --tgt_train ./data/$dataset/tgt-train.txt \
        --src_test ./data/$dataset/src-test.txt \
        --tgt_test ./data/$dataset/tgt-test.txt \
        --src_dev ./data/$dataset/src-dev.txt \
        --tgt_dev ./data/$dataset/tgt-dev.txt \
        --src_vocab $src_vocab \
        --tgt_vocab $tgt_vocab \
        --pred ./processed/${dataset}/${model}/pure-pred.txt \
        --min_threshold 0 \
        --max_threshold 100 \
        --seed 30 \
        --epochs 1000 \
        --lr $lr \
        --batch_size $batch_size \
        --model $model \
        --teach_force 1 \
        --patience 1000 \
        --dataset $dataset \
        --grad_clip 10.0 \
        --dropout $dropout \
        --embed_size 512 \
        --d_model 512 \
        --n_head 8 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --num_turn_embeddings 30 \
        --dim_feedforward 1024\
        --maxlen $maxlen \
        --tgt_maxlen $tgtmaxlen \
        --position_embed_size 102 \
        --knowlege_length $knowlege_length \
        --no-debug \
        --lr_mini $lr_mini \
        --lr_gamma 0.5 \

elif [ $mode = 'eval' ]; then
    # before this mode, make sure you run the translate mode to generate the pred.txt file for evaluating.
    CUDA_VISIBLE_DEVICES="$CUDA" python eval.py \
        --model $model \
        --file ./processed/${dataset}/${model}/pure-pred.txt

else
    echo "Wrong mode for running"
fi

echo "========== $mode done =========="
