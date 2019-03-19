#!/bin/bash

bash ./get_data.sh
bash ./tokenize.sh
bash ./get_data_wmt.sh
bash ./tokenize_wmt.sh
bash ./bpeize.sh

bash ./align_data.sh
python3 dicts_from_alignment.py --datasets de-en,en-fr,fr-en

declare -a lang_pairs=("de-en" "en-fr" "fr-en")

for pair in ${lang_pairs[@]}; do
    python3 train.py --dataset ${pair} --token-type word --loss xent
    python3 train.py --dataset ${pair} --token-type bpe --loss xent
    python3 train.py --dataset ${pair} --token-type word_bpe --loss xent
    python3 train.py --dataset ${pair} --token-type word --loss l2  --emb-type w2v --emb-dir . --lr 0.0005
    python3 train.py --dataset ${pair} --token-type word --loss cosine  --emb-type w2v --emb-dir . --lr 0.0005
    python3 train.py --dataset ${pair} --token-type word --loss maxmarg  --emb-type w2v --emb-dir . --lr 0.0005
    python3 train.py --dataset ${pair} --token-type word --loss cosine  --emb-type fasttext --emb-dir . --lr 0.0005
    python3 train.py --dataset ${pair} --token-type word --loss cosine  --emb-type fasttext  --emb-dir . --lr 0.0005 --tied
    python3 train.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --lr 0.0005 --reg1 0.001
    python3 train.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --lr 0.0005 --reg1 0.001 --reg2 0.1
    python3 train.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --lr 0.0005 --tied --reg1 0.001 --reg2 0.1
    python3 train.py --dataset ${pair} --token-type word --loss vmf  --emb-type fasttext --emb-dir . --lr 0.0005 --reg1 0.001 --reg2 0.1
    python3 train.py --dataset ${pair} --token-type word --loss vmf  --emb-type fasttext --emb-dir . --lr 0.0005 --tied --reg1 0.001 --reg2 0.1
    
    
    python3 decode.py --dataset ${pair} --token-type word --loss xent --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type bpe --loss xent --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word_bpe --loss xent --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss l2  --emb-type w2v --emb-dir . --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss cosine  --emb-type w2v --emb-dir . --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss maxmarg  --emb-type w2v --emb-dir . --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss cosine  --emb-type fasttext --emb-dir . --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss cosine  --emb-type fasttext  --emb-dir . --tied --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --reg1 0.001 --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --reg1 0.001 --reg2 0.1 --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss vmf  --emb-type w2v --emb-dir . --tied --reg1 0.001 --reg2 0.1 --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss vmf  --emb-type fasttext --emb-dir . --reg1 0.001 --reg2 0.1 --batch-size 2048 --eval-checkpoint all
    python3 decode.py --dataset ${pair} --token-type word --loss vmf  --emb-type fasttext --emb-dir . --tied --reg1 0.001 --reg2 0.1 --batch-size 2048 --eval-checkpoint all
done
