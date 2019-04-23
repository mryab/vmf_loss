This repository contains an unofficial implementation of the paper
	
> [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_
  
  # Requirements

  * Python 3.7
  * Pytorch 1.0
  * torchtext 0.3.1
  * sacremoses 0.0.10
  * sacrebleu 1.2.17
  * [fast-align](https://github.com/clab/fast_align) 
  
  
  # Quick Start
  ## Preprocessing the data
  
  IWSLT data
```
  bash scripts/get_data.sh
  bash scripts/tokenize.sh
  bash scripts/bpeize.sh
```

  (For cross-entropy training) Word alignment
  
```
  bash scripts/align_data.sh
  python3 dicts_from_alignment.py --datasets de-en,en-fr,fr-en
```

  WMT data for embeddings
  
```
  bash scripts/get_data_wmt.sh
  bash scripts/tokenize_wmt.sh
```
  
  ## Training
```
  python3 train.py --dataset de-en --token-type word --loss vmf --emb-type w2v --tied --reg1 1e-3 --reg2 0.1
```
  
```
	Options:
	 --dataset {de-en,en-fr,fr-en}
	 --token-type {word,bpe,word_bpe}
	 --loss {xent,l2,cosine,maxmarg,vmfapprox_paper,vmfapprox_fixed,vmf}
	 --batch-size BATCH_SIZE
	 --num-epoch NUM_EPOCH
	 --lr LR
	 --emb-type {w2v,fasttext}
	 --emb-dir EMB_DIR
	 --device-id DEVICE_ID
	 --reg_1 REG_1
	 --reg_2 REG_2
	 --tied
```
    
   ## Evaluation
   
```
    python3 decode.py --dataset de-en --token-type word --loss vmf --emb-type w2v --batch-size 2048 --tied --reg1 1e-3 --reg2 0.1 --eval-checkpoint all 
```
    

To run all 39 experiments with one command

```
    bash run_all.sh
```
