This repository contains the unofficial implementation of the paper
	
> [Von Mises-Fisher Loss for Training Sequence to Sequence Models with Continuous Outputs](https://arxiv.org/pdf/1812.04616.pdf) _Sachin Kumar_ and _Yulia Tsvetkov_
  
  # Requirements

  * Python 3.7
  * Pytorch 1.0
  * torchtext 0.3.1
  * sacremoses 0.0.10
  
  
  # Quick Start
  ## Preprocessing the data
	
  * Tokenization and Truecasing (Using [Moses Scripts](https://github.com/moses-smt/mosesdecoder))
	
  ISWLT data
```
  bash get_data.sh
  bash tokenize.sh
  bash bpeize.sh
```

  WMT data for embeddings
  
```
  bash get_data_wmt.sh
  bash tokenize_wmt.sh
```
  
  ## Training example
```
  python3 train.py --dataset de-en --token_type word --loss vmf --emb-type w2v --tied --reg1 1e-3 --reg2 0.1
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
    
   ## Evaluation example
   
```
    need to specify training parameters
    python3 decode.py --dataset de-en --token-type word --loss vmf --emb-type w2v --batch-size 2048 --tied --reg1 1e-3 --reg2 0.1 --eval-checkpoint all 
```
    

To run all 39 experiments with one command

```
    bash run_all.sh
```
