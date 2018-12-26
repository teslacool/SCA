# Introduction

This code is modified from [fairseq](https://github.com/pytorch/fairseq).

# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6

Currently fairseq requires PyTorch version >= 0.4.0.
Please follow the instructions here: https://github.com/pytorch/pytorch#installation.


After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```


# Getting Started
## Tokenize and bpe data
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..
```
## Binarize the dataset
```
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ python preprocess.py --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.tokenized.de-en --joined-dictionary
```
## Train your NMT model
#### First, you have to train a language model for your source language.
##### Iwslt14 example
```
$ python train.py --task language_modeling data-bin/wikitext-103 --arch transformer_lm \
  --optimizer adam --lr 0.0005 --label-smoothing 0.1 --dropout 0.3 --min-lr '1e-09' \
  --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
  --max-update 500000 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9,0.98)' \
  --max-tokens 1024 --tokens-per-sample 1024 --save-dir checkpoints/lm
```
#### Second, train your NMT model and in config file specify the language model checkpoint `--load-lm  --load-lm-file lm_file_name`

The added args is
```
--load-lm default=False
--load-lm-file default='checkpoint_lm.pt'
--tradeoff default=1.
--tradeoff-step default=4000
```
##### Iwslt14 example
```
$ python train.py data-bin/iwslt14.tokenized.de-en --task lm_translation -a transformer_lmnmt \
  --optimizer adam --lr 0.0005 -s de -t en --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy \
  --max-update 50000 --warmup-updates 4000 --warmup-init-lr '1e-07' --adam-betas '(0.9,0.98)' \
  --save-dir checkpoints/transformer --share-all-embeddings --load-lm
```
## Test your model
### Just one checkpint
```
$ python generate.py data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/transformer/model.pt \
  --batch-size 128 --beam 5 --remove-bpe
```
### Test your all ckts
```
python getbleu.py checkpoints/transformer/ 0 data-bin/iwslt14.tokenized.de-en/
```

