# Introduction
This is code for ACL 2019 short paper:
```
@inproceedings{jinhua2019soft,
  title={Soft Contextual Data Augmentation for Neural Machine Translation},
  author={Zhu, Jinhua and Gao, Fei and Wu, Lijun and Qin, Tao and Zhou, Wengang and Cheng, Xueqi and Liu, Tie-Yan},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2019}
}
```
# Requirements and Installation
* A [PyTorch installation](http://pytorch.org/)

After PyTorch is installed, you can install fairseq with:
```
pip install -r requirements.txt
python setup.py build develop
```

# Getting Started
Our method is divided into two steps:
1. Train two language models for source language and target languae.
2. Train NMT model with pretrained language models.

### Data prepeocessing

Following standard fairseq data preprocessing, you will get binarized translation dataset.

After that, you can copy dataset for language modeling in order to get **the same vocabulary** as NMT.

**I have to shift a sentence twice in decoder input, so the shortest sentence length after bpe should not be less than 2.**
```
src=en
tgt=ru
for l in $src $tgt; do
    srcdir=${src}2${tgt}
    tgtdir=lmof${l}
    mkdir -p $tgtdir
    cp $srcdir/dict.${l}.txt $tgtdir/dict.txt
    cp $srcdir/train.${src}-${tgt}.${l}.bin $tgtdir/train.bin
    cp $srcdir/train.${src}-${tgt}.${l}.idx $tgtdir/train.idx
    cp $srcdir/valid.${src}-${tgt}.${l}.bin $tgtdir/valid.bin
    cp $srcdir/valid.${src}-${tgt}.${l}.idx $tgtdir/valid.idx
done

```

### Training of language model
I have modified this fairseq repo's dataloader, you'd better train language models with standard [fairseq](https://github.com/pytorch/fairseq) repo.

An example of our script:
```
python train.py $DATA  --task language_modeling --arch $ARCH \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
  --lr 0.0005 --min-lr 1e-09 \
  --dropout 0.1 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096  --tokens-per-sample 4096  --save-dir $SAVE --update-freq 16
```

You can change the language model architecture according to your dataset size.

### Training of NMT
The things I have added are as follows:
```
parser.add_argument('--load-lm', action='store_true',)
parser.add_argument('--load-srclm-file', type=str, default='checkpoint_src.pt')
parser.add_argument('--load-tgtlm-file', type=str, default='checkpoint_tgt.pt')
parser.add_argument('--load-nmt', action='store_true', )
parser.add_argument('--load-nmt-file', type=str, default='checkpoint_nmt.pt')
parser.add_argument('--tradeoff', type=float, default=0.1)
```
1. `--load-lm` is the flag to decide whether to load language model.
2. If you have specify `--load-lm` flag, you have place your two langage model checkpoints in `$SAVE/load-srclm-file(load-tgtlm-file)`
3. `--tradeoff` is the probability to add noise (i usually set it 0.1, 0.15, 0.2).
3. `--load-nmt` is the flag to decide whether to load a NMT model (only a nmt encoder and a decoder without language models) for warmup (I have not verified that this method will work, and i suggest you train your model from scratch).
4. If you have specify `--load-nmt` flag, you should also specify PATH for NMT model (--load-nmt-file).

An example of our script:
```
HOME=/blob/v-jinhzh/code/lm_fairseq
cd $HOME

export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi

src={src}
tgt={tgt}
DATA_PATH=/blob/v-jinhzh/data/bt02/coldnmtdata01/{src}2{tgt}
python -c "import torch; print(torch.__version__)"

TRADEOFF={tradeoff}
SAVE_DIR=checkpoints/dataaug_{src}_{tgt}_{tradeoff}
mkdir -p ${{SAVE_DIR}}
chmod 777 ${{SAVE_DIR}}

cp /blob/v-jinhzh/code/fairseq_baseline/checkpoints/lmof{src}/checkpoint2.pt ${{SAVE_DIR}}/checkpoint_src.pt
cp /blob/v-jinhzh/code/fairseq_baseline/checkpoints/lmof{tgt}/checkpoint2.pt ${{SAVE_DIR}}/checkpoint_tgt.pt



echo SAVE_DIR ${{SAVE_DIR}}
echo TRADEOFF ${{TRADEOFF}}


python train.py ${{DATA_PATH}} --task lm_translation \
--arch transformer_vaswani_wmt_en_de_big --share-decoder-input-output-embed  \
--optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
--lr 0.0009 --min-lr 1e-09 \
--dropout 0.3 --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--max-tokens 2084 --update-freq 48 \
--save-dir ${{SAVE_DIR}} \
--tradeoff ${{TRADEOFF}} --load-lm --save-interval-updates 1000  --seed 200 \
```

I have modified the original `transformer_vaswani_wmt_en_de_big` arch config. You need to
specify language model arch according to your own language model if you have changed the above
language model config.




