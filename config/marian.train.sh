#!/usr/bin/env bash

'''
Script for training a small Transformer using Marian with all parameters for model and training
The vocabulary size is 90k, which was training with SentencePiece with all the datasets
'''

cd `dirname $0`
. ../../vars.sh
. ./vars.sh

i=1
WORKSPACE=${1-9500}
devices=${2-"0 1 2 3"}
validate_script=${3-"validate.sh"}
ngpus=`wc -w <<< $devices `

translation=translation
valid_freq=2500
if [ "$ngpus" -eq 1 ] ; then
    valid_freq=1000
fi
optimizer_delay=1
if [ "$ngpus" -eq 4 ] ; then
    optimizer_delay=2
    valid_freq=5000
fi

$marian/build/marian \
    --devices $devices \
    --model $working_dir/model.XS.npz \
    --type transformer \
    --train-sets $working_dir/corpus.multi.bpe.{$src,$tgt} \
    --max-length 100 \
    --max-length-crop \
    --vocabs $working_dir/vocab.multi.{yml,yml} \
    -w $WORKSPACE \
    --mini-batch-fit \
    --maxi-batch 1000 \
    --valid-freq $valid_freq \
    --save-freq $valid_freq \
    --disp-freq 500 \
    --valid-metrics ce-mean-words perplexity $translation \
    --valid-sets $working_dir/dev.trunc.multi.bpe.{$src,$tgt} \
    --valid-script-path $working_dir/scripts/$validate_script \
    --quiet-translation \
    --beam-size 6 --normalize=0.6 \
    --valid-mini-batch 32 \
    --overwrite \
    --keep-best \
    --early-stopping 5 \
    --cost-type=ce-mean-words \
    --log $working_dir/train.XS.log \
    --valid-log $working_dir/valid.XS.log \
    --enc-depth 2 \
    --dec-depth 2 \
    --transformer-dropout 0.1 \
    --label-smoothing 0.1 \
    --learn-rate 0.0001 \
    --lr-warmup 16000 \
    --lr-decay-inv-sqrt 16000 \
    --lr-report \
    --optimizer-params 0.9 0.98 1e-09 \
    --clip-norm 0 \
    --sync-sgd \
    --seed $i$i$i$i \
    --exponential-smoothing \
    --dim-emb 512 \
    --transformer-dim-ffn 1024 --transformer-dim-aan 1024 \
    --transformer-ffn-depth 2 --transformer-aan-depth 2 \
    --transformer-heads 8 \
    --transformer-postprocess-emb d \
    --transformer-postprocess dan \
    --tied-embeddings \
    --optimizer-delay $optimizer_delay \
    --shuffle-in-ram \
    2> error.XS
