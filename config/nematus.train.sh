#!/usr/bin/env bash

'''
Script for training a small Transformer using Nematus and with factored embeddings
The vocabulary size is 90k, which was training with SentencePiece with all the datasets
'''

cd `dirname $0`
. ../../vars.sh
. ./vars.sh

devices=${1-"0,1,2,3"}

#source vars
#devices=4,5

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $working_dir/corpus.multi.factors.$src \
    --target_dataset $working_dir/corpus.multi.bpe.$tgt \
    --dictionaries $working_dir/corpus.multi.bpe.both.json \
                   $working_dir/corpus.multi.factors.1.src.json \
                   $working_dir/corpus.multi.bpe.both.json \
    --save_freq 30000 \
    --model $working_dir/model.XS \
    --reload latest_checkpoint \
    --model_type transformer \
    --transformer_enc_depth 2 \
    --transformer_dec_depth 2 \
    --transformer_ffn_hidden_size 1024 \
    --factors 2 \
    --embedding_size 512 \
    --dim_per_factor 256 256 \
    --state_size 512 \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule transformer \
    --warmup_steps 4000 \
    --maxlen 100 \
    --batch_size 256 \
    --token_batch_size 16384 \
    --max_tokens_per_device 3276 \
    --valid_source_dataset $working_dir/dev.multi-trunc.factors.$src \
    --valid_target_dataset $working_dir/dev.multi-trunc.bpe.$tgt \
    --valid_batch_size 120 \
    --valid_token_batch_size 3276 \
    --valid_freq 10000 \
    --valid_script $working_dir/scripts/validate.sh \
    --patience 5 \
    --disp_freq 1000 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 0.6 \
    --exponential_smoothing 0.0001 \
    --tie_encoder_decoder_embeddings \
    --tie_decoder_embeddings
