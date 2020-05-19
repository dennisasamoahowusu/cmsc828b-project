#!/usr/bin/env bash


TGT_LANG=ja
BASE_DIR=${1:-/Users/dennis/coding/cmsc828b-project}
ORIG_MODEL_DIR=$BASE_DIR/${TGT_LANG}.2
SRC_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.src.model
train_src=$BASE_DIR/for_alex/train.src
lengths_file=$BASE_DIR/for_alex/train.src.lengths
python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src --ofile $lengths_file --nos > $train_src.new_segment