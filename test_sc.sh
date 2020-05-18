#!/usr/bin/env bash

CLUSTER_DIR=/Users/dennis/coding/cmsc828b-project/laser-embeds
script=/Users/dennis/coding/cmsc828b-project/get_sc_bitexts.py

### Transforming bitext to include sentence codes
ja_train_with_sc_map=$CLUSTER_DIR/ja/train.k-med.subtract-mean.map.json
ja_train_prompts=$CLUSTER_DIR/ja/train.prompts
ja_train_trans=$CLUSTER_DIR/ja/train.translations

OUTPUT_DIR=/Users/dennis/coding/cmsc828b-project/data

python $script --mfile $ja_train_with_sc_map --pfile $ja_train_prompts --tfile $ja_train_trans \
    --sofile $OUTPUT_DIR/source --tofile $OUTPUT_DIR/target