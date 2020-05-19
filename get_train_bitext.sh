#!/usr/bin/env bash


TGT_LANG=ja
BASE_DIR=${1:-/Users/dennis/coding/cmsc828b-project}
ORIG_MODEL_DIR=$BASE_DIR/${TGT_LANG}.2
SRC_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.src.model
TGT_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.trg.model

DUO_DATA_DIR=$BASE_DIR/dataverse_files/staple-2020-train
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt

output_dir=$BASE_DIR/for_alex
mkdir $output_dir

echo "Getting Training Data"
python get_traintest_data.py --fname $DUO_EN_JA_FILE --srcfname $output_dir/train.src --tgtfname $output_dir/train.tgt

python segment.py --model $SRC_SPM_MODEL_FILE --input $output_dir/train.src > $output_dir/train.sp.src
python segment.py --model $TGT_SPM_MODEL_FILE --input $output_dir/train.tgt > $output_dir/train.sp.tgt