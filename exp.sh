#!/usr/bin/env bash

SRC_LANG=en
TGT_LANG=ja
BASE_DIR=/Users/dennis/coding/cmsc828b-project
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$SRC_LANG.model
TGT_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$TGT_LANG.model

data_dir=$BASE_DIR/data
mkdir -p $data_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG

train_src_bpe=$train_src.bpe
train_tgt_bpe=$train_tgt.bpe

project_venv=$BASE_DIR/venv

if [ ! -d "${project_venv}" ]; then
  mkdir -p $project_venv
  virtualenv -p python3 $project_venv
  source $project_venv/bin/activate
  pip install -r $BASE_DIR/requirements.txt
  deactivate
fi
source $project_venv/bin/activate

cd $BASE_DIR
if [ ! -f $train_tgt ]; then
    echo "Getting Training Data"
    python get_traintest_data.py --fname $DUO_EN_JA_FILE --srcfname $train_src --tgtfname $train_tgt
    echo "train src file: ${train_src}"
    head -n 5 $train_src
    echo "train tgt file: ${train_tgt}"
    head -n 5 $train_tgt
fi

# TODO Confirm that bpe is happening correctly. Do we need alpha=0.5?
if [ ! -f $train_tgt_bpe ]; then
    echo "Running bpe on src and target"
    python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src > $train_src_bpe
    python segment.py --model $TGT_SPM_MODEL_FILE --input $train_tgt > $train_tgt_bpe
    echo "train src bpe file: ${train_src_bpe}"
    head -n 5 $train_src_bpe
    echo "train tgt bpe file: ${train_tgt_bpe}"
    head -n 5 $train_tgt_bpe
fi





