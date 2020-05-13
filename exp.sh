#!/usr/bin/env bash

SRC_LANG=en
TGT_LANG=ja
BASE_DIR=/Users/dennis/coding/cmsc828b-project
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt

data_dir=$BASE_DIR/data
mkdir -p $data_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG
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
python get_traintest_data.py --fname $DUO_EN_JA_FILE --srcfname $train_src --tgtfname $train_tgt






