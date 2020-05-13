#!/usr/bin/env bash

### Variables
SRC_LANG=en
TGT_LANG=ja
BASE_DIR=/Users/dennis/coding/cmsc828b-project
DUO_DATA_DIR=$BASE_DIR/dataverse_files/staple-2020-train
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$SRC_LANG.model
TGT_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$TGT_LANG.model

data_dir=$BASE_DIR/data
mkdir -p $data_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG

train_src_bpe=$train_src.bpe
train_tgt_bpe=$train_tgt.bpe

### Creating and activating virtual environment
project_venv=$BASE_DIR/venv
if [ ! -d "${project_venv}" ]; then
  mkdir -p $project_venv
  virtualenv -p python3 $project_venv
  source $project_venv/bin/activate
  #TODO replace pytorch with appropriate version
  pip install torch torchvision
  pip install -r $BASE_DIR/requirements.txt
  deactivate
fi
source $project_venv/bin/activate

### Transforming duolingo data to bitext
### Creating train and test splits from duolingo data
cd $BASE_DIR
if [ ! -f $train_tgt ]; then
#    echo "Getting Training Data"
#    python get_traintest_data.py --fname $DUO_EN_JA_FILE --srcfname $train_src --tgtfname $train_tgt
#    echo "train src file: ${train_src}"
#    head -n 5 $train_src
#    echo "train tgt file: ${train_tgt}"
#    head -n 5 $train_tgt

    echo "Getting data splits ...."
    python new_create_splits.py --langs "ja" --output_dir $data_dir --duo_data_dir $DUO_DATA_DIR
    echo "Done getting data splits"

    echo "Converting train split to bitext ...."
    train_split=$data_dir/en_ja_split.train
    python get_traintest_data.py --fname $train_split --srcfname $train_src --tgtfname $train_tgt
    echo "train src file: ${train_src}"
    head -n 2 $train_src
    echo "train tgt file: ${train_tgt}"
    head -n 2 $train_tgt

    for var in 0 1 2
    do
        echo "Converting test${var} split to bitext ...."
        test_split=$data_dir/en_ja_split.test${var}
        python get_traintest_data.py --fname $test_split --srcfname $test_split.$SRC_LANG --tgtfname $test_split.$TGT_LANG
        echo "src file: $test_split.$SRC_LANG"
        head -n 2 $test_split.$SRC_LANG
        echo "tgt file: $test_split.$TGT_LANG"
        head -n 2 $test_split.$TGT_LANG
    done
fi

### Transforming bitext to include sentence codes
#TODO

### Running bpe using sentence piece
# TODO Confirm that bpe is happening correctly. Do we need alpha=0.5?
if [ ! -f $train_tgt_bpe ]; then
    echo "Running bpe on train src and target ...."
    python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src > $train_src_bpe
    python segment.py --model $TGT_SPM_MODEL_FILE --input $train_tgt > $train_tgt_bpe

    echo "train src bpe file: ${train_src_bpe}"
    head -n 5 $train_src_bpe
    echo "train tgt bpe file: ${train_tgt_bpe}"
    head -n 5 $train_tgt_bpe

    for var in 0 1 2
    do
        echo "Running bpe on test split ${var} ...."
        test_split_src=$data_dir/en_ja_split.test${var}.$SRC_LANG
        test_split_tgt=$data_dir/en_ja_split.test${var}.$TGT_LANG

        python segment.py --model $SRC_SPM_MODEL_FILE --input $test_split_src > $test_split_src.bpe
        python segment.py --model $TGT_SPM_MODEL_FILE --input $test_split_tgt > $test_split_tgt.bpe

        echo "src bpe file: ${test_split_src}.bpe"
        head -n 5 ${test_split_src}.bpe
        echo "tgt bpe file: ${test_split_src}.bpe"
        head -n 5 ${test_split_src}.bpe
    done
fi


### Fairseq Preprocessing
#TODO
#fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG  \
# --trainpref $data_links_lang/train.sp \
# --validpref $data_links_lang/test0.sp \
# --testpref  $data_links_lang/test1.sp,$data_links_lang/test2.sp \
# --workers 30 \
# --tgtdict /exp/mpost/duo20/runs/models/${trg}.$matt_run_num/dict.${trg}.txt  \
# --srcdict /exp/mpost/duo20/runs/models/${trg}.$matt_run_num/dict.${src}.txt  \
# --destdir $databin_lang

### Fairseq Training
#TODO

### Decoding and Evaluation
#TODO








