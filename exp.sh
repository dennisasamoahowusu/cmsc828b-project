#!/usr/bin/env bash

### Variables
SRC_LANG=en
TGT_LANG=ja
BASE_DIR=/Users/dennis/coding/cmsc828b-project
DUO_DATA_DIR=$BASE_DIR/dataverse_files/staple-2020-train
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$SRC_LANG.model
TGT_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.$TGT_LANG.model
SRC_DICT=$BASE_DIR/ja.2/dict.en.txt
TGT_DICT=$BASE_DIR/ja.2/dict.ja.txt

data_dir=$BASE_DIR/data
mkdir -p $data_dir

models_dir=$BASE_DIR/models
mkdir -p $models_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG

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
if [ ! -f $train_tgt.sp.$TGT_LANG ]; then
    echo "Running sp on train src and target ...."
    python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src > $train_src-$TGT_LANG.sp.$SRC_LANG
    python segment.py --model $TGT_SPM_MODEL_FILE --input $train_tgt > $train_src-$TGT_LANG.sp.$TGT_LANG

    echo "train src bpe file: $train_src.sp.$SRC_LANG"
    head -n 2 $train_src.sp.$SRC_LANG
    echo "train tgt bpe file: $train_tgt.sp.$TGT_LANG"
    head -n 2 $train_tgt.sp.$TGT_LANG

    for var in 0 1 2
    do
        echo "Running sp on test split ${var} ...."
        test_split_src=$data_dir/en_ja_split.test${var}.$SRC_LANG
        test_split_tgt=$data_dir/en_ja_split.test${var}.$TGT_LANG

        python segment.py --model $SRC_SPM_MODEL_FILE --input $test_split_src > $test_split_src-$TGT_LANG.sp.$SRC_LANG
        python segment.py --model $TGT_SPM_MODEL_FILE --input $test_split_tgt > $test_split_src-$TGT_LANG.sp.$TGT_LANG

        echo "src bpe file: $test_split_src.sp.$SRC_LANG"
        head -n 2 $test_split_src.sp.$SRC_LANG
        echo "tgt bpe file: $test_split_tgt.sp.$TGT_LANG"
        head -n 2 $test_split_tgt.sp.$TGT_LANG
    done
fi


### Fairseq Preprocessing
echo "Running fairseq preprocessing"
fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG  \
 --trainpref $train_src-$TGT_LANG.sp \
 --validpref $data_dir/en_ja_split.test0.$SRC_LANG-$TGT_LANG.sp \
 --testpref  $data_dir/en_ja_split.test1.$SRC_LANG-$TGT_LANG.sp,$data_dir/en_ja_split.test2.$SRC_LANG-$TGT_LANG.sp \
 --workers 1 \
 --tgtdict $TGT_DICT  \
 --srcdict $SRC_DICT  \
 --destdir $models_dir \
 --cpu


### Fairseq Training
#TODO

### Decoding and Evaluation
#TODO








