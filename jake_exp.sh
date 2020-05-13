#!/usr/bin/env bash

### Variables
SRC_LANG=en
TGT_LANG=ja
BASE_DIR=/exp/jbremerman/cmsc828b-project
DATA_DIR=/exp/hkhayrallah/duolingo_sharedtask_2020/data
DUO_DATA_DIR=$DATA_DIR/staple-2020-train
DUO_EN_JA_FILE=$DATA_DIR/en_ja/train.en_ja.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.src.model
TGT_SPM_MODEL_FILE=$BASE_DIR/ja.2/subword.trg.model

data_dir=$BASE_DIR/data
mkdir -p $data_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG

train_src_bpe=$train_src.bpe
train_tgt_bpe=$train_tgt.bpe

### Creating and activating virtual environment

#project_venv=$BASE_DIR/venv
#if [ ! -d "${project_venv}" ]; then
#  mkdir -p $project_venv
#  virtualenv -p python3 $project_venv
#  source $project_venv/bin/activate
#  #TODO replace pytorch with appropriate version
#  pip install torch torchvision
#  pip install -r $BASE_DIR/requirements.txt
#  deactivate
#fi
#source $project_venv/bin/activate

source activate fairseq

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
    python get_traintest_data.py --fname $train_split --srcfname $train_src --tgtfname $train_tgt --prefix train
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
#sed -e 's/^/<CODE-123>/' $train_tgt > ${train_tgt}.code

cat $train_tgt | while read line; do
  echo "<CODE-$(($RANDOM % 10))> $line"
done > ${train_tgt}.code


### Running bpe using sentence piece
# TODO Confirm that bpe is happening correctly. Do we need alpha=0.5?
if [ ! -f $train_tgt_bpe ]; then
    echo "Running bpe on train src and target ...."
    python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src > $train_src_bpe
    python segment.py --model $TGT_SPM_MODEL_FILE --input ${train_tgt}.code > $train_tgt_bpe

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
code=1
model=
tmpdir=$(mktemp -d --tmpdir=./scratch)
test_src=
checkpoint=
output=

echo "TMPDIR=$tmpdir"

cat $test_src | $model/preprocess.sh > $tmpdir/test.src
cat $tmpdir/test.src | while read line; do
    echo "<CODE-${code}>"
done > $tmpdir/test.trg

fairseq-preprocess --source-lang src --target-lang trg --testpref $tmpdir/test --destdir $tmpdir --scrdict $model/dict.src.txt --tgtdict $model/dict.trg.txt > $tmpdir/preprocess.log

fairseq-generate $tmpdir --path $checkpoint/checkpoint_best.pt --gen-subset test --max-tokens 500 --max-sentences 100 --prefix-size 5 > $tmpdir/out

mv $tmpdir/out $output

rm -rf $tmpdir


