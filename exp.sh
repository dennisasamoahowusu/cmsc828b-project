#!/usr/bin/env bash

### Variables
SRC_LANG=en
TGT_LANG=ja
BASE_DIR=${1:-/Users/dennis/coding/cmsc828b-project}
OWN_ENV=${2:-false}
ORIG_MODEL_DIR=$BASE_DIR/${TGT_LANG}.2
CLUSTER_DIR=$BASE_DIR/laser-embeds
DUO_DATA_DIR=$BASE_DIR/dataverse_files/staple-2020-train
DUO_EN_JA_FILE=$BASE_DIR/dataverse_files/staple-2020-train/en_ja/train.en_ja.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.src.model
TGT_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.trg.model
SRC_DICT=$ORIG_MODEL_DIR/dict.en.txt
TGT_DICT=$ORIG_MODEL_DIR/dict.ja.txt

data_dir=$BASE_DIR/data
mkdir -p $data_dir

models_dir=$BASE_DIR/models
mkdir -p $models_dir

### Creating and activating virtual environment
project_venv=$BASE_DIR/venv
if [ ! -d "${project_venv}" ] && [ $OWN_ENV = "false" ]; then
  mkdir -p $project_venv
  virtualenv -p python3 $project_venv
  source $project_venv/bin/activate
  #TODO replace pytorch with appropriate version
  pip install torch torchvision
  pip install -r $BASE_DIR/requirements.txt
  deactivate
fi

if [ $OWN_ENV = "true" ]; then
  source activate whale2020
else
  source $project_venv/bin/activate
fi

### Transforming duolingo data to bitext
### Creating train and test splits from duolingo data
cd $BASE_DIR
if [ ! -f $data_dir/en_ja_split.test1.$SRC_LANG ]; then
#    echo "Getting Training Data"
#    python get_traintest_data.py --fname $DUO_EN_JA_FILE --srcfname $train_src --tgtfname $train_tgt
#    echo "train src file: ${train_src}"
#    head -n 5 $train_src
#    echo "train tgt file: ${train_tgt}"
#    head -n 5 $train_tgt

    echo "Getting data splits ...."
    python new_create_splits.py --langs "ja" --output_dir $data_dir --duo_data_dir $DUO_DATA_DIR
    echo "Done getting data splits"
#
#    echo "Converting train split to bitext ...."
#    train_split=$data_dir/en_ja_split.train
#    python get_traintest_data.py --fname $train_split --srcfname $train_src --tgtfname $train_tgt --prefix train
#    echo "train src file: ${train_src}"
#    head -n 2 $train_src
#    echo "train tgt file: ${train_tgt}"
#    head -n 2 $train_tgt

    for var in 1 2
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
sc_output_dir=$data_dir/sc

if [ ! -d "${sc_output_dir}" ]; then
    echo "Creating bitexts with sentence codes"

    mkdir -p $sc_output_dir

    for split in train test0
    do
        for method in subtract-mean subtract-none subtract-prompt
        do
            ja_with_sc_map=$CLUSTER_DIR/ja/$split.k-med.$method.map.json
            ja_prompts=$CLUSTER_DIR/ja/$split.prompts
            ja_trans=$CLUSTER_DIR/ja/$split.translations

            python $BASE_DIR/get_sc_bitexts.py --mfile $ja_with_sc_map --pfile $ja_prompts --tfile $ja_trans \
                    --sofile $sc_output_dir/$split-$method.sc.src --tofile $sc_output_dir/$split-$method.sc.tgt
        done
    done
fi

split=train
method=subtract-prompt

train_src=$sc_output_dir/$split-$method.sc.src
train_tgt=$sc_output_dir/$split-$method.sc.tgt

split=test0
val_src=$sc_output_dir/$split-$method.sc.src
val_tgt=$sc_output_dir/$split-$method.sc.tgt

train_sp=$data_dir/train.sp.$method
val_sp=$data_dir/val.sp.$method

### Running bpe using sentence piece
if [ ! -f $train_sp.$SRC_LANG ]; then
    echo "Running sp on train src and target ...."
    python segment.py --model $SRC_SPM_MODEL_FILE --input $train_src > $train_sp.$SRC_LANG
    python segment.py --model $TGT_SPM_MODEL_FILE --input $train_tgt > $train_sp.$TGT_LANG

    echo "train src bpe file: $train_sp.$SRC_LANG"
    head -n 2 $train_sp.$SRC_LANG
    echo "train tgt bpe file: $train_sp.$TGT_LANG"
    head -n 2 $train_sp.$TGT_LANG


    echo "Running sp on val src and target ...."
    python segment.py --model $SRC_SPM_MODEL_FILE --input $val_src > $val_sp.$SRC_LANG
    python segment.py --model $TGT_SPM_MODEL_FILE --input $val_tgt > $val_sp.$TGT_LANG

    echo "val src bpe file: $val_sp.$SRC_LANG"
    head -n 2 $val_sp.$SRC_LANG
    echo "val tgt bpe file: $val_sp.$TGT_LANG"
    head -n 2 $val_sp.$TGT_LANG


    for var in 1 2
    do
        echo "Running sp on test split ${var} ...."
        test_split_src=$data_dir/en_ja_split.test${var}.$SRC_LANG
        test_split_tgt=$data_dir/en_ja_split.test${var}.$TGT_LANG

        python segment.py --model $SRC_SPM_MODEL_FILE --input $test_split_src > $test_split_src-$TGT_LANG.sp.$SRC_LANG
        python segment.py --model $TGT_SPM_MODEL_FILE --input $test_split_tgt > $test_split_src-$TGT_LANG.sp.$TGT_LANG

        echo "src bpe file: $test_split_src-$TGT_LANG.sp.$SRC_LANG"
        head -n 2 $test_split_src-$TGT_LANG.sp.$SRC_LANG
        echo "tgt bpe file: $test_split_src-$TGT_LANG.sp.$TGT_LANG"
        head -n 2 $test_split_src-$TGT_LANG.sp.$TGT_LANG
    done
fi

mkdir -p $models_dir/$method

### Fairseq Preprocessing
echo "Running fairseq preprocessing"
fairseq-preprocess --source-lang $SRC_LANG --target-lang $TGT_LANG  \
 --trainpref $train_sp \
 --validpref $val_sp \
 --testpref  $data_dir/en_ja_split.test1.$SRC_LANG-$TGT_LANG.sp,$data_dir/en_ja_split.test2.$SRC_LANG-$TGT_LANG.sp \
 --workers 1 \
 --tgtdict $TGT_DICT  \
 --srcdict $SRC_DICT  \
 --destdir $models_dir/$method \
 --cpu


module load cuda10.1/toolkit
module load cudnn/7.6.3_cuda10.1
nvidia-smi

save_dir=$BASE_DIR/new_model_$method
mkdir -p $save_dir

fairseq-train $models_dir/$method \
  --restore-file /exp/mpost/duo20/runs/models/ja.2/checkpoint_best.pt \
  --fp16 \
  --memory-efficient-fp16 \
  --num-workers 0 \
  --source-lang en \
  --target-lang ja \
  --save-dir $save_dir \
  --seed 2 \
  --arch transformer \
  --share-decoder-input-output-embed \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --encoder-embed-dim 512 \
  --decoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 \
  --decoder-ffn-embed-dim 2048 \
  --encoder-attention-heads 8 \
  --decoder-attention-heads 8 \
  --dropout 0.1 \
  --attention-dropout 0.1 \
  --relu-dropout 0.1 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --clip-norm 0.0 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-7 --lr 0.0005 --min-lr 1e-9 \
  --max-tokens 8000 \
  --max-epoch 200 \
  --update-freq 10 \
  --ddp-backend=no_c10d \
  --no-epoch-checkpoints \
  --log-format json --log-interval 1  &> $save_dir/train.log

#supposed to have a --patience 10 hyperparam too...

### Decoding and Evaluation
#TODO








