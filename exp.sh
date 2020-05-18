#!/usr/bin/env bash

### Variables
SRC_LANG=en
TGT_LANG=ja
BASE_DIR=${1:-/fs/clip-scratch/hoyle/sentence-codes/cmsc828b-project}
OWN_ENV=${2:-false}
ORIG_MODEL_DIR=$BASE_DIR/${TGT_LANG}.2
DUO_DATA_DIR=$BASE_DIR/data/staple-2020-train
DUO_EN_TGT_FILE=$BASE_DIR/data/staple-2020-train/en_$TGT_LANG/train.en_$TGT_LANG.2020-01-13.gold.txt
SRC_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.src.model
TGT_SPM_MODEL_FILE=$ORIG_MODEL_DIR/subword.trg.model
SRC_DICT=$ORIG_MODEL_DIR/dict.en.txt
TGT_DICT=$ORIG_MODEL_DIR/dict.$TGT_LANG.txt

# set LASER directory --- will be created if does not already exist
export LASER=/fs/clip-scratch/hoyle/sentence-codes/LASER 

# Clustering parameters
NUM_CLUSTERS=50
SUBTRACTION_METHOD=none # one of "mean", "prompt", or "none"

data_dir=$BASE_DIR/data/$TGT_LANG
mkdir -p $data_dir

models_dir=$BASE_DIR/models
mkdir -p $models_dir

train_src=$data_dir/train_sents.$SRC_LANG
train_tgt=$data_dir/train_sents.$TGT_LANG

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
if [ ! -f $train_tgt ]; then
    echo "Getting Training Data"
    python get_traintest_data.py --fname $DUO_EN_TGT_FILE --srcfname $train_src --tgtfname $train_tgt
    echo "train src file: ${train_src}"
    head -n 5 $train_src
    echo "train tgt file: ${train_tgt}"
    head -n 5 $train_tgt

    echo "Getting data splits ...."
    python new_create_splits.py --langs "$TGT_LANG" --output_dir $data_dir --duo_data_dir $DUO_DATA_DIR
    echo "Done getting data splits"

    echo "Converting train split to bitext ...."
    train_split=$data_dir/en_${TGT_LANG}_split.train
    python get_traintest_data.py --fname $train_split --srcfname $train_src --tgtfname $train_tgt --prefix train
    echo "train src file: ${train_src}"
    head -n 2 $train_src
    echo "train tgt file: ${train_tgt}"
    head -n 2 $train_tgt

    for var in 0 1 2
    do
        echo "Converting test${var} split to bitext ...."
        test_split=$data_dir/en_${TGT_LANG}_split.test${var}
        python get_traintest_data.py --fname $test_split --srcfname $test_split.$SRC_LANG --tgtfname $test_split.$TGT_LANG
        echo "src file: $test_split.$SRC_LANG"
        head -n 2 $test_split.$SRC_LANG
        echo "tgt file: $test_split.$TGT_LANG"
        head -n 2 $test_split.$TGT_LANG
    done
fi


### Transforming bitext to include sentence codes
# Set up LASER if it is not already (TODO: this is untested)
if [ ! -f ${LASER}/README.md ]; then
    echo "Cloning LASER repository..."
    git clone https://github.com/facebookresearch/LASER.git ${LASER}
    echo "Downloading LASER models..."
    bash ${LASER}/install_models.sh
    echo "Install external tools LASER relies on (uses modified script to fix japanese tokenization)"
    bash ./install_external_laser_tools.sh
fi


## First, embed the sentences: 
train_split=$data_dir/en_${TGT_LANG}_split.train
laser_prompt_file=$data_dir/laser/train.prompts
laser_translation_file=$data_dir/laser/train.translations

if [ ! -f ${laser_translation_file}.npy ]; then
    echo "Creating LASER-friendly files ..."
    python process_data_for_laser.py "${train_split}" "${laser_prompt_file}" "${laser_translation_file}"

    ## (Below copied from the LASER repo)
    model_dir="${LASER}/models"
    encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
    bpe_codes="${model_dir}/93langs.fcodes"
    
    echo "Creating LASER-embeddings ..."
    cat $laser_prompt_file \
    | python ${LASER}/source/embed.py \
        --encoder ${encoder} \
        --token-lang ${SRC_LANG} \
        --bpe-codes ${bpe_codes} \
        --output ${laser_prompt_file}.npy \
        --verbose

    cat $laser_translation_file \
    | python ${LASER}/source/embed.py \
        --encoder ${encoder} \
        --token-lang ${TGT_LANG} \
        --bpe-codes ${bpe_codes} \
        --output ${laser_translation_file}.npy \
        --verbose

    for var in 0 1 2
    do
        test_split_fname=$data_dir/en_${TGT_LANG}_split.test${var}.${TGT_LANG}

        cat ${test_split_fname} \
        | python ${LASER}/source/embed.py \
            --encoder ${encoder} \
            --token-lang ${TGT_LANG} \
            --bpe-codes ${bpe_codes} \
            --output $data_dir/laser/test${var}.translations.npy \
            --verbose
    done
fi

## Then, cluster the data
echo "Generating clusters from training data..."
clustering_name=k-${NUM_CLUSTERS}.subtract-${SUBTRACTION_METHOD}

python generate_clusters.py \
    --input_dir ${data_dir}/laser \
    --cluster_output_name $clustering_name \
    --num_clusters $NUM_CLUSTERS \
    --subtraction_method $SUBTRACTION_METHOD

echo "Assigning test sentences to clusters..."
for var in 0 1 2
do
    python assign_clusters.py \
        --embeddings_fpath ${data_dir}/laser/test${var}.translations.npy \
        --centroids_fpath ${data_dir}/laser/train.${clustering_name}.centroids.npy \
        --output_fpath ${data_dir}/laser/test${var}.${clustering_name}.map.json
done


### Fairseq Training
#TODO


module load cuda10.1/toolkit
module load cudnn/7.6.3_cuda10.1
nvidia-smi

save_dir=$BASE_DIR/new_model
mkdir -p $save_dir

fairseq-train $models_dir \
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








