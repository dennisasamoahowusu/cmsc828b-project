#!/bin/bash
#
# Usage: 
#
#   translate.sh source.txt output.txt CODE [args]
#
# Translates from file source.txt into sentence code CODE
#
# - source.txt is the source file
# - CODE is the sentence code
# - args is optional arguments passed to fairseq-generate

#$ -S /bin/bash -V -cwd -j y -o logs/
#$ -l gpu=1,num_proc=1,h_rt=0:59:00 -q gpu.q@@2080

source /opt/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate ~jbremerman/.conda/envs/whale2020

set -eu

module load cuda10.1/toolkit cudnn


PYTHONPATH+=:$HOME/code/fairseq

# path to source file
path=$1

# path to write to
output=$2

if [[ -s $output ]]; then
    echo "$output already exists ($(cat $output | wc -l) lines), quitting."
    exit
fi

# the sentence code to select
code=$3

# path to directory containing 'dict.src.txt' and 'checkpoint_best.pt'

model=/.../

shift
shift
shift

tmpdir=$(mktemp -d --tmpdir=/exp/scale19/data/ir_task/scratch)
echo "TMPDIR=$tmpdir"
echo "HOST=$(hostname)"
env | grep CUDA
env | grep SGE

if [[ $source == sacrebleu://* ]]; then
    # format: "sacrebleu://test-set/source-target
    testset=$(echo $source | cut -d/ -f3)
    langpair=$(echo $source | cut -d/ -f4)
    sacrebleu -t $testset -l $langpair --echo src | $model/preprocess.sh > $tmpdir/test.src
else
    cat $source | $model/preprocess.sh > $tmpdir/test.src
fi

cat $tmpdir/test.src | while read line; do
  echo "<$code>"
done > $tmpdir/test.trg

cat $target | $model/preprocess.sh en > $tmpdir/test.trg

fairseq-preprocess --source-lang src --target-lang trg --testpref $tmpdir/test --destdir $tmpdir  --srcdict $model/dict.src.txt --tgtdict $model/dict.trg.txt > $tmpdir/preprocess.log

fairseq-generate $tmpdir --path $checkpoint/checkpoint_best.pt --gen-subset test --max-tokens 200 --max-sentences 100 --prefix-size 1 "$@" > $tmpdir/out
mv $tmpdir/out $output

rm -rf $tmpdir
