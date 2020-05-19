#!/usr/bin/env bash

source activate whale2020

module load cuda10.1/toolkit
module load cudnn/7.6.3_cuda10.1
nvidia-smi

model=/exp/jbremerman/cmsc828b-project/ja.2
test_src=/exp/jbremerman/cmsc828b-project/data/en_ja_split.test2.en-ja.sp.en
checkpoint=/exp/jbremerman/cmsc828b-project/new_model/

for i in {0..189..1}
    do
	code=$i
	output=/exp/jbremerman/cmsc828b-project/output/ja/${code}.txt
	tmpdir=/exp/jbremerman/cmsc828b-project/scratch/${code}

	mkdir -p $tmpdir

	echo "TMPDIR=$tmpdir"

#cat $test_src | $model/preprocess.sh > $tmpdir/test.src
	cat $test_src > $tmpdir/test.src

	cat $tmpdir/test.src | while read line; do
    	    echo "<CODE-${code}>"
    	    done > $tmpdir/test_precode.trg

	python segment.py --model $model/subword.trg.model --input $tmpdir/test_precode.trg > $tmpdir/test.trg

    	fairseq-preprocess --source-lang src --target-lang trg --testpref $tmpdir/test --destdir $tmpdir --srcdict $model/dict.src.txt --tgtdict $model/dict.trg.txt > $tmpdir/preprocess.log

    	fairseq-generate $tmpdir --path $checkpoint/checkpoint_last.pt --gen-subset test --max-tokens 500 --max-sentences 100 --prefix-size 5 > $tmpdir/out

    	mv $tmpdir/out $output

        rm -rf $tmpdir
done
