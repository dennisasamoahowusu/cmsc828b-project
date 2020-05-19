#!/usr/bin/env bash

source activate whale2020

data_dir=/exp/jbremerman/cmsc828b-project/output/ja/mean

for file in $data_dir/fairseq/*.txt; do
	echo $file
	my_cands_extract.py -i $file -c 1 > $data_dir/staple/$(basename -- $file)
done
