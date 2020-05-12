matt_run_num=2
#prep data for en to each language using just the 1 best translation
#fairseq=/home/hltcoe/hkhayrallah/fairseq-para-b_dist_multi_pp-orig
fairseq=/exp/mpost/duo20/runs/tape4nmt/out/.packages/fairseq/fb76dac1c4e314db75f9d7a03cb4871c532000cb
for lang in  ja ko  ; do

src=en
trg=$lang
src_spm_model=/exp/mpost/duo20/runs/models/${trg}.$matt_run_num/subword.src.model
trg_spm_model=/exp/mpost/duo20/runs/models/${trg}.$matt_run_num/subword.trg.model



source activate  /home/hltcoe/hkhayrallah/.conda/envs/duolingo-2020

  for split in train test0 test1 test2; do
    

    databin_lang=/exp/hkhayrallah/duolingo_sharedtask_2020/data/processed/matt$matt_run_num/4k/duo/$src-$trg/1best
    data_links_lang=$databin_lang/links
    mkdir -p $data_links_lang

    raw_data_path=/exp/hkhayrallah/duolingo_sharedtask_2020/data/staple-2020-train/en_$lang/
    infile=$raw_data_path/split.${split}.en_${lang}.2020-01-13.gold.lc.txt
    src_out=$data_links_lang/${split}.raw.$src
    trg_out=$data_links_lang/${split}.raw.$trg



    
    #get out of the shared task format and in to bitext. using prefix test means that you only get the 1best
    python /exp/hkhayrallah/duolingo_sharedtask_2020/task_scripts/duolingo-sharedtask-2020/get_traintest_data.py   --fname $infile  --srcfname $src_out --tgtfname $trg_out --prefix test

#Apply BPE

~mpost/local/bin/spm_encode --model  $src_spm_model --output $data_links_lang/${split}.sp.$src $data_links_lang/${split}.raw.$src --alpha 0.5 --output_format=sample_piece

~mpost/local/bin/spm_encode --model  $trg_spm_model --output $data_links_lang/${split}.sp.$trg $data_links_lang/${split}.raw.$trg --alpha 0.5 --output_format=sample_piece



done
    
python $fairseq/preprocess.py --source-lang $src --target-lang $trg  \
 --trainpref $data_links_lang/train.sp \
 --validpref $data_links_lang/test0.sp \
 --testpref  $data_links_lang/test1.sp,$data_links_lang/test2.sp \
 --workers 30 \
 --tgtdict /exp/mpost/duo20/runs/models/${trg}.$matt_run_num/dict.${trg}.txt  \
 --srcdict /exp/mpost/duo20/runs/models/${trg}.$matt_run_num/dict.${src}.txt  \
 --destdir $databin_lang    

  done




