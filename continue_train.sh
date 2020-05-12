#$ -S /bin/bash -V
#$ -cwd
#$  -q gpu.q@@2080 -q gpu.q   -l gpu=2,h_rt=999:99:99
#$ -j y -o /exp/hkhayrallah/duolingo_sharedtask_2020/expts/en-ja/matt_lc.2+duo_9x1best_and_all_0.1dropout_200epochs_0.0005lr/$JOB_NAME.o$JOB_ID

module load cuda10.1/toolkit
module load cudnn/7.6.3_cuda10.1   # latest is cudnn/7.6.4_cuda10.1, not on our system yet

#source activate /home/hltcoe/bthompson/anaconda3/envs/bigdata
source activate /home/hltcoe/mpost/.conda/envs/fairseq

nvidia-smi

hostname
printf -v DATE '%(%Y-%m-%d-%H-%M-%S)T\n' -1
echo 
fairseq=/exp/mpost/duo20/runs/tape4nmt/out/.packages/fairseq/fb76dac1c4e314db75f9d7a03cb4871c532000cb/
#fairseq=/exp/bthompson/wikimatrix/fairseq/fairseq_cli/
#fairseq=/home/hltcoe/hkhayrallah/fairseq-para-b_dist_multi_pp-orig
cd 
pwd
#git rev-parse HEAD
cd -

stdbuf -o0 -e0  python $fairseq/train.py /exp/hkhayrallah/duolingo_sharedtask_2020/data/processed/matt2/4k/duo/en-ja/9x1best_and_all \
  --restore-file /exp/mpost/duo20/runs/models/ja.2/checkpoint_best.pt \
  --fp16 --patience 10 \
  --memory-efficient-fp16 \
  --num-workers 0 \
  --source-lang en \
  --target-lang ja \
  --save-dir /exp/hkhayrallah/duolingo_sharedtask_2020/expts/en-ja/matt_lc.2+duo_9x1best_and_all_0.1dropout_200epochs_0.0005lr \
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
  --log-format json --log-interval 1  &> /exp/hkhayrallah/duolingo_sharedtask_2020/expts/en-ja/matt_lc.2+duo_9x1best_and_all_0.1dropout_200epochs_0.0005lr/train.log

  printf -v DATE '%(%Y-%m-%d-%H-%M-%S)T\n' -1
  qsub /exp/hkhayrallah/duolingo_sharedtask_2020/expts/en-ja/matt_lc.2+duo_9x1best_and_all_0.1dropout_200epochs_0.0005lr/test.qsub
  echo    
