#!/bin/bash
module load anaconda/2022.10
module load CUDA/11.6
source activate virEnv
sh run_script/run_pheme_bert_ft.sh