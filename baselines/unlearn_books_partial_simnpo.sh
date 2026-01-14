#!/bin/bash

CORPUS="books"

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

# TARGET_DIR="muse-bench/MUSE-Books_target"
TARGET_DIR="/scratch/aebrahim/muse_rep/finetuned_books_model/"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

# TARGET_DIR="meta-llama/Meta-Llama-3-8B"
# LLAMA_DIR="meta-llama/Meta-Llama-3-8B"

MAX_LEN=2048
EPOCHS=1
LR='1e-5'
# LR='1e-3'
PER_DEVICE_BATCH_SIZE=8  # Reduced from 8 to 2, with gradient_accumulation_steps=4 effective batch is 2*4*4=32
FT_EPOCHS=1
FT_LR='1e-5'

SEED=1


# algo_list=('npo' 'npo_gdr' 'npo_klr')
# algo_list=('rmu' 'npo_klr' 'simnpo')
# algo_list=('gdr_simnpo')
algo='gdr_simnpo'
# algo_list=('rmu')
# algo_list=('npo' 'npo_gdr')
# forget_portion_list=(0.75 1.0)
# forget_portion_list=(0.05 0.1 0.25 0.5)
# forget_portion_list=(0.05 0.1 0.25 0.5)
# forget_portion_list=(0.05 0.1 0.25 0.5 0.75 1.0)
# forget_portion_list=(0.05 0.1 0.25 0.5 1.0)
forget_portion_list=(0.1)
# forget_portion_list=(0.25 0.5)
# forget_portion_list=(1.0)


beta_vals=(3.0 4.0)
gamma_vals=(0.1 0.05)


for gamma in "${gamma_vals[@]}"; do
    for beta in "${beta_vals[@]}"; do
        for forget_portion in "${forget_portion_list[@]}"; do
            out_dir="./Llama2_ft_mine/ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_UW_s${SEED}"
            # out_dir="./ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_s${SEED}"
            if [ "$forget_portion" == "1.0" ]; then
                out_dir="./Llama2_ft_mine/ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_UW_s${SEED}"
                # out_dir="./ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_s${SEED}"
            fi
            CUDA_VISIBLE_DEVICES=0,2 python unlearn.py \
                --algo $algo \
                --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
                --data_file $FORGET --retain_data_file $RETAIN \
                --out_dir $out_dir \
                --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
                --forget_portion $forget_portion \
                --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
                --seed $SEED \
                --auto_upsample \
                --beta $beta \
                --gamma $gamma \
                --save_only_final \
                --use_wikitext --wikitext_coeff 0.1 \
                --retain_coeff 0.9
        done
    done
done


                # --scale_retain_with_forget_portion \
                # --retain_data_file ../data/books/raw/retain1.txt \