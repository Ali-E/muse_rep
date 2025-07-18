CORPUS="books"

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR="muse-bench/MUSE-Books_target"
LLAMA_DIR="meta-llama/Llama-2-7b-hf"

MAX_LEN=2048
EPOCHS=5
LR='1e-5'
PER_DEVICE_BATCH_SIZE=8 # 4 GPUs
FT_EPOCHS=1
FT_LR='1e-5'

SEED=1


# algo_list=('npo' 'npo_gdr' 'npo_klr')
# algo_list=('npo')
algo_list=('npo' 'npo_gdr')
forget_portion_list=(0.05 0.1)
# forget_portion_list=(0.05 0.1 0.25 0.5 0.75)
# forget_portion_list=(0.05 0.1 0.25 0.5 0.75 1.0)


for algo in "${algo_list[@]}"; do
    for forget_portion in "${forget_portion_list[@]}"; do
        out_dir="./ckpt/$CORPUS/${algo}_${forget_portion}_s${SEED}"
        # if portion is 1.0 don't include seed in output directory
        if [ "$forget_portion" == "1.0" ]; then
            out_dir="./ckpt/$CORPUS/${algo}_${forget_portion}"
        fi
        CUDA_VISIBLE_DEVICES=0,1,2,3 python unlearn.py \
            --algo $algo \
            --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
            --data_file $FORGET --retain_data_file $RETAIN \
            --out_dir $out_dir \
            --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
            --forget_portion $forget_portion \
            --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
            --seed $SEED
    done
done
