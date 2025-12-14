CORPUS="books"
# algo="npo"
algo="npo_gdr"
# algo="npo_klr"
algo_name="npo_gdr"
res_types=("knowmem_r" "knowmem_f" "privleak" "privleak++" "privleak_zlib" "fluency_wikitext" "fluency_c4" "fluency_lambada" "fluency_hellaswag" )
# res_types=("privleak" "privleak++" "privleak_zlib" )
# res_types=("knowmem_r" "knowmem_f" "privleak" "privleak++" "privleak_zlib" )
# res_types=("knowmem_f")
# res_types=("knowmem_r")

indices_seed=1

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.05_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.1_s1/" \
#     --names "${algo}_0.05" "${algo}_0.1" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --including_ratios "0.05" "0.1" \
#     --out_file "${CORPUS}_knowmem_f_${algo}.csv" \
#     --metrics "${res_types[@]}" 

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.05_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.1_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.25_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.5_s1/" \
#     --names "${algo}_0.05" "${algo}_0.1" "${algo}_0.25" "${algo}_0.5" \
#     --corpus "${CORPUS}" \
#     --epoch -1 \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_knowmem_f_some_${algo}.csv" \
#     --metrics "${res_types[@]}" 

python eval.py \
    --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.75_s1/" \
                             "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_1.0/" \
    --names "${algo}_0.75" "${algo}_1.0" \
    --corpus "${CORPUS}" \
    --epoch -1 \
    --indices_seed ${indices_seed} \
    --out_file "${CORPUS}_knowmem_f_rest_${algo}.csv" \
    --metrics "${res_types[@]}" 

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.05_s1/Epoch_1" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.1_s1/Epoch_1" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.25_s1/Epoch_1" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.5_s1/Epoch_1" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" \
#     --corpus "${CORPUS}" \
#     --epoch -1 \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_knowmem_f_some_${algo_name}.csv" \
#     --metrics "${res_types[@]}" 

# python eval.py \
#     --names "base" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_knowmem_f_base.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext


# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.5_s1/" \
#     --names "${algo}_0.5" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_knowmem_f_${algo}.csv" \
#     --metrics "${res_types[@]}" 

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.05_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.1_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.25_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.5_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.75_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_1.0/" \
#     --names "${algo}_0.05" "${algo}_0.1" "${algo}_0.25" "${algo}_0.5" "${algo}_0.75" "${algo}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --including_ratios "0.05" "0.1" "0.25" "0.5" "0.75" "1.0" \
#     --out_file "${CORPUS}_knowmem_f_${algo}.csv" \
#     --metrics "${res_types[@]}" 