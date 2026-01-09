#!/bin/bash

CORPUS="books"
res_types=("knowmem_r" "knowfacts_f" "privleak" "privleak++" "privleak_zlib" "fluency_wikitext" "fluency_c4" "fluency_lambada" "fluency_hellaswag" )
# res_types=("knowmem_r" "knowmem_f" "privleak" "privleak++" "privleak_zlib" "fluency_wikitext" "fluency_c4" "fluency_lambada" "fluency_hellaswag")
# res_types=("knowfacts_f")
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


# algo="gdr_simnpo"
# algo_name="simnpo_U_llama2"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}_beta0.5_gamma0.3.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length

# beta_vals=(1.0 0.5)
# gamma_vals=(0.4 0.5)
# out_dir="./Llama2_ft/ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_U_s${SEED}"
# if [ "$forget_portion" == "1.0" ]; then
    # out_dir="./Llama2_ft/ckpt/$CORPUS/${algo}_b${beta}_g${gamma}_${forget_portion}_U"

# algo="gdr_simnpo"
# beta=1.0
# gamma=0.5
# algo_name="simnpo_U_llama3_beta${beta}_gamma${gamma}"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}_e1.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length


algo="gdr_simnpo"
beta=1.5
gamma=0.7
algo_name="simnpo_U_llama3_beta${beta}_gamma${gamma}"
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.05_s1/" \
                             "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.1_s1/" \
                             "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.25_s1/" \
                             "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.5_s1/" \
                             "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_1.0_s1/" \
    --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
    --corpus "${CORPUS}" \
    --indices_seed ${indices_seed} \
    --out_file "${CORPUS}_results_${algo_name}.csv" \
    --metrics "${res_types[@]}" \
    --privleak_use_wikitext \
    --tokenizer_dir "meta-llama/Meta-Llama-3-8B" \
    --privleak_truncate_same_length


# algo="gdr_simnpo"
# beta=0.5
# gamma=0.4
# algo_name="simnpo_U_llama2_beta${beta}_gamma${gamma}"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length
# 
# 
# algo="gdr_simnpo"
# beta=0.5
# gamma=0.5
# algo_name="simnpo_U_llama2_beta${beta}_gamma${gamma}"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_b${beta}_g${gamma}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length

# algo="npo_gdr_wiki"
# algo_name="npo_gdr_wiki_U_llama2"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}_1e-5.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length
# 
# 
# algo="gdr_simnpo_wiki"
# algo_name="gdr_simnpo_wiki_U_llama2"
# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.05_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.1_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.25_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_0.5_U_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/Llama2_ft/ckpt/${CORPUS}/${algo}_1.0_U/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}_gamma0.3_beta1.0.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.75_s1/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_1.0/" \
#     --names "${algo}_0.75" "${algo}_1.0" \
#     --corpus "${CORPUS}" \
#     --epoch -1 \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_knowmem_f_rest_${algo}.csv" \
#     --metrics "${res_types[@]}" 

# python eval.py \
#     --model_dirs "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.05_PS/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.1_PS/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.25_PS/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_0.5_PS/" \
#                              "/scratch/aebrahim/muse_rep/baselines/ckpt/${CORPUS}/${algo}_1.0_PS/" \
#     --names "${algo_name}_0.05" "${algo_name}_0.1" "${algo_name}_0.25" "${algo_name}_0.5" "${algo_name}_1.0" \
#     --corpus "${CORPUS}" \
#     --epoch 0 \
#     --indices_seed ${indices_seed} \
#     --out_file "${CORPUS}_results_${algo_name}.csv" \
#     --metrics "${res_types[@]}" \
#     --privleak_use_wikitext \
#     --privleak_truncate_same_length

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
#     --out_file "${CORPUS}_knowfacts_f_base_redo.csv" \
#     --metrics "${res_types[@]}" \
    ## --privleak_use_wikitext


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