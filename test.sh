#!/bin/bash

set -e

seq_len=512
seq_per_shard=1024
layer=6

N_TRAIN=800000
N_VAL=40000
N_TEST=40000

expansion_factor=32
top_k=32
epoch=10
lr=0.0003
batch_size=512
checkpoint_freq=100000
perf_log_freq=1000
# For reproducibility
SEED=42

act_size=256
dict_size=$(($act_size * $expansion_factor))
num_train_tokens=$(($N_TRAIN * $epoch * $seq_len))

disk2_embed_dir=/mnt/disk2/2005027/data/embeddings
docker_base="/workspace"
docker_wdr=/workspace/mnt/disk1/swastika/Interpret-dlm

CONTAINER_NAME="hyena"

eval "$(conda shell.bash hook)"
conda activate interpret

# model_basename="layer${layer}_${dict_size}_batchtopk_${top_k}_${lr}"

# Gated SAE alternative: set l1_coeff_gated / gated_aux_coeff and use model_basename_gated in eval paths below.
l1_coeff_gated=0.01
gated_aux_coeff=1.0
model_basename_gated="layer${layer}_${dict_size}_gated_l1${l1_coeff_gated}_aux${gated_aux_coeff}_${lr}"
model_basename=$model_basename_gated
# # train SAE on training embeddings (SAE_training)
# python main/SAE_training/main.py \
#     --layer $layer \
#     --num_tokens $num_train_tokens \
#     --top_k $top_k \
#     --dict_size $dict_size \
#     --batch_size $batch_size \
#     --perf_log_freq $perf_log_freq \
#     --checkpoint_freq $checkpoint_freq \

# # Gated SAE training example (uncomment to run instead of SAE_training; comment out the block above):
# python main/SAE_training/main.py \
#     --sae_type gated \
#     --layer $layer \
#     --num_tokens $num_train_tokens \
#     --dict_size $dict_size \
#     --batch_size $batch_size \
#     --l1_coeff $l1_coeff_gated \
#     --gated_aux_coeff $gated_aux_coeff \
#     --perf_log_freq $perf_log_freq \
#     --checkpoint_freq $checkpoint_freq \
#     # --embedding_glob test_shards/*.pt


# if [ -n "$(docker ps -f "name=^/${CONTAINER_NAME}$" -f "status=running" -q)" ]; then
#     echo "The container $CONTAINER_NAME is running."
# else
#     echo "The container $CONTAINER_NAME is not running."
#     docker start $CONTAINER_NAME
# fi

# for split in "val" "test"; do
#     if [ ! -d "$disk2_embed_dir/$split/layer_${layer}" ]; then
#         echo "Did not find directory $disk2_embed_dir/$split/layer_${layer}, extracting embeddings..."
#         echo "Extracting embeddings for $split split..."
#         docker exec $CONTAINER_NAME bash -c "
#         cd $docker_wdr &&
#         python main/extract_hyena_embeddings.py \
#         --fasta data/raw/GRCh38.primary_assembly.genome.fa \
#         --bed data/preprocessed/$split.sub.bed \
#         --split $split \
#         --save_dir $disk2_embed_dir \
#         --seq_len $seq_len \
#         --layers $layer \
#         --batch_size $seq_per_shard \
#         --dtype_save float32
#         "
#     else
#         echo "Embeddings for $split split already exist, skipping extraction."
#     fi
# done

# # Evaluate on val and test split
# # need to run inside docker container since we need to patch hyena model
# echo "Evaluating on val and test splits..."
# docker exec $CONTAINER_NAME bash -c "
#     cd $docker_wdr &&
#     python main/evaluate_sae.py \
#       --sae_path "trained_models/$model_basename/checkpoints/latest.pt" \
#       --cfg_path "trained_models/$model_basename/config.json" \
#       --val_embeddings_path $docker_base/$disk2_embed_dir/val/layer_${layer} \
#       --test_embeddings_path $docker_base/$disk2_embed_dir/test/layer_${layer} \
#       --output_file "results/$model_basename/eval_metrics.yaml" \
#       --device cuda \
#       --val_bed_path data/preprocessed/val.sub.bed \
#       --test_bed_path data/preprocessed/test.sub.bed \
#       --genome_path data/raw/GRCh38.primary_assembly.genome.fa \
#       --hyenadna_checkpoint_path LongSafari/hyenadna-large-1m-seqlen-hf \
#       --fidelity_max_seq_len $seq_len \
#       --layer_idx $(($layer - 1)) 
#       "


# # find top firing tokens per feature
# python main/find_top_activations.py \
#         --sae_checkpoint  trained_models/$model_basename/checkpoints/latest.pt \
#         --sae_cfg         trained_models/$model_basename/config.json \
#         --embed_dir       $disk2_embed_dir \
#         --layer           $layer \
#         --splits          val test \
#         --top_n           200 \
#         --out_dir         results/$model_basename/top_activations \
#         --device          cuda \
#         --batch_size      2048 \
#         --num_workers     4


# # find feature to concept association
# python main/annotate_top_activations.py \
#         --top_activations  results/$model_basename/top_activations/top_activations.pt \
#         --bed_dir          all_annotations \
#         --out_dir          results/$model_basename/feature_annotation_assoc

echo "running concept -> feature analysis for $model_basename"
python main/concept_feature_analysis.py \
        --sae_checkpoint  trained_models/$model_basename/checkpoints/latest.pt \
        --sae_cfg         trained_models/$model_basename/config.json \
        --save_dir        $disk2_embed_dir \
        --layer           $layer \
        --splits          val test \
        --bed_dir         all_annotations/ \
        --out_dir         results/$model_basename/concept_analysis \
        --device          cuda \
        --batch_size      1024 \
        --top_k_features  10 \
        --seed            $SEED

