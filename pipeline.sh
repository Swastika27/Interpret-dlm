#!/bin/bash

set  -e # stop on error

seq_len=512
seq_per_shard=1024
layer=6

N_TRAIN=800000
N_VAL=40000
N_TEST=40000

disk2_embed_dir=/mnt/disk2/2005027/data/embeddings
docker_base=/workspace
docker_wdr=/workspace/mnt/disk1/swastika/Interpret-dlm

expansion_factor=64
top_k=32
lr=0.0003
batch_size=512
checkpoint_freq=10000
perf_log_freq=1000
# For reproducibility
SEED=42

act_size=256
dict_size=$(($act_size * $expansion_factor))
num_train_tokens=$(($N_TRAIN * $seq_len))



eval "$(conda shell.bash hook)"
conda activate interpret

# remove blacklist regions and centromere regions from primary assembly. these regions are highly repetitive but not biologically regulatory regions.

python3 data_utils/make_windows.py \
--fasta data/training/GRCh38.primary_assembly.genome.fa \
--out_dir data/preprocessed \
--window $seq_len \
--stride $seq_len \
--max_n_frac 0.01 \
--blacklist_bed data/training/hg38-blacklist.v2.bed \
--centromere_bed data/training/hg38_centromeres.bed \
--seed $SEED


echo "Total windows after removing blacklist and centromere regions:"
wc -l data/preprocessed/train_windows.bed
wc -l data/preprocessed/test_windows.bed
# conda install -c bioconda bedtools
# split into windows length <= seq_len

bedtools makewindows -b data/preprocessed/train_windows.bed -w $seq_len > train.w${seq_len}.bed
bedtools makewindows -b data/preprocessed/test_windows.bed -w $seq_len > test.w${seq_len}.bed

echo "Total windows before filtering:"
wc -l train.w${seq_len}.bed
wc -l test.w${seq_len}.bed

# remove windows lenght less than < L
awk -v L=$seq_len '($3-$2)==L' train.w${seq_len}.bed > train.w${seq_len}.full.bed
awk -v L=$seq_len '($3-$2)==L' test.w${seq_len}.bed  > test.w${seq_len}.full.bed

echo "Total windows after filtering:"
wc -l train.w${seq_len}.full.bed
wc -l test.w${seq_len}.full.bed

shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TRAIN train.w${seq_len}.full.bed > data/preprocessed/train.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TEST test.w${seq_len}.full.bed > data/preprocessed/test.sub.bed

# run hyena docker container to use hyena model for embedding extraction
CONTAINER_NAME="hyena"

if [ -n "$(docker ps -f "name=^/${CONTAINER_NAME}$" -f "status=running" -q)" ]; then
    echo "The container $CONTAINER_NAME is running."
else
    echo "The container $CONTAINER_NAME is not running."
    docker start $CONTAINER_NAME
fi

# generate embeddings for train split and save to data/embeddings
split="train"
if [ ! -d "data/embeddings/$split/layer_${layer}" ]; then
    echo "Extracting embeddings for $split split..."
    docker exec $CONTAINER_NAME bash -c "
    cd /workspace/mnt/disk1/swastika/Interpret-dlm &&
    python main/extract_hyena_embeddings.py \
    --fasta data/raw/GRCh38.primary_assembly.genome.fa \
    --bed data/preprocessed/$split.sub.bed \
    --split $split \
    --save_dir data/embeddings \
    --seq_len $seq_len \
    --layers $layer \
    --batch_size $seq_per_shard \
    --dtype_save float32
    "
else
    echo "Embeddings for $split split already exist, skipping extraction."
fi


model_basename="layer${layer}_${dict_size}_batchtopk_${top_k}_${lr}"

# Gated SAE alternative: set l1_coeff_gated / gated_aux_coeff and use model_basename_gated in eval paths below.
# l1_coeff_gated=0.01
# gated_aux_coeff=1.0
# model_basename_gated="layer${layer}_${dict_size}_gated_l1${l1_coeff_gated}_aux${gated_aux_coeff}_${lr}"

# train SAE on training embeddings (BatchTopK)
python main/BatchTopK/main.py \
    --layer $layer \
    --num_tokens $num_train_tokens \
    --top_k $top_k \
    --dict_size $dict_size \
    --batch_size $batch_size \
    --perf_log_freq $perf_log_freq \
    --checkpoint_freq $checkpoint_freq \

# Gated SAE training example (uncomment to run instead of BatchTopK; comment out the block above):
# python main/BatchTopK/main.py \
#     --sae_type gated \
#     --layer $layer \
#     --num_tokens $num_train_tokens \
#     --dict_size $dict_size \
#     --batch_size $batch_size \
#     --l1_coeff $l1_coeff_gated \
#     --gated_aux_coeff $gated_aux_coeff \
#     --perf_log_freq $perf_log_freq \
#     --checkpoint_freq $checkpoint_freq \

# after training, delete training embeddings to save space (optional)
# rm -rf data/embeddings/train/layer_${layer}


disk2_embed_dir=/workspace/mnt/disk2/2005027/data/embeddings
# generate embeddings for test split (optional, can also be done before training)
for split in "test"; do
    if [ ! -d "$disk2_embed_dir/$split/layer_${layer}" ]; then
        echo "Did not find directory $disk2_embed_dir/$split/layer_${layer}, extracting embeddings..."
        echo "Extracting embeddings for $split split..."
        docker exec $CONTAINER_NAME bash -c "
        cd $docker_wdr &&
        python main/extract_hyena_embeddings.py \
        --fasta data/raw/GRCh38.primary_assembly.genome.fa \
        --bed data/preprocessed/$split.sub.bed \
        --split $split \
        --save_dir $disk2_embed_dir \
        --seq_len $seq_len \
        --layers $layer \
        --batch_size $seq_per_shard \
        --dtype_save float32
        "
    else
        echo "Embeddings for $split split already exist, skipping extraction."
    fi
done

# Evaluate on test split (val_* args point at test data so evaluate_sae still runs both sections)
# need to run inside docker container since we need to patch hyena model
echo "Evaluating on test split..."
docker exec $CONTAINER_NAME bash -c "
    cd $docker_wdr &&
    python main/evaluate_sae.py \
      --sae_path "trained_models/$model_basename/checkpoints/step_$(($N_TRAIN - 1)).pt" \
      --cfg_path "trained_models/$model_basename/config.json" \
      --test_embeddings_path $docker_base/$disk2_embed_dir/test/layer_${layer} \
      --output_file "results/$model_basename/eval_metrics.yaml" \
      --device_str cuda \
      --test_bed_path data/preprocessed/test.sub.bed \
      --genome_path data/raw/GRCh38.primary_assembly.genome.fa \
      --hyenadna_checkpoint_path LongSafari/hyenadna-large-1m-seqlen-hf \
      --fidelity_max_seq_len $seq_len \
      --layer_idx $(($layer - 1)) 
      "

# find top firing tokens per feature
python main/find_top_activations.py \
        --sae_checkpoint  trained_models/$model_basename/checkpoints/step_$(($N_TRAIN - 1)).pt \
        --sae_cfg         trained_models/$model_basename/config.json \
        --embed_dir       $disk2_embed_dir \
        --layer           $layer \
        --splits          test \
        --top_n           200 \
        --out_dir         results/top_activations \
        --device          cuda \
        --batch_size      4096 \
        --num_workers     4