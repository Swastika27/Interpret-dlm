#!/bin/bash

set  -e # stop on error

seq_len=512
seq_per_shard=1024
layer=7

N_TRAIN=800000
N_TEST=80000

disk2_embed_dir=/mnt/disk2/2005027/data/embeddings
docker_base=/workspace
docker_wdr=/workspace/mnt/disk1/swastika/Interpret-dlm


# For reproducibility
SEED=42




eval "$(conda shell.bash hook)"
conda activate interpret

# remove blacklist regions and centromere regions from primary assembly. these regions are highly repetitive but not biologically regulatory regions.

# python3 data_utils/make_windows.py \
# --fasta data/raw/GRCh38.primary_assembly.genome.fa \
# --out_dir data/preprocessed \
# --window $seq_len \
# --stride $seq_len \
# --max_n_frac 0.01 \
# --blacklist_bed data/raw/hg38-blacklist.v2.bed \
# --centromere_bed data/raw/hg38_centromeres.bed \
# --seed $SEED


echo "Total windows after removing blacklist and centromere regions:"
wc -l data/preprocessed/train_windows.bed
wc -l data/preprocessed/test_windows.bed
# conda install -c bioconda bedtools
# split into windows length <= seq_len

bedtools makewindows -b data/preprocessed/train_windows.bed -w $seq_len > data/preprocessed/train.w${seq_len}.bed
bedtools makewindows -b data/preprocessed/test_windows.bed -w $seq_len > data/preprocessed/test.w${seq_len}.bed

echo "Total windows before filtering:"
wc -l data/preprocessed/train.w${seq_len}.bed
wc -l data/preprocessed/test.w${seq_len}.bed

# remove windows lenght less than < L
awk -v L=$seq_len '($3-$2)==L' data/preprocessed/train.w${seq_len}.bed > data/preprocessed/train.w${seq_len}.full.bed
awk -v L=$seq_len '($3-$2)==L' data/preprocessed/test.w${seq_len}.bed  > data/preprocessed/test.w${seq_len}.full.bed

echo "Total windows after filtering:"
wc -l data/preprocessed/train.w${seq_len}.full.bed
wc -l data/preprocessed/test.w${seq_len}.full.bed

shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TRAIN data/preprocessed/train.w${seq_len}.full.bed > data/preprocessed/train.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TEST data/preprocessed/test.w${seq_len}.full.bed > data/preprocessed/test.sub.bed

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

for split in "test"; do
    if [ ! -d "$docker_base/$disk2_embed_dir/$split/layer_${layer}" ]; then
        echo "Did not find directory $docker_base/$disk2_embed_dir/$split/layer_${layer}, extracting embeddings..."
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