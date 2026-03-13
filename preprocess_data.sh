#!/bin/bash

set  -e # stop on error

seq_len=512
seq_per_shard=1024
layer=6

N_TRAIN=200000
N_VAL=10000
N_TEST=10000
# For reproducibility
SEED=42
splits=("train" "val" "test")

eval "$(conda shell.bash hook)"
conda activate interpret
cd data/preprocessed
# conda install -c bioconda bedtools
# split into windows length <= seq_len

bedtools makewindows -b train.bed -w $seq_len > train.w${seq_len}.bed
bedtools makewindows -b val.bed   -w $seq_len > val.w${seq_len}.bed
bedtools makewindows -b test.bed  -w $seq_len > test.w${seq_len}.bed

echo "Total windows before filtering:"
wc -l train.w${seq_len}.bed
wc -l val.w${seq_len}.bed
wc -l test.w${seq_len}.bed

# remove windows lenght less than < L
awk -v L=$seq_len '($3-$2)==L' train.w${seq_len}.bed > train.w${seq_len}.full.bed
awk -v L=$seq_len '($3-$2)==L' val.w${seq_len}.bed   > val.w${seq_len}.full.bed
awk -v L=$seq_len '($3-$2)==L' test.w${seq_len}.bed  > test.w${seq_len}.full.bed

echo "Total windows after filtering:"
wc -l train.w${seq_len}.full.bed
wc -l val.w${seq_len}.full.bed
wc -l test.w${seq_len}.full.bed

shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TRAIN train.w${seq_len}.full.bed > train.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_VAL val.w${seq_len}.full.bed > val.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TEST test.w${seq_len}.full.bed > test.sub.bed

# run hyena docker container to use hyena model for embedding extraction
CONTAINER_NAME="hyena"

if [ -n "$(docker ps -f "name=^/${CONTAINER_NAME}$" -f "status=running" -q)" ]; then
    echo "The container $CONTAINER_NAME is running."
else
    echo "The container $CONTAINER_NAME is not running."
    docker start $CONTAINER_NAME
fi

# generate embeddings for each split and save to data/embeddings
for split in ${splits[@]}; do
echo "Extracting embeddings for $split split..."

docker exec $CONTAINER_NAME bash -c "
cd /workspace &&
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

done