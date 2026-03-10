#!/bin/bash

layer="8"
split="test"
n_tokens="2048"

ckpt_step="1000"
output_root="results_${split}/l${layer}_ckpt_${ckpt_step}"
annotation_dir="data/annotations"
embedding_dir="data/embeddings"
model_dir="trained_models/layer8"
ckpt_path="$model_dir/checkpoints/step_0000${ckpt_step}.pt"

seq_len="2000"
k_per_token="8"
batch_tokens="512"
pos_frac_in_pool="0.5"
pos_frac_in_batch="0.1"
neg_mode="background"


echo "creating directory $output_root"
mkdir -p "$output_root"
echo "created output directory: $output_root"

mkdir -p "$output_root/feat_assoc"
echo "created directory $output_root/feat_assoc"

mkdir -p "$output_root/feat_assoc_summary"
echo "created directory $output_root/feat_assoc_summary"

if [ -d "$annotation_dir" ]; then
    echo "annotation directory exists: $annotation_dir"
else
    echo "annotation directory does not exist: $annotation_dir"
    exit 1
fi

if [ -f "$ckpt_path" ]; then
    echo "checkpoint file exists: $ckpt_path"
else
    echo "checkpoint file does not exist: $ckpt_path"
    exit 1
fi

find "$annotation_dir" -type f -name "*.bed" | while read -r annotation_file; do
    
    filename=$(basename "$annotation_file" .bed)
    if [ ! -f "$output_root/feat_assoc/$split_$filename.csv" ]; then
        echo "Processing annotation file: $annotation_file"

        python3 main/eval_concept_batchtopk_final.py \
        --ckpt $ckpt_path \
        --data_root $embedding_dir \
        --split $split \
        --layer_dir_name layer_$layer \
        --seq_len $seq_len \
        --k_per_token $k_per_token \
        --batch_tokens $batch_tokens \
        --concept_bed $annotation_file \
        --n_tokens $n_tokens \
        --pos_frac_in_pool $pos_frac_in_pool \
        --pos_frac_in_batch $pos_frac_in_batch \
        --neg_mode $neg_mode \
        --index_path $embedding_dir/$split/indices/$filename.pt \
        --out_csv $output_root/feat_assoc/$filename.csv \
        --cache_emb 64

        else
            echo "Output CSV already exists for $annotation_file, skipping..."
        fi
done

# summarize results
python3 utils/summarize_assoc.py \
  --csv_glob "$output_root/feat_assoc/*.csv" \
  --top_examples_pt "$model_dir/feature_top_examples.pt" \
  --fasta "data/raw/GRCh38.primary_assembly.genome.fa" \
  --out_dir "$output_root/feat_assoc_summary/" \
  --topk 10 \
  --examples_per_feature 200 \
  --radius 10 \
  --center_k 9