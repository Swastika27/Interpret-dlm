#!/bin/bash

set -e

seq_len=512
seq_per_shard=1024
layer=6

N_TRAIN=800000
N_VAL=40000
N_TEST=40000

expansion_factor=64
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
num_batches_total=$(( num_train_tokens / batch_size ))
batches_per_epoch=$(( num_batches_total / epoch ))

disk2_embed_dir=/mnt/disk2/2005027/data/embeddings
docker_base="/workspace"
docker_wdr=/workspace/mnt/disk1/swastika/Interpret-dlm

CONTAINER_NAME="hyena"

eval "$(conda shell.bash hook)"
conda activate interpret

# model_basename="layer${layer}_${dict_size}_batchtopk_${top_k}_${lr}"

# Gated SAE alternative: set l1_coeff_gated / gated_aux_coeff and use model_basename_gated in eval paths below.
l1_coeff_gated=0.5
gated_aux_coeff=1.0
model_basename_gated="layer${layer}_${dict_size}_gated_l1${l1_coeff_gated}_aux${gated_aux_coeff}_${lr}"
model_basename=$model_basename_gated
# # train SAE on training embeddings (BAtchtopk)
# python main/SAE_training/main.py \
#     --layer $layer \
#     --num_tokens $num_train_tokens \
#     --top_k $top_k \
#     --dict_size $dict_size \
#     --batch_size $batch_size \
#     --perf_log_freq $perf_log_freq \
#     --checkpoint_freq $checkpoint_freq \
      # --name $model_basename

# Gated SAE training example (uncomment to run instead of SAE_training; comment out the block above):
if [ -f "trained_models/$model_basename/checkpoints/step_${num_train_tokens}" ]; then
echo "Final checkpoint already exists. Skipping training..."
else
python main/SAE_training/main.py \
    --sae_type gated \
    --layer $layer \
    --num_tokens $num_train_tokens \
    --dict_size $dict_size \
    --batch_size $batch_size \
    --l1_coeff $l1_coeff_gated \
    --gated_aux_coeff $gated_aux_coeff \
    --perf_log_freq $perf_log_freq \
    --checkpoint_freq $checkpoint_freq \
    --name $model_basename \
    # --embedding_glob test_shards/*.pt
  fi


# Plot training metrics
echo "plotting training info"
python utils/plot_training_info.py trained_models/$model_basename

# Start running docker container forrrrrr evaluation
if [ -n "$(docker ps -f "name=^/${CONTAINER_NAME}$" -f "status=running" -q)" ]; then
    echo "The container $CONTAINER_NAME is running."
else
    echo "The container $CONTAINER_NAME is not running."
    docker start $CONTAINER_NAME
fi

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

# Checkpoints to analyse: one marker per training epoch (largest saved step <= epoch end) plus final step
CK_STEPS=()
for ep in $(seq 1 "$epoch"); do
  nominal=$(( ep * batches_per_epoch )) # should be -1 but I did not save that checkpoint
  # if [ "$nominal" -ge "$num_batches_total" ]; then nominal=$(( num_batches_total - 1 )); fi
  ck=$(( (nominal / checkpoint_freq) * checkpoint_freq ))
  CK_STEPS+=("$ck")
done

readarray -t EPOCH_CKPTS < <(printf '%s\n' "${CK_STEPS[@]}" | sort -nu)

for ckpt_step in "${EPOCH_CKPTS[@]}"; do
  sae_ckpt="trained_models/$model_basename/checkpoints/step_${ckpt_step}.pt"
  if [ ! -f "$sae_ckpt" ]; then
    echo "Skipping step ${ckpt_step}: checkpoint not found at $sae_ckpt"
    continue
  fi
  result_tag="${model_basename}/step${ckpt_step}"
  echo "========== Pipeline for checkpoint step ${ckpt_step} ($result_tag) =========="

  # Evaluate on val and test split (HyenaDNA fidelity — run inside docker)
  echo "Evaluating on val and test splits..."
  if [ -f "results/$result_tag/eval_metrics.yaml" ]; then
    echo "Output file already exists. Skipping..."
  else
  docker exec \
  --user $(id -u):$(id -g) \
  -e HF_HOME=/workspace/mnt/disk1/swastika/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/workspace/mnt/disk1/swastika/.cache/huggingface \
  $CONTAINER_NAME bash -c "
    cd $docker_wdr && \
    python main/evaluate_sae.py \
      --sae_path trained_models/$model_basename/checkpoints/step_${ckpt_step}.pt \
      --cfg_path trained_models/$model_basename/config.json \
      --val_embeddings_path $docker_base/$disk2_embed_dir/val/layer_${layer} \
      --test_embeddings_path $docker_base/$disk2_embed_dir/test/layer_${layer} \
      --output_file results/$result_tag/eval_metrics.yaml \
      --device cuda \
      --resume \
      --val_bed_path data/preprocessed/val.sub.bed \
      --test_bed_path data/preprocessed/test.sub.bed \
      --genome_path data/raw/GRCh38.primary_assembly.genome.fa \
      --hyenadna_checkpoint_path LongSafari/hyenadna-large-1m-seqlen-hf \
      --fidelity_max_seq_len $seq_len \
      --layer_idx $(($layer - 1)) \
      --resume
     "
  fi

  # Dense-feature diagnostics (exclude dense features, correlate with token stats, worst-MSE dims)
  # Read dense_top_n from eval_metrics.yaml (uses sparsity.highly_active_features, which is what you observed as "8").
  eval_yaml="results/$result_tag/eval_metrics.yaml"
  # #region agent log
  printf '{"sessionId":"ff640b","runId":"pre-fix","hypothesisId":"H0","location":"experiment.sh:eval_yaml","message":"Dense-top-N parse starting","data":{"eval_yaml":"%s","ckpt_step":"%s"},"timestamp":%s}\n' \
    "$eval_yaml" "$ckpt_step" "$(date +%s%3N)" >> debug-ff640b.log
  # #endregion
  dense_top_n=$(
    python - "$eval_yaml" <<'PY'
import sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit("Missing PyYAML. Install into this environment: pip install pyyaml") from e

path = Path(sys.argv[1])
if not path.is_file():
    print("8")  # safe fallback
    raise SystemExit(0)

data = yaml.safe_load(path.read_text(encoding="utf-8"))
def get_int(d, keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return int(cur)
    except Exception:
        return default

v = get_int(data, ["val", "sparsity", "highly_active_features"])
t = get_int(data, ["test", "sparsity", "highly_active_features"])

# Prefer a conservative choice (max across splits), fall back to 8 if missing.
vals = [x for x in [v, t] if x is not None]
print(max(vals) if vals else 8)
PY
  )
  # #region agent log
  printf '{"sessionId":"ff640b","runId":"pre-fix","hypothesisId":"H1","location":"experiment.sh:dense_top_n","message":"Dense-top-N parsed","data":{"dense_top_n":"%s"},"timestamp":%s}\n' \
    "$dense_top_n" "$(date +%s%3N)" >> debug-ff640b.log
  # #endregion
  echo "Running dense-feature epoch diagnostics (dense_top_n=$dense_top_n) for $result_tag"
  if [ -f "results/$result_tag/epoch_diagnostics/epoch_summary.csv" ]; then
    echo "Diagnostics already exist. Skipping..."
  else
    python main/sae_epoch_diagnostics.py \
      --sae_cfg trained_models/$model_basename/config.json \
      --checkpoints_glob trained_models/$model_basename/checkpoints/step_${ckpt_step}.pt \
      --save_dir $disk2_embed_dir \
      --layer $layer \
      --splits val test \
      --bed_dir all_annotations/ \
      --out_dir results/$result_tag/epoch_diagnostics \
      --device cuda \
      --eval_batch_size 1024 \
      --dense_top_n "$dense_top_n" \
      --dense_freq_threshold 0.10 \
      --assoc_f1_threshold 0.10 \
      --high_mse_top_k 20
  fi

  # python main/find_top_activations.py \
  #   --sae_checkpoint  trained_models/$model_basename/checkpoints/step_${ckpt_step}.pt \
  #   --sae_cfg         trained_models/$model_basename/config.json \
  #   --embed_dir       $disk2_embed_dir \
  #   --layer           $layer \
  #   --splits          val test \
  #   --top_n           200 \
  #   --out_dir         results/$result_tag/top_activations \
  #   --device          cuda \
  #   --batch_size      2048 \
  #   --num_workers     4 \
  #   --resume

  # echo "Anotating top activations with overlapping concepts" 
  # python main/annotate_top_activations.py \
  #   --top_activations  results/$result_tag/top_activations/top_activations.pt \
  #   --bed_dir          all_annotations \
  #   --out_dir          results/$result_tag/activation_concept_assoc \
  #   --resume

  echo "running concept -> feature analysis for $result_tag"
  if [ -f "results/$result_tag/feature_concept_analysis/summary.csv" ]; then
  echo "output file already exists. Skipping..."
  else
  python main/concept_feature_analysis.py \
    --sae_checkpoint  trained_models/$model_basename/checkpoints/step_${ckpt_step}.pt \
    --sae_cfg         trained_models/$model_basename/config.json \
    --save_dir        $disk2_embed_dir \
    --layer           $layer \
    --splits          val test \
    --bed_dir         all_annotations/ \
    --out_dir         results/$result_tag/feature_concept_analysis \
    --device          cuda \
    --batch_size      1024 \
    --top_k_features  10 \
    --seed            $SEED \
    --resume
  fi

  echo "running concept -> neuron analysis for $result_tag"
  if [ -f "results/$result_tag/neuron_concept_analysis/summary.csv" ]; then
  echo "Output file already exists. Skipping..."
  else
  python main/concept_feature_analysis.py \
    --raw_neurons \
    --sae_cfg         trained_models/$model_basename/config.json \
    --save_dir        $disk2_embed_dir \
    --layer           $layer \
    --splits          val test \
    --bed_dir         all_annotations/ \
    --out_dir         results/$result_tag/neuron_concept_analysis \
    --device          cuda \
    --batch_size      1024 \
    --top_k_features  10 \
    --seed            $SEED \
    --resume
  fi
done

python utils/plot_feature_neuron_concept_assoc.py \
    --results_root results/$model_basename \
    --out_dir results/$model_basename/concept_assoc_plots