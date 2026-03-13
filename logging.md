## step 1: Prepare data
### Download and preprocess training data
1. download GENCODE hg38 primary assembly

``` bash
wget -P data/training/ https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz
gzip -d data/training/GRCh38.primary_assembly.genome.fa.gz

```

2. download ENCODE blacklist

```bash
wget -P data/training/ https://raw.githubusercontent.com/Boyle-Lab/Blacklist/master/lists/hg38-blacklist.v2.bed.gz
gzip -d data/training/hg38-blacklist.v2.bed.gz
```

3. download UCSC cytoband table

```bash
wget -P data/training/ http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cytoBand.txt.gz
gzip -d data/training/cytoBand.txt.gz
```

4. remove blacklist regions and centromere regions from primary assembly. these regions are highly repetitive but not biologically regulatory regions.

```bash
python3 data_utils/make_windows.py \
--fasta data/training/GRCh38.primary_assembly.genome.fa \
--out_dir data/preprocessed \
--window 2048 \
--stride 2048 \
--max_n_frac 0.01 \
--blacklist_bed data/training/hg38-blacklist.v2.bed \
--centromere_bed data/training/hg38_centromeres.bed \
--seed 42
```

5. Break genome intervals ito fixed length windows (2k here)
```bash
cd data/preprocessed
conda install -c bioconda bedtools
# ssplit into windows lenght <= L
L=2000
bedtools makewindows -b train.bed -w $L > train.w${L}.bed
bedtools makewindows -b val.bed   -w $L > val.w${L}.bed
bedtools makewindows -b test.bed  -w $L > test.w${L}.bed

# remove windows lenght less than < L
awk -v L=$L '($3-$2)==L' train.w${L}.bed > train.w${L}.full.bed
awk -v L=$L '($3-$2)==L' val.w${L}.bed   > val.w${L}.full.bed
awk -v L=$L '($3-$2)==L' test.w${L}.bed  > test.w${L}.full.bed
```

6. Random subsample each split
```bash
N_TRAIN=200000
N_VAL=10000
N_TEST=10000
# For reproducibility
SEED=42
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TRAIN train.w${L}.full.bed > train.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_VAL val.w${L}.full.bed > val.sub.bed
shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) -n $N_TEST test.w${L}.full.bed > test.sub.bed
```

### download and preprocess annotation data
https://www.gencodegenes.org/human/
region: CHR
```bash
python data_utils/extract_gene_collapsed_labels.py \
--gtf data/annotations/gencode.v49.annotation.gtf \
--outdir data/annotations/gencode_v49_beds \
--promoter_upstream 1000 \
--promoter_downstream 100 \
--splice_radius 2 \
--write_combined
```

Repeats: https://genome.ucsc.edu/cgi-bin/hgTables?hgsid=3685956193_lbaEmcCX4BJc1sHWAEjKRcOfKHse&clade=mammal&org=Human&db=hg38&hgta_group=rep&hgta_track=knownGene&hgta_table=0&hgta_regionType=genome&position=chr7%3A155%2C799%2C529-155%2C812%2C871&hgta_outputType=primaryTable&hgta_outFileName=


CpG island: https://genome.ucsc.edu/cgi-bin/hgTables?hgsid=3685956193_lbaEmcCX4BJc1sHWAEjKRcOfKHse&clade=mammal&org=Human&db=hg38&hgta_group=regulation&hgta_track=rmsk&hgta_table=0&hgta_regionType=genome&position=chr7%3A155%2C799%2C529-155%2C812%2C871&hgta_outputType=primaryTable&hgta_outFileName=


ClinVar variant summary https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/version_summary.txt.gz
size 422,383,509; firstrelease; 2024-03-31 19:30:16; lastmodified: 2026-02-08 14:41:42

Encode CCREs
check unique tags
```bash
$ cut -f6 GRCh38-cCREs.bed | t
r ',' '\n' | sed 's/^ *//;s/ *$//' | sort -u | head
CTCF-bound
CTCF-only
dELS
DNase-H3K4me3
pELS
PLS
```

Extract tag-wise beds
```bash
$ python3 data_utils/extract_ccre_labels.py --in_bed data/annotations/encode_ccres/GRCh38-cCREs.bed --outdir data/annotations/encode_ccres/
```

UCSC repeats
```bash
python3 data_utils/extract_repeat_labels.py \
--in_csv data/annotations/repeats/rmsk.hg38.csv \
--outdir data/annotations/repeats \
--mode class

```

CpG islands
```bash
python data_utils/extract_cpg_labels.py \
--in_csv data/annotations/cpg/cpgIslandExt.csv \
--out_bed data/annotations/cpg/cpg_islands.hg38.bed

```

Clinvar labels benign, pathogenic
```bash
python data_utils/extract_clinvar_labels.py \
--infile data/annotations/clinvar/variant_summary.txt \
--outdir data/annotations/clinvar \
--assembly GRCh38 \
--germline_only \
--include_likely \
--exclude_conflicting
```


**MOdel change to HyenaDNA**
1. clone hyenadna repo
2. pull docker image 
3. run container
```bash
 docker run --gpus all -it  \
 --shm-size=16g \
 --name hyena  \
  -v /mnt/disk1/swastika/Interpret-dlm:/workspace     \
    hyenadna/hyena-dna     /bin/bash
```

embedding extraction works on cpu, the gpt script, not on gpu. GPT says it will need   pytorch, cuda rebuild to fix

python - <<EOF
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name())
EOF

pip uninstall torch -y

pip install torch==2.1.2 torchvision torchaudio \
--index-url https://download.pytorch.org/whl/cu118




cuda rebuild did not fix it. Actually, internal HyenaDNA calculation expects fp32, but GPT was casting  it to fp16 to save memory. Initializeeeed with fp32 and it worked (Is pytorch reinstallation responsible too? YES)



### Download motif databases
```bash
# JASPAR
mkdir -p motif_dbs
cd motif_dbs

wget https://jaspar.genereg.net/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt

mv JASPAR2024_CORE_vertebrates_non-redundant_pfms_meme.txt jaspar.meme
```

```bash
# Dfam repeat database
wget https://www.dfam.org/releases/Dfam_3.8/families/Dfam_curatedonly.embl.gz

# convert to meme format
python convert_embl_to_meme.py \
    --embl ../data/motif_dbs/Dfam_curatedonly.embl \
    --out ../data/motif_dbs/dfam_curated.meme
```


## step 2: Train SAE using Hyena embeddigns
1. Extract embeddings **inside hyena docker container**
```bash
# train split
python main/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/train.sub.bed \
--split train \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32

# val split
python main/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/val.sub.bed \
--split val \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32

# test split
python main/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/test.sub.bed \
--split test \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32
```

Modify main/BatchTopK/main.py file as needed 
python main/BatchTopK/main.py


## step: Evaluate sae training
```bash
  # With pre-saved embeddings (val and test sets):
  python main/evaluate_sae.py \
      --sae_path trained_models/layer8_8192_batchtopk_32_0.0003/checkpoints/step_199999.pt \
      --cfg_path trained_models/layer8_8192_batchtopk_32_0.0003/config.json \
      --val_embeddings_path data/embeddings/val/layer_8 \
      --test_embeddings_path data/embeddings/test/layer_8 \
      --output_file results/layer8_8192_batchtopk_32_0.0003/eval_metrics.yaml \
      --device cuda

  # With fidelity evaluation (requires sequences + HyenaDNA checkpoint):
  python main/evaluate_sae.py \
      --sae_path trained_models/layer8_8192_batchtopk_32_0.0003/checkpoints/step_199999.pt \
      --cfg_path trained_models/layer8_8192_batchtopk_32_0.0003/config.json \
      --val_embeddings_path data/embeddings/val/layer_8 \
      --test_embeddings_path data/embeddings/test/layer_8 \
      --output_file results/layer8_8192_batchtopk_32_0.0003/eval_metrics.yaml \
      --device cuda \
      --val_bed_path data/preprocessed/val.w512.full.bed \
      --test_bed_path data/preprocessed/test.w512.full.bed \
      --genome_path data/raw/GRCh38.primary_assembly.genome.fa \
      --hyenadna_checkpoint_path LongSafari/hyenadna-large-1m-seqlen-hf \
      --fidelity_max_seq_len 512 \
      --layer_idx 8 
```
## step3: finnd feature firing information
```bash
python3 feature_activation_batchtopk.py
```

v2
```bash
python3 find_top_tokens_per_feature.py \
    --sae_checkpoint trained_models/layer8_8192_batchtopk_32_0.0003/checkpoints/step_199999.pt \
    --sae_cfg        trained_models/layer8_8192_batchtopk_32_0.0003/config.json \
    --save_dir       data/embeddings \
    --layer          8 \
    --splits         train val test \
    --top_n          200 \
    --context_len    5 \
    --out_dir        results/layer8_8192_batchtopk_32_0.0003/top_tokens \
    --device         cuda \
    --batch_size     2048
```

## step 5: find feature-concept association
```bash
python3 main/eval_concept_batchtopk_final.py \
  --ckpt trained_models/layer8_bt8/checkpoints/final.pt \
  --data_root data/embeddings \
  --split train \
  --layer_dir_name layer_8 \
  --seq_len 2000 \
  --k_per_token 8 \
  --batch_tokens 512 \
  --concept_bed data/annotations/repeats/LINE.bed \
  --n_tokens 500 \
  --pos_frac_in_pool 0.5 \
  --pos_frac_in_batch 0.1 \
  --neg_mode background \
  --index_path data/embeddings/train/indices/line.pt \
  --out_csv results/layer8_bt8/feat_assoc/line.csv \
  --cache_emb 64
```

python main/find_concept_assoc.py \
    --sae_checkpoint trained_models/layer8_8192_batchtopk_32_0.0003/checkpoints/step_199999.pt\
    --sae_cfg        trained_models/layer8_8192_batchtopk_32_0.0003/config.json \
    --save_dir       data/embeddings \
    --layer          8 \
    --splits         train val test \
    --bed_dir        all_annotations/ \
    --out_dir        results/concept_analysis \
    --top_k_features 10 \
    --seed           42

## step 6: Summarize results
```bash
python3 main/summarize_assoc.py \
  --csv_glob "trained_models/layer8_bt8/feat_assoc/*.csv" \
  --top_examples_pt "trained_models/layer8_bt8/feature_top_examples.pt" \
  --fasta "data/raw/GRCh38.primary_assembly.genome.fa" \
  --out_dir "results/layer8_bt8/feat_assoc_summary/" \
  --topk 10 \
  --examples_per_feature 200 \
  --radius 20 \
  --center_k 9
```

## step 5+6: run the evaluation script
```bash
chmod +x all_annot_feat_assoc.sh

./all_annot_feat_assoc.sh
```

## step 7: find matching motifs from databases
```bash
python3 find_tomtom_matches.py \
  --meme_glob "../runs/sae/layer8_bt8/feat_assoc_summary/pls.csv.motifs.meme" \
  --jaspar_db "../data/motif_dbs/jaspar.meme" \
  --out_dir "../runs/sae/layer8_bt8/tomtom"

python3 find_tomtom_matches.py \
  --meme_glob "../runs/sae/layer8_bt8/feat_assoc_summary/pls.csv.motifs.meme" \
  --repeat_db "../data/motif_dbs/dfam_curated.meme" \
  --out_dir "../runs/sae/layer8_bt8/tomtom"
```

