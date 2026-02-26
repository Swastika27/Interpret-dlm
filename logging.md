### step 1: Prepare data
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

5. Brak genome intervals ito fixed length windows (2k here)
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


### step 2: Train SAE using DNABERT-2 embeddings

Training only (no validation, no plots/CSV):

```bash
python training/train_sae.py \
--fasta data/training/GRCh38.primary_assembly.genome.fa \
--train_bed data/preprocessed/train_windows.bed \
--out ckpt/sae.pt \
--pool_windows 512 --tokens_per_window 16 \
--batch_tokens 2048
```

Training + validation + CSV + plots:

```bash
python training/train_sae.py \
--fasta data/training/GRCh38.primary_assembly.genome.fa \
--train_bed data/preprocessed/train_windows.bed \
--val_bed data/preprocessed/val_windows.bed \
--out ckpt/sae.pt \
--metrics_csv logs/metrics.csv \
--pool_windows 512 --tokens_per_window 16 \
--plot_png logs/loss_curves.png \
--val_every 500 --val_batches 10
```

To “lower correlation” more aggressively, modify two values:
increase --pool_windows
decrease --tokens_per_window

1. DNABERT-2
* create a new environment and install dependencies following instructions of their github repository
* CONDA INSTALL BIOCONDA::PYFAIDX
* DNABERT_2 DOES NOT RETURN HIDDEN_STATE OUTPUTS EVEN WHEN HIDDEN_STATE=TRUE, SO EXTRACT EMBEDDINGS USING HOOKS

~~CURRENT CODE TAKES **MODEL_MAX_LENGTH=512** FROM CONFIG ~~-> fixed

### step 2.1: Train SAE using Hyena embeddigns
1. Extract embeddings **inside hyena docker container**
```bash
python training/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/train.sub.bed \
--split train \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32

python training/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/val.sub.bed \
--split val \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32

python training/extract_hyena_embeddings.py \
--fasta data/raw/GRCh38.primary_assembly.genome.fa \
--bed data/preprocessed/test.sub.bed \
--split test \
--save_dir data/embeddings \
--seq_len 2000 \
--layers 8 \
--batch_size 64 \
--dtype_save float32
```

train ReLU SAE
```bash
python training/train_sae_saved_embedding.py \
  --emb_root data/embeddings \
  --layer 5 \
  --seq_len 2000 \
  --d_hidden 8192 \
  --batch_size 256 \
  --tokens_per_window 1 \
  --l1_coeff 1e-3 \
  --lr 3e-4 \
  --steps 20000 \
  --log_every 50 \
  --eval_every 500 \
  --eval_steps 50 \
  --save_every 2000
  ```

BatchTopK SAE
```bash
  python training/train_batchtopk.py \
    --data_root data/embeddings \
    --split_train train --split_val val \
    --layer_dir_name layer_8 \
    --d_in 256 --d_sae 8192 \
    --batch_tokens 2048 --seq_len 2000 \
    --k_per_token 8 \
    --l1_coeff 1e-4 \
    --lr 2e-4 --weight_decay 0.0 \
    --max_steps 1000000 \
    --log_every 500 --val_every 2000 --ckpt_every 5000 \
    --out_dir runs/sae/layer8_bt8
```

normalize activations (using activation of validation set, interPLM style)
```bash
python training/normalize_sae_val.py \
    --ckpt runs/sae/layer5_din256_dh8192_L2000_l10.001_bs256_seed42/ckpt_step_0020000.pt \
    --val_layer_dir data/embeddings/val/layer_5 \
    --out_ckpt runs/sae/layer5_din256_dh8192_L2000_l10.001_bs256_seed42/ckpt_step_00020000.interplm_norm.pt \
    --device cuda --batch_tokens 8192
```

### step 3: find feature-concept associations
```bash
conda install conda-forge::intervaltree
```
Search features
```bash
python training/feature_search_farbg.py \
  --fasta data/raw/GRCh38.primary_assembly.genome.fa \
  --feature_bed data/annotations/cpg/cpg_islands.hg38.bed \
  --sae_ckpt runs/sae/layer5_din256_dh8192_L2000_l10.001_bs256_seed42/ckpt_step_00020000.interplm_norm.pt \
  --out_dir runs/cpg_search_perbase_farbg \
  --model_id LongSafari/hyenadna-large-1m-seqlen-hf \
  --layer 5 \
  --seq_len 5000 \
  --n_pos 100 --n_neg 100 \
  --seed 42 \
  --device cuda
```

python promoter_feature_search_perbase_farbg.py \
  --fasta data/hg38.primary.fa \
  --promoter_bed data/annotations/gencode/promoter_TSS_1000up_100down.bed \
  --sae_ckpt runs/sae/<your_run>/ckpt_step_XXXXXXX.pt \
  --out_dir runs/promoter_feature_search \
  --model_id LongSafari/hyenadna-large-1m-seqlen-hf \
  --layer 5 \
  --seq_len 2000 \
  --n_pos 500 --n_neg 500 \
  --seed 42 \
  --device cuda \
  --save_tensors

### step 3: Collect annotations + process them
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

**don't forget "conda activate dna"!**

**MOdel change to HyenaDNA**
1. clone hyenadna repo
2. pull docker image 
3. run container
```bash


 docker run --gpus all -it  \
 --shm-size=16g \
 --name hyena  \
  -v /mnt/disk1/swastika/Interpretability:/workspace     \
    hyenadna/hyena-dna     /bin/bash
```

embeddiiiing extractiooonnnnnnnnnnnnnnnnnnnnn wwworks on cpu, the gpt scriiipt, not on gpuuu. GPT says itttttttttttttttttttttttttttttttttttttttttttttttttt will need   pytorch, cudaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa rebuild toooooo fix

root@0eb1763ec4ad:/wdr# cd ../workspace/code/test_fms/
root@0eb1763ec4ad:/workspace/code/test_fms# pip uninstall -y torch torchvision torchaudio
Found existing installation: torch 1.13.0
Uninstalling torch-1.13.0:
  Successfully uninstalled torch-1.13.0
Found existing installation: torchvision 0.14.0
Uninstalling torchvision-0.14.0:
  Successfully uninstalled torchvision-0.14.0






hyenadna gives per tooooooooooooooooooooooken embeddings

Iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii can proceed as in InteeeeeerPPPPLM


cuda rebuild did not fix it. Actually, internal HyyyyyyyenaDNA calculationnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn expects fffffffffffp32, but GPT was casting  it to fp16 to save memory. Initializeeeed with fp32 and it worked (Is pytorch reinstallation responsible too? idkkkkkkkkkkkkkkkkkkkkkkk)

