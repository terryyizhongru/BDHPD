#!/bin/bash

# Only run testing over 5 folds using existing checkpoints.
# Usage: bash test_5fold.sh [cuda_device] [meta_root]
#   cuda_device: GPU id (default 0)
#   meta_root: metadata base directory (default matches run_5fold.sh)

cuda_device=${1:-0}
meta_root=${2:-"split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel"}
config_file=configs/config_early.yaml

wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
freeze_ssl=false

for fold in 1 2 3 4 5; do
  checkpoint_dir="/data/storage2/projects/early/BDHPD/run_${meta_root}/fold_${fold}"
  mkdir -p logs
  log_file="logs/testing_${meta_root}_fold${fold}_$(date +%Y%m%d_%H%M%S).log"

  echo "================ Testing fold ${fold} ================"
  echo "Starting testing fold ${fold} at $(date)" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck} \
    --Neurovoz_and_PC_GITA.train_metadata_path="${meta_root}/fold_${fold}/sub_splits/train.tsv" \
    --Neurovoz_and_PC_GITA.validation_metadata_path="${meta_root}/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
    --Neurovoz_and_PC_GITA.test_metadata_path="${meta_root}/fold_${fold}/test_early6PD6HC.tsv" \
    2>&1 | tee -a "${log_file}"

  echo "Testing fold ${fold} finished at $(date)" | tee -a "${log_file}"
  echo

done

echo "All 5 folds testing completed."
