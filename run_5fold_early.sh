#!/bin/bash

cuda_device=${1:-0}
meta_root=${2:-"split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel"}
config_file=configs/config_early.yaml

wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
balance_dataloaders=true
freeze_ssl=false

for fold in 1 2 3 4 5; do
  checkpoint_dir="/data/storage2/projects/early/BDHPD/run_earlybalance${meta_root}/fold_${fold}"
  # Put logs under a structured directory per meta_root
  # Use only the last path component of meta_root to keep logs one-level deep
  meta_name="${meta_root##*/}"
  log_dir="logs/${meta_name}"
  mkdir -p "${log_dir}"
  log_file="${log_dir}/training_fold${fold}_$(date +%Y%m%d_%H%M%S).log"

  echo "================ Fold ${fold} ================"
  echo "Starting training fold ${fold} at $(date)" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --training.balance_dataloaders=${balance_dataloaders} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck} \
    --Neurovoz_and_PC_GITA.train_metadata_path="${meta_root}/fold_${fold}/sub_splits/train_earlybalance.tsv" \
    --Neurovoz_and_PC_GITA.validation_metadata_path="${meta_root}/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
    --Neurovoz_and_PC_GITA.test_metadata_path="${meta_root}/fold_${fold}/test_early6PD6HC.tsv" \
    2>&1 | tee -a "${log_file}"

  echo "Training fold ${fold} finished at $(date)" | tee -a "${log_file}"
  # Remove intermediate epoch checkpoints to save space, keep best_model.pt
  rm -f "${checkpoint_dir}"/checkpoint_epoch_*.pt

  echo "Starting testing fold ${fold} at $(date)" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck} \
    --Neurovoz_and_PC_GITA.train_metadata_path="${meta_root}/fold_${fold}/sub_splits/train_earlybalance.tsv" \
    --Neurovoz_and_PC_GITA.validation_metadata_path="${meta_root}/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
    --Neurovoz_and_PC_GITA.test_metadata_path="${meta_root}/fold_${fold}/test_early6PD6HC.tsv" \
    2>&1 | tee -a "${log_file}"

  echo "Testing fold ${fold} finished at $(date)" | tee -a "${log_file}"
done

echo "All 5 folds completed."