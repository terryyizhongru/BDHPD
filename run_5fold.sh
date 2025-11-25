#!/bin/bash

# filepath: run_5fold_early.sh
cuda_device=0
config_file=configs/config_early.yaml

wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
balance_dataloaders=false
freeze_ssl=false

for fold in 1 2; do
  checkpoint_dir="/data/storage1t/projects/early/BDHPD/run_fold${fold}"
  # 日志统一放到本项目的 logs 目录下
  mkdir -p logs
  log_file="logs/training_fold${fold}_$(date +%Y%m%d_%H%M%S).log"

  echo "================ Fold ${fold} ================"
  echo "Starting training fold ${fold} at $(date)" | tee -a "${log_file}"

  # 覆盖当前 fold 对应的 metadata 路径
  CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --training.balance_dataloaders=${balance_dataloaders} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck} \
    --Neurovoz_and_PC_GITA.train_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/sub_splits/train.tsv" \
    --Neurovoz_and_PC_GITA.validation_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
    --Neurovoz_and_PC_GITA.test_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/test_early6PD6HC.tsv" \
    2>&1 | tee -a "${log_file}"

  echo "Training fold ${fold} finished at $(date)" | tee -a "${log_file}"
  echo "Starting testing fold ${fold} at $(date)" | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck} \
    --Neurovoz_and_PC_GITA.train_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/sub_splits/train.tsv" \
    --Neurovoz_and_PC_GITA.validation_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
    --Neurovoz_and_PC_GITA.test_metadata_path="split_5fold/folds_v2_early_validation/folds_csv_idonly_v2_tsv_noVowel/fold_${fold}/test_early6PD6HC.tsv" \
    2>&1 | tee -a "${log_file}"

  echo "Testing fold ${fold} finished at $(date)" | tee -a "${log_file}"
done

echo "All 5 folds completed."