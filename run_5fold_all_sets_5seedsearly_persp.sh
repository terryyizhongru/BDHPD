#!/bin/bash

# Run 5-fold training/testing for multiple metadata sets.
# For each metadata set, repeat training 5 times with different initial seeds.
#
# Output checkpoint directory layout:
#   /data/storage1t/projects/early/BDHPD/<meta_dir>/run_<run_idx>/fold_<fold>/
# Example:
#   .../folds_tsv_SENTENCES/run_1/fold_1/

set -euo pipefail

cuda_device=${1:-0}

# Base directory that contains the metadata fold directories (same as existing scripts)
BASE_META_ROOT="./split_5fold/folds_v2.1_early_validation_newcut//"

config_file="configs/config_early_maxf1.yaml"

wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
balance_dataloaders=true
freeze_ssl=false

# 5 repeated runs with different seeds
NUM_RUNS=5
BASE_SEED=42

META_SETS=(
  "folds_tsv_DDK_ANALYSIS_PATAKA"
  "folds_tsv_SUSTAINED-VOWELS_onlyA123"
)

for meta_dir in "${META_SETS[@]}"; do
  full_meta_root="${BASE_META_ROOT}/${meta_dir}"
  meta_log_dir="logs/5run_5fold_early_persp/${meta_dir}"
  mkdir -p "${meta_log_dir}"
  meta_log_file="${meta_log_dir}/${meta_dir}_$(date +%Y%m%d_%H%M%S).log"

  {
    echo "================ Meta set: ${full_meta_root} ================"
    echo "Log file: ${meta_log_file}"

    for run_idx in $(seq 1 ${NUM_RUNS}); do
    # for run_idx in 1; do
      seed=$((BASE_SEED + run_idx - 1))
      echo "------------ Run ${run_idx}/${NUM_RUNS} (seed=${seed}) ------------"

      for fold in 1 2 3 4 5; do
        checkpoint_dir="/data/storage1t/projects/early/BDHPD/early_persp/${meta_dir}/run_${run_idx}/fold_${fold}"

        echo "================ Fold ${fold} (run=${run_idx}, seed=${seed}) ================"
        echo "Starting training fold ${fold} at $(date)"

        CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --config "${config_file}" \
          --training.checkpoint_dir="${checkpoint_dir}" \
          --training.seed="${seed}" \
          --data.wavelets=${wavelets} \
          --training.balance_dataloaders=${balance_dataloaders} \
          --model.freeze_ssl=${freeze_ssl} \
          --training.contrastive_loss.active=${contrastive_loss} \
          --model.use_adain_layers=${adain_layers} \
          --model.use_conv_bottleneck_layer=${conv_bottleneck} \
          --Neurovoz_and_PC_GITA.train_metadata_path="${full_meta_root}/fold_${fold}/sub_splits/train_early_persp.tsv" \
          --Neurovoz_and_PC_GITA.validation_metadata_path="${full_meta_root}/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
          --Neurovoz_and_PC_GITA.test_metadata_path="${full_meta_root}/fold_${fold}/test_early6PD6HC.tsv"

        echo "Training fold ${fold} finished at $(date)"

        # Remove intermediate epoch checkpoints to save space, keep best_model.pt
        rm -f "${checkpoint_dir}"/checkpoint_epoch_*.pt || true

        echo "Starting testing fold ${fold} at $(date)"

        CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config "${config_file}" \
          --training.checkpoint_dir="${checkpoint_dir}" \
          --training.seed="${seed}" \
          --data.wavelets=${wavelets} \
          --model.freeze_ssl=${freeze_ssl} \
          --training.contrastive_loss.active=${contrastive_loss} \
          --model.use_adain_layers=${adain_layers} \
          --model.use_conv_bottleneck_layer=${conv_bottleneck} \
          --Neurovoz_and_PC_GITA.train_metadata_path="${full_meta_root}/fold_${fold}/sub_splits/train_early_persp.tsv" \
          --Neurovoz_and_PC_GITA.validation_metadata_path="${full_meta_root}/fold_${fold}/sub_splits/val_early6PD6HC.tsv" \
          --Neurovoz_and_PC_GITA.test_metadata_path="${full_meta_root}/fold_${fold}/test_early6PD6HC.tsv"

        echo "Testing fold ${fold} finished at $(date)"
      done
    done

    echo "================ Finished meta set: ${full_meta_root} ================"
    echo
  } 2>&1 | tee -a "${meta_log_file}"
done

echo "All meta sets, 5 runs each, completed."
