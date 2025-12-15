#!/bin/bash

# Usage: bash run_5fold_all_sets.sh [cuda_device]
# Default GPU is 0 if not provided.

cuda_device=${1:-0}

BASE_META_ROOT="./split_5fold/folds_v2.1_early_validation_newcut//"

META_SETS=(
  "folds_tsv_DDK_ANALYSIS_noPETAKA"
  "folds_tsv_DDK_ANALYSIS_PATAKA"
  "folds_tsv_SUSTAINED-VOWELS_onlyA123"
)


for meta_dir in "${META_SETS[@]}"; do
  full_meta_root="${BASE_META_ROOT}/${meta_dir}"
  echo "================ Running 5-fold with metadata root: ${full_meta_root} ================"
  # Logs for this meta_dir under dedicated subdirectory
  run_log_dir="logs/run_5fold/${meta_dir}"
  mkdir -p "${run_log_dir}"
  # bash run_5fold.sh "${cuda_device}" "${full_meta_root}" > "${run_log_dir}/run_5fold_all${meta_dir}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  bash run_5fold_early.sh "${cuda_device}" "${full_meta_root}" > "${run_log_dir}/run_5fold_persp${meta_dir}_$(date +%Y%m%d_%H%M%S).log" 2>&1

  echo "================ Finished 5-fold with metadata root: ${full_meta_root} ================"
  echo
done
