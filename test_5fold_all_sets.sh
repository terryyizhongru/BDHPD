#!/bin/bash

# Usage: bash run_5fold_all_sets.sh [cuda_device]
# Default GPU is 0 if not provided.

cuda_device=${1:-0}

BASE_META_ROOT="split_5fold/folds_v2_early_validation"

META_SETS=(
  "folds_csv_idonly_v2_tsv_noDDK"
  "folds_csv_idonly_v2_tsv_noVowel"
  "folds_tsv_DDK_ANALYSIS"
  "folds_tsv_DDK_ANALYSIS_PATAKA"
  "folds_tsv_MONOLOGUE"
)


for meta_dir in "${META_SETS[@]}"; do
  full_meta_root="${BASE_META_ROOT}/${meta_dir}"
  echo "================ Running 5-fold with metadata root: ${full_meta_root} ================"
  bash test_5fold.sh "${cuda_device}" "${full_meta_root}" 
  echo "================ Finished 5-fold with metadata root: ${full_meta_root} ================"
  echo
done
