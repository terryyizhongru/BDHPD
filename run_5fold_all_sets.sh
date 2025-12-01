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
  "folds_tsv_SENTENCES"
  "folds_tsv_SUSTAINED-VOWELS"
)

# META_SETS=(
#   "folds_tsv_SENTENCES"
#   "folds_tsv_SUSTAINED-VOWELS"
# )

for meta_dir in "${META_SETS[@]}"; do
  full_meta_root="${BASE_META_ROOT}/${meta_dir}"
  echo "================ Running 5-fold with metadata root: ${full_meta_root} ================"
  bash run_5fold.sh "${cuda_device}" "${full_meta_root}" > "logs/run_5fold_${meta_dir}_$(date +%Y%m%d_%H%M%S).log" 2>&1
  echo "================ Finished 5-fold with metadata root: ${full_meta_root} ================"
  echo
done
