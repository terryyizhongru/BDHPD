# filepath: /home/yzhong/data/storage2/gits/BDHPD/analyze_neurovoz_pcgita_folds_macro.py
import os
import argparse
import csv

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)


def load_fold_tsv(folder_root, fold_idx, filename="Neurovoz_and_PC_GITA_detailed_results.tsv"):
    fold_dir = os.path.join(folder_root, f"fold_{fold_idx}")
    tsv_path = os.path.join(fold_dir, filename)
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV not found for fold {fold_idx}: {tsv_path}")

    y_true = []
    y_pred = []
    y_score = []  # probability/logit for positive class

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty TSV for fold {fold_idx}: {tsv_path}")

        # Expect columns ... logit, true_label, predicted_label
        try:
            logit_idx = header.index("logit")
            true_idx = header.index("true_label")
            pred_idx = header.index("predicted_label")
        except ValueError:
            raise ValueError(f"TSV header must contain 'logit', 'true_label', 'predicted_label': {tsv_path}")

        for row in reader:
            if not row:
                continue
            try:
                score = float(row[logit_idx]) if row[logit_idx] != "" else None
            except ValueError:
                score = None
            try:
                y_t = int(row[true_idx]) if row[true_idx] != "" else None
                y_p = int(row[pred_idx]) if row[pred_idx] != "" else None
            except ValueError:
                continue

            if y_t is None or y_p is None:
                continue

            y_true.append(y_t)
            y_pred.append(y_p)
            if score is not None:
                y_score.append(score)
            else:
                y_score.append(float(y_p))  # fallback

    return np.array(y_true), np.array(y_pred), np.array(y_score)


def compute_metrics_macro(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    # 使用 macro 平均
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    # Sens/Spec：还是按二分类 confusion matrix 来算
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        sens = float("nan")
        spec = float("nan")

    # AUC：用概率
    try:
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")

    bal_acc = (sens + spec) / 2.0
    return {
        "Acc": acc,
        "F1-Score": f1,
        "Prec": prec,
        "AUC": auc,
        "Sens": sens,
        "Spec": spec,
        "balanced_accuracy": bal_acc,
    }


def format_metrics(m):
    return " ".join(
        [f"{k}: {m[k]:.4f}" for k in ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]]
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-fold Neurovoz+PC-GITA TSVs (macro averaged metrics).")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing fold_1..fold_5 with Neurovoz_and_PC_GITA_detailed_results.tsv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt file path (default: root/neurovoz_pcgita_5fold_analysis_macro.txt)",
    )
    args = parser.parse_args()

    fold_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for fold in range(1, 6):
        y_true, y_pred, y_score = load_fold_tsv(args.root, fold)
        m = compute_metrics_macro(y_true, y_pred, y_score)
        fold_metrics.append(m)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_score.append(y_score)

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_y_score = np.concatenate(all_y_score, axis=0)

    keys = ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]
    avg_metrics = {k: float(np.nanmean([m[k] for m in fold_metrics])) for k in keys}

    merged_metrics = compute_metrics_macro(all_y_true, all_y_pred, all_y_score)

    output_path = args.output or os.path.join(args.root, "neurovoz_pcgita_5fold_analysis_macro.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Per-fold metrics (macro):\n")
        for i, m in enumerate(fold_metrics, start=1):
            f.write(f"Fold {i}: {format_metrics(m)}\n")

        f.write("\nAverage over 5 folds (macro):\n")
        f.write(format_metrics(avg_metrics) + "\n")

        f.write("\nMerged 5 folds overall metrics (macro):\n")
        f.write(format_metrics(merged_metrics) + "\n")

    print(f"Macro analysis written to: {output_path}")

if __name__ == "__main__":
    main()