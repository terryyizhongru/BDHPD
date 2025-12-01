import os
import argparse
import csv
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix


def load_fold_tsv(folder_root, fold_idx, filename="Neurovoz_and_PC_GITA_detailed_results.tsv"):
    fold_dir = os.path.join(folder_root, f"fold_{fold_idx}")
    tsv_path = os.path.join(fold_dir, filename)
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV not found for fold {fold_idx}: {tsv_path}")

    y_true = []
    y_pred = []
    y_score = []  # probability/logit for positive class
    ids = []      # subject IDs

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
                # skip if labels are not parseable
                continue

            if y_t is None or y_p is None:
                continue

            y_true.append(y_t)
            y_pred.append(y_p)
            ids.append(row[0])
            if score is not None:
                y_score.append(score)
            else:
                y_score.append(float(y_p))  # fallback

    return np.array(ids), np.array(y_true), np.array(y_pred), np.array(y_score)


def compute_metrics_binary(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    # AUC: need both classes present; otherwise set to NaN
    try:
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")
    sens = recall_score(y_true, y_pred)  # recall for positive class
    # specificity: recall for negative class
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
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
    return " ".join([f"{k}: {m[k]:.4f}" for k in ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]])


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-fold Neurovoz+PC-GITA TSVs.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing fold_1..fold_5 with Neurovoz_and_PC_GITA_detailed_results.tsv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt file path (default: root/neurovoz_pcgita_5fold_analysis.txt)",
    )
    args = parser.parse_args()

    fold_metrics = []            # overall per-fold
    fold_metrics_pcgita = []     # per-fold, PCGITA-only
    fold_metrics_neurovoz = []   # per-fold, NeuroVoz-only
    all_ids = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    # per-dataset (PCGITA vs NeuroVoz) accumulators for merged metrics
    pcgita_ids_all = []
    pcgita_y_true_all = []
    pcgita_y_pred_all = []
    pcgita_y_score_all = []

    neurovoz_ids_all = []
    neurovoz_y_true_all = []
    neurovoz_y_pred_all = []
    neurovoz_y_score_all = []

    # 1. per-fold metrics
    for fold in range(1, 6):
        ids, y_true, y_pred, y_score = load_fold_tsv(args.root, fold)
        # overall fold metrics
        m = compute_metrics_binary(y_true, y_pred, y_score)
        fold_metrics.append(m)
        all_ids.append(ids)
        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_y_score.append(y_score)

        # split this fold into PCGITA vs NeuroVoz by ID prefix
        pc_mask = np.array([str(i).startswith("AVPEPUDE") for i in ids])
        nv_mask = ~pc_mask

        if pc_mask.any():
            y_true_pc = y_true[pc_mask]
            y_pred_pc = y_pred[pc_mask]
            y_score_pc = y_score[pc_mask]
            pcgita_ids_all.append(ids[pc_mask])
            pcgita_y_true_all.append(y_true_pc)
            pcgita_y_pred_all.append(y_pred_pc)
            pcgita_y_score_all.append(y_score_pc)
            # per-fold PCGITA metrics
            fold_metrics_pcgita.append(compute_metrics_binary(y_true_pc, y_pred_pc, y_score_pc))

        if nv_mask.any():
            y_true_nv = y_true[nv_mask]
            y_pred_nv = y_pred[nv_mask]
            y_score_nv = y_score[nv_mask]
            neurovoz_ids_all.append(ids[nv_mask])
            neurovoz_y_true_all.append(y_true_nv)
            neurovoz_y_pred_all.append(y_pred_nv)
            neurovoz_y_score_all.append(y_score_nv)
            # per-fold NeuroVoz metrics
            fold_metrics_neurovoz.append(compute_metrics_binary(y_true_nv, y_pred_nv, y_score_nv))

    all_ids = np.concatenate(all_ids, axis=0)
    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)
    all_y_score = np.concatenate(all_y_score, axis=0)

    # merged per-dataset arrays
    if pcgita_y_true_all:
        pcgita_y_true_merged = np.concatenate(pcgita_y_true_all, axis=0)
        pcgita_y_pred_merged = np.concatenate(pcgita_y_pred_all, axis=0)
        pcgita_y_score_merged = np.concatenate(pcgita_y_score_all, axis=0)
    else:
        pcgita_y_true_merged = pcgita_y_pred_merged = pcgita_y_score_merged = None

    if neurovoz_y_true_all:
        neurovoz_y_true_merged = np.concatenate(neurovoz_y_true_all, axis=0)
        neurovoz_y_pred_merged = np.concatenate(neurovoz_y_pred_all, axis=0)
        neurovoz_y_score_merged = np.concatenate(neurovoz_y_score_all, axis=0)
    else:
        neurovoz_y_true_merged = neurovoz_y_pred_merged = neurovoz_y_score_merged = None

    # 2. average metrics across folds
    keys = ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]
    avg_metrics = {}
    std_metrics = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics]
        avg_metrics[k] = float(np.nanmean(vals))
        std_metrics[k] = float(np.nanstd(vals, ddof=0))

    # dataset-specific average metrics across folds (only if we have metrics for all 5 folds)
    avg_metrics_pcgita = None
    std_metrics_pcgita = None
    if len(fold_metrics_pcgita) == 5:
        avg_metrics_pcgita = {}
        std_metrics_pcgita = {}
        for k in keys:
            vals = [m[k] for m in fold_metrics_pcgita]
            avg_metrics_pcgita[k] = float(np.nanmean(vals))
            std_metrics_pcgita[k] = float(np.nanstd(vals, ddof=0))

    avg_metrics_neurovoz = None
    std_metrics_neurovoz = None
    if len(fold_metrics_neurovoz) == 5:
        avg_metrics_neurovoz = {}
        std_metrics_neurovoz = {}
        for k in keys:
            vals = [m[k] for m in fold_metrics_neurovoz]
            avg_metrics_neurovoz[k] = float(np.nanmean(vals))
            std_metrics_neurovoz[k] = float(np.nanstd(vals, ddof=0))

    # 3. merged TSV metrics (overall)
    merged_metrics = compute_metrics_binary(all_y_true, all_y_pred, all_y_score)

    # merged metrics per dataset (PCGITA / NeuroVoz)
    pcgita_merged_metrics = None
    neurovoz_merged_metrics = None
    if pcgita_y_true_merged is not None:
        pcgita_merged_metrics = compute_metrics_binary(pcgita_y_true_merged, pcgita_y_pred_merged, pcgita_y_score_merged)
    if neurovoz_y_true_merged is not None:
        neurovoz_merged_metrics = compute_metrics_binary(neurovoz_y_true_merged, neurovoz_y_pred_merged, neurovoz_y_score_merged)

    # decide output path: default to same level as fold_* dirs
    output_path = args.output or os.path.join(args.root, "neurovoz_pcgita_5fold_analysis.txt")

    # write to txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Per-fold metrics (Method):\n")
        for i, m in enumerate(fold_metrics, start=1):
            f.write(f"Fold {i}: {format_metrics(m)}\n")

        f.write("\nAverage over 5 folds (Method):\n")
        # overall mean +/- std
        avg_std_line = " ".join(
            [f"{k}: {avg_metrics[k]:.4f} +/- {std_metrics[k]:.4f}" for k in keys]
        )
        f.write(avg_std_line + "\n")
        print("Average over 5 folds (Method):", avg_std_line)

        # dataset-level average over folds
        if avg_metrics_pcgita is not None:
            f.write("\nAverage over 5 folds PCGITA-only (Method):\n")
            line_pc = " ".join(
                [f"{k}: {avg_metrics_pcgita[k]:.4f} +/- {std_metrics_pcgita[k]:.4f}" for k in keys]
            )
            f.write(line_pc + "\n")
            print("Average over 5 folds PCGITA-only (Method):", line_pc)

        if avg_metrics_neurovoz is not None:
            f.write("\nAverage over 5 folds NeuroVoz-only (Method):\n")
            line_nv = " ".join(
                [f"{k}: {avg_metrics_neurovoz[k]:.4f} +/- {std_metrics_neurovoz[k]:.4f}" for k in keys]
            )
            f.write(line_nv + "\n")
            print("Average over 5 folds NeuroVoz-only (Method):", line_nv)

        f.write("\nMerged 5 folds overall metrics (Method):\n")
        f.write(format_metrics(merged_metrics) + "\n")

        # dataset-level merged metrics
        if pcgita_merged_metrics is not None:
            f.write("\nMerged 5 folds PCGITA-only metrics (Method):\n")
            f.write(format_metrics(pcgita_merged_metrics) + "\n")

        if neurovoz_merged_metrics is not None:
            f.write("\nMerged 5 folds NeuroVoz-only metrics (Method):\n")
            f.write(format_metrics(neurovoz_merged_metrics) + "\n")

    print(f"Analysis written to: {output_path}")


if __name__ == "__main__":
    main()
