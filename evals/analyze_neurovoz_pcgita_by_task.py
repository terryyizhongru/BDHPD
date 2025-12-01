import os
import argparse
import csv

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, roc_auc_score, recall_score, confusion_matrix


TASK_KEYS = [
    "_DDK_ANALYSIS_",
    "_MONOLOGUECUT_",
    "_SENTENCES_",
    "_SUSTAINED-VOWELS_",
]


def detect_task(audio_path: str) -> str | None:
    for key in TASK_KEYS:
        if key in audio_path:
            return key.strip("_")  # e.g. _DDK_ANALYSIS_ -> DDK_ANALYSIS
    return None


def load_fold_tsv_by_task(folder_root, fold_idx, filename="Neurovoz_and_PC_GITA_detailed_results.tsv"):
    """Load one fold TSV and group samples by task name.

    Returns: dict[task_name] -> (ids, y_true, y_pred, y_score)
    """
    fold_dir = os.path.join(folder_root, f"fold_{fold_idx}")
    tsv_path = os.path.join(fold_dir, filename)
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV not found for fold {fold_idx}: {tsv_path}")

    task_data = {}

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty TSV for fold {fold_idx}: {tsv_path}")

        try:
            id_idx = 0
            audio_idx = header.index("AUDIOFILE")
            logit_idx = header.index("logit")
            true_idx = header.index("true_label")
            pred_idx = header.index("predicted_label")
        except ValueError as e:
            raise ValueError(f"TSV header missing expected columns in {tsv_path}: {e}")

        for row in reader:
            if not row:
                continue
            audio_path = row[audio_idx]
            task = detect_task(audio_path)
            if task is None:
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

            if task not in task_data:
                task_data[task] = {
                    "ids": [],
                    "y_true": [],
                    "y_pred": [],
                    "y_score": [],
                }

            td = task_data[task]
            td["ids"].append(row[id_idx])
            td["y_true"].append(y_t)
            td["y_pred"].append(y_p)
            td["y_score"].append(score if score is not None else float(y_p))

    # convert lists to numpy arrays
    for task, td in task_data.items():
        td["ids"] = np.array(td["ids"])
        td["y_true"] = np.array(td["y_true"])
        td["y_pred"] = np.array(td["y_pred"])
        td["y_score"] = np.array(td["y_score"])

    return task_data


def compute_metrics_binary(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    try:
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = float("nan")
    except ValueError:
        auc = float("nan")
    sens = recall_score(y_true, y_pred)
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
    keys = ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]
    return " ".join([f"{k}: {m[k]:.4f}" for k in keys])


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-fold Neurovoz+PC-GITA TSVs by task type.")
    parser.add_argument(
        "--root",
        required=True,
        help="Root directory containing fold_1..fold_5 with Neurovoz_and_PC_GITA_detailed_results.tsv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output txt file path (default: root/neurovoz_pcgita_by_task_5fold_analysis.txt)",
    )
    args = parser.parse_args()

    # per-task: fold_metrics[task] -> list of metrics per fold (overall)
    fold_metrics = {task: [] for task in TASK_KEYS}
    # per-task, per-dataset metrics
    fold_metrics_pcgita = {task: [] for task in TASK_KEYS}
    fold_metrics_neurovoz = {task: [] for task in TASK_KEYS}

    # merged arrays across all folds per task (overall)
    merged_y_true = {task: [] for task in TASK_KEYS}
    merged_y_pred = {task: [] for task in TASK_KEYS}
    merged_y_score = {task: [] for task in TASK_KEYS}

    for fold in range(1, 6):
        task_data = load_fold_tsv_by_task(args.root, fold)
        for task_key in TASK_KEYS:
            task_name = task_key.strip("_")
            if task_name not in task_data:
                continue
            td = task_data[task_name]
            ids = td["ids"]
            y_true = td["y_true"]
            y_pred = td["y_pred"]
            y_score = td["y_score"]

            # overall per-fold metrics for this task
            m = compute_metrics_binary(y_true, y_pred, y_score)
            fold_metrics[task_key].append(m)

            merged_y_true[task_key].append(y_true)
            merged_y_pred[task_key].append(y_pred)
            merged_y_score[task_key].append(y_score)

            # split by dataset based on ID prefix
            pc_mask = np.array([str(i).startswith("AVPEPUDE") for i in ids])
            nv_mask = ~pc_mask

            if pc_mask.any():
                y_true_pc = y_true[pc_mask]
                y_pred_pc = y_pred[pc_mask]
                y_score_pc = y_score[pc_mask]
                fold_metrics_pcgita[task_key].append(
                    compute_metrics_binary(y_true_pc, y_pred_pc, y_score_pc)
                )

            if nv_mask.any():
                y_true_nv = y_true[nv_mask]
                y_pred_nv = y_pred[nv_mask]
                y_score_nv = y_score[nv_mask]
                fold_metrics_neurovoz[task_key].append(
                    compute_metrics_binary(y_true_nv, y_pred_nv, y_score_nv)
                )

    keys = ["Acc", "F1-Score", "Prec", "AUC", "Sens", "Spec", "balanced_accuracy"]

    output_path = args.output or os.path.join(args.root, "neurovoz_pcgita_by_task_5fold_analysis.txt")

    with open(output_path, "w", encoding="utf-8") as f:
        for task_key in TASK_KEYS:
            task_name = task_key.strip("_")
            if not fold_metrics[task_key]:
                continue

            f.write(f"===== Task: {task_name} =====\n")

            # 1) per-fold metrics
            f.write("Per-fold metrics (Method):\n")
            for i, m in enumerate(fold_metrics[task_key], start=1):
                f.write(f"Fold {i}: {format_metrics(m)}\n")

            # 2) average over folds (mean +/- std) overall
            avg_metrics = {}
            std_metrics = {}
            for k in keys:
                vals = [m[k] for m in fold_metrics[task_key]]
                avg_metrics[k] = float(np.nanmean(vals))
                std_metrics[k] = float(np.nanstd(vals, ddof=0))

            f.write("\nAverage over 5 folds (Method):\n")
            avg_std_line = " ".join(
                [f"{k}: {avg_metrics[k]:.4f} +/- {std_metrics[k]:.4f}" for k in keys]
            )
            f.write(avg_std_line + "\n")
            print(f"[Task {task_name}] Average over 5 folds:", avg_std_line)

            # dataset-specific averages for this task (if we have all 5 folds)
            if len(fold_metrics_pcgita[task_key]) == 5:
                avg_pc = {}
                std_pc = {}
                for k in keys:
                    vals = [m[k] for m in fold_metrics_pcgita[task_key]]
                    avg_pc[k] = float(np.nanmean(vals))
                    std_pc[k] = float(np.nanstd(vals, ddof=0))
                line_pc = " ".join(
                    [f"{k}: {avg_pc[k]:.4f} +/- {std_pc[k]:.4f}" for k in keys]
                )
                f.write("Average over 5 folds PCGITA-only (Method):\n")
                f.write(line_pc + "\n")
                print(f"[Task {task_name}] Average over 5 folds PCGITA-only:", line_pc)

            if len(fold_metrics_neurovoz[task_key]) == 5:
                avg_nv = {}
                std_nv = {}
                for k in keys:
                    vals = [m[k] for m in fold_metrics_neurovoz[task_key]]
                    avg_nv[k] = float(np.nanmean(vals))
                    std_nv[k] = float(np.nanstd(vals, ddof=0))
                line_nv = " ".join(
                    [f"{k}: {avg_nv[k]:.4f} +/- {std_nv[k]:.4f}" for k in keys]
                )
                f.write("Average over 5 folds NeuroVoz-only (Method):\n")
                f.write(line_nv + "\n")
                print(f"[Task {task_name}] Average over 5 folds NeuroVoz-only:", line_nv)

            # 3) merged 5 folds metrics for this task (overall)
            y_true_merged = np.concatenate(merged_y_true[task_key], axis=0)
            y_pred_merged = np.concatenate(merged_y_pred[task_key], axis=0)
            y_score_merged = np.concatenate(merged_y_score[task_key], axis=0)
            merged_metrics = compute_metrics_binary(y_true_merged, y_pred_merged, y_score_merged)

            f.write("\nMerged 5 folds overall metrics (Method):\n")
            f.write(format_metrics(merged_metrics) + "\n\n")

    print(f"Task-wise analysis written to: {output_path}")


if __name__ == "__main__":
    main()
