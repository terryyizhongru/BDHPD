import torch
import numpy as np
from yaml_config_manager import load_config
from tqdm import tqdm
import os
import csv

from model_classes.audio_classification_model import AudioClassificationModel
from additional_classes.checkpoint_manager import CheckpointManager

from utils import get_dataset, get_device, create_model, get_single_dataloader
from utils import get_classification_loss, compute_metrics, save_results_file, save_confusion_matrix


def save_detailed_results_tsv(config, dataset_name, test_metadata_path, audio_paths, logits, true_labels, predictions):
    """Save per-sample logits and predicted labels together with original TSV rows.

    Uses the AUDIOFILE path (second column in TSV) as key to align rows.
    For multi-class, logits are the probability vector; for binary, a single probability.
    """
    if test_metadata_path is None or not os.path.exists(test_metadata_path):
        print(f"[WARN] Cannot save detailed results for {dataset_name}: metadata TSV not found: {test_metadata_path}")
        return

    # Build mapping from AUDIOFILE path to (logit/probabilities, true_label, predicted_label).
    # If there are multiple rows with same path, we'll assign predictions in order.
    path_to_entries = {}
    for path, logit_row, y_true, y_pred in zip(audio_paths, logits, true_labels, predictions):
        if path not in path_to_entries:
            path_to_entries[path] = []
        path_to_entries[path].append((logit_row, int(y_true), int(y_pred)))

    output_tsv = os.path.join(config.training.checkpoint_dir, f"{dataset_name}_detailed_results.tsv")

    with open(test_metadata_path, "r", encoding="utf-8") as fin, \
            open(output_tsv, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")

        header = next(reader, None)
        if header is None:
            print(f"[WARN] Metadata TSV for {dataset_name} has no header: {test_metadata_path}")
            return

        # Extend header with new columns
        # For binary: one logit column; for multi-class: logit_0, logit_1, ...
        num_classes = config.model.num_classes
        if num_classes == 2:
            logit_cols = ["logit"]
        else:
            logit_cols = [f"logit_{i}" for i in range(num_classes)]

        new_header = header + logit_cols + ["true_label", "predicted_label"]
        writer.writerow(new_header)

        # Assume second column in TSV is AUDIOFILE
        for row in reader:
            if not row:
                continue
            audio_path = row[1]
            if audio_path in path_to_entries and path_to_entries[audio_path]:
                logit_row, y_true, y_pred = path_to_entries[audio_path].pop(0)
                if num_classes == 2:
                    # logit_row is shape (1,) or scalar probability
                    if np.ndim(logit_row) == 0:
                        logit_vals = [float(logit_row)]
                    else:
                        logit_vals = [float(logit_row[0])]
                else:
                    logit_vals = [float(x) for x in logit_row]
                out_row = row + [str(v) for v in logit_vals] + [str(y_true), str(y_pred)]
            else:
                # No prediction found; pad with empty fields
                out_row = row + [""] * (len(logit_cols) + 2)
            writer.writerow(out_row)

    print(f"[INFO] Saved detailed results TSV for {dataset_name} to {output_tsv}")


def evaluate_test_set(config, model, dataloader, device, criterions, return_embeddings=False, return_logits_and_paths=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_embeddings = []
    all_sample_types = []
    all_logits = []
    all_paths = []
    
    with torch.no_grad():
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False)
        for i, batch in p_bar:
            # Keep a copy of non-tensor fields (e.g., paths) before moving tensors to device
            raw_paths = None
            if return_logits_and_paths and "audio_path" in batch:
                raw_paths = batch["audio_path"]

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch)
            if config.model.num_classes == 2:
                # outputs = outputs.squeeze(1)
                logits = outputs["logits"].squeeze(1)
                targets = batch["labels"].float()
            else:
                logits = outputs["logits"]
                targets = batch["labels"]
            loss = criterions["classification"](logits, targets)
            running_loss += loss.item()
            
            if config.model.num_classes == 2:
                prob = torch.sigmoid(logits)
                current_predictions = prob.detach().cpu().numpy()
                current_predictions = np.where(current_predictions > 0.5, 1, 0)
            else:
                prob = torch.softmax(logits, dim=-1)
                current_predictions = prob.argmax(dim=-1).cpu().numpy()
            
            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(current_predictions)

            if return_logits_and_paths:
                all_logits.append(prob.detach().cpu().numpy())
                if raw_paths is not None:
                    # assume list[str] or list
                    all_paths.extend(list(raw_paths))
            
            if return_embeddings:
                all_embeddings.append(outputs["embeddings"].cpu().numpy())
                if "sample_type" in batch:
                    all_sample_types.extend(batch["sample_type"].cpu().numpy().tolist())
                else:
                    # Default to speech (0) if sample_type is not provided
                    all_sample_types.extend([0] * len(batch["labels"]))
                    print("Warning: 'sample_type' not found in batch. Defaulting to speech type (0).")
    
    metrics = compute_metrics(all_labels, all_predictions, is_binary_classification=config.model.num_classes == 2)
    metrics["loss"] = running_loss / len(dataloader)

    if return_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        result = (metrics, all_embeddings, all_labels, all_sample_types)
    else:
        result = (metrics,)

    if return_logits_and_paths:
        if all_logits:
            all_logits = np.concatenate(all_logits, axis=0)
        result = result + (all_logits, all_paths, all_predictions)

    if len(result) == 1:
        return result[0]
    return result

def main(config):
    # Load the test datasets based on active configuration
    test_ewadb = None
    test_pcgita = None
    
    if config.ewadb.active:
        test_ewadb = get_dataset(config, "test", "ewadb", domain_id=0)
        print("Test EWADB dataset length: ", len(test_ewadb))
    else:
        print("EWADB dataset is not active, skipping EWADB testing")
    
    if config.pc_gita.active:
        test_pcgita = get_dataset(config, "test", "pc_gita", domain_id=1)
        print("Test PCGITA dataset length: ", len(test_pcgita))
    else:
        print("PC-GITA dataset is not active, skipping PC-GITA testing")
        
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active:
        test_neurovoz_pcgita = get_dataset(config, "test", "Neurovoz_and_PC_GITA", domain_id=1)
        print("Test Neurovoz_and_PC_GITA dataset length: ", len(test_neurovoz_pcgita))
    else:
        print("Neurovoz_and_PC_GITA dataset is not active, skipping Neurovoz_and_PC_GITA testing")
    
    if not config.ewadb.active and not config.pc_gita.active and not (hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active):
        raise ValueError("At least one dataset should be active for testing")
    
    print("Test datasets loaded successfully")

    # Initialize device, model, and dataloader
    device = get_device(config)
    print(f"Using device: {device}")
    
    # set number of domains
    config.model.num_domains = 2
    
    model = create_model(config, device)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        lower_is_better=config.validation.metric_lower_is_better
    )
    
    # Load the best model from checkpoint
    checkpoint_manager.load_best_model()
    print("Loaded best model from checkpoint")
    
    # Create separate test dataloaders for each active dataset
    test_dl_ewadb = None
    test_dl_pcgita = None
    test_dl_neurovoz_pcgita = None

    if config.ewadb.active and test_ewadb is not None:
        test_dl_ewadb = get_single_dataloader(config, test_ewadb, "test")

    if config.pc_gita.active and test_pcgita is not None:
        test_dl_pcgita = get_single_dataloader(config, test_pcgita, "test")
        
    if hasattr(config, 'Neurovoz_and_PC_GITA') and config.Neurovoz_and_PC_GITA.active and test_neurovoz_pcgita is not None:
        test_dl_neurovoz_pcgita = get_single_dataloader(config, test_neurovoz_pcgita, "test")

    criterions = {}
    criterions["classification"] = get_classification_loss(config.model.num_classes)
    # criterions["domain_classification"] = get_classification_loss(config.model.num_domains)
    
    
    # Evaluate the model on the test sets with embeddings and save detailed TSVs
    if test_dl_ewadb is not None:
        test_metrics_ewadb, embeddings_ewadb, labels_ewadb, sample_types_ewadb, logits_ewadb, paths_ewadb, preds_ewadb = evaluate_test_set(
            config, model, test_dl_ewadb, device, criterions, return_embeddings=True, return_logits_and_paths=True
        )
        print(f"[EWADB] Test Metrics:")
        for m in test_metrics_ewadb:
            print(f"Test {m}: {test_metrics_ewadb[m]}")
        
        # Save the results and confusion matrices for EWADB
        save_results_file(config.training.checkpoint_dir, test_metrics_ewadb, prefix="ewadb_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_ewadb["confusion_matrix"], prefix="ewadb_")

        # Save detailed per-sample results TSV if metadata path is available in config
        tsv_path_ewadb = getattr(config.ewadb, "test_metadata_path", None) if hasattr(config, "ewadb") else None
        save_detailed_results_tsv(config, "ewadb", tsv_path_ewadb, paths_ewadb, logits_ewadb, labels_ewadb, preds_ewadb)
    
    if test_dl_pcgita is not None:
        test_metrics_pcgita, embeddings_pcgita, labels_pcgita, sample_types_pcgita, logits_pcgita, paths_pcgita, preds_pcgita = evaluate_test_set(
            config, model, test_dl_pcgita, device, criterions, return_embeddings=True, return_logits_and_paths=True
        )
        print(f"[PCGITA] Test Metrics:")
        for m in test_metrics_pcgita:
            print(f"Test {m}: {test_metrics_pcgita[m]}")
        
        # Save the results and confusion matrices for PC-GITA
        save_results_file(config.training.checkpoint_dir, test_metrics_pcgita, prefix="pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_pcgita["confusion_matrix"], prefix="pcgita_")

        tsv_path_pcgita = getattr(config.pc_gita, "test_metadata_path", None) if hasattr(config, "pc_gita") else None
        save_detailed_results_tsv(config, "pcgita", tsv_path_pcgita, paths_pcgita, logits_pcgita, labels_pcgita, preds_pcgita)
    
    if test_dl_neurovoz_pcgita is not None:
        test_metrics_neurovoz_pcgita, embeddings_neurovoz_pcgita, labels_neurovoz_pcgita, sample_types_neurovoz_pcgita, logits_neurovoz_pcgita, paths_neurovoz_pcgita, preds_neurovoz_pcgita = evaluate_test_set(
            config, model, test_dl_neurovoz_pcgita, device, criterions, return_embeddings=True, return_logits_and_paths=True
        )
        print(f"[Neurovoz_and_PC_GITA] Test Metrics:")
        for m in test_metrics_neurovoz_pcgita:
            print(f"Test {m}: {test_metrics_neurovoz_pcgita[m]}")
        
        # Save the results and confusion matrices for Neurovoz_and_PC_GITA
        save_results_file(config.training.checkpoint_dir, test_metrics_neurovoz_pcgita, prefix="neurovoz_pcgita_")
        save_confusion_matrix(config.training.checkpoint_dir, test_metrics_neurovoz_pcgita["confusion_matrix"], prefix="neurovoz_pcgita_")

        tsv_path_neurovoz_pcgita = getattr(config.Neurovoz_and_PC_GITA, "test_metadata_path", None) if hasattr(config, "Neurovoz_and_PC_GITA") else None
        save_detailed_results_tsv(config, "Neurovoz_and_PC_GITA", tsv_path_neurovoz_pcgita, paths_neurovoz_pcgita, logits_neurovoz_pcgita, labels_neurovoz_pcgita, preds_neurovoz_pcgita)


if __name__ == "__main__":
    config = load_config()
    main(config)
