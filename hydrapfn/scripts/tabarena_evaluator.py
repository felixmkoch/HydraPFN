from __future__ import annotations

import openml
import torch
import numpy as np
import pandas as pd

from hydrapfn.scripts.hydra_prediction_interface import hydra_predict
from hydrapfn.scripts.tabular_metrics import accuracy_metric
from hydrapfn.scripts.tabular_metrics import auc_metric
from hydrapfn.scripts.tabular_metrics import log_loss_metric

class TabArenaEvaluator:

    def eval_on_tabarena(
            self,
            model,
            num_pcps: int = 1,
            is_lite: bool = False,
            tabarena_version: str = "tabarena-v0.1",
            max_instances: int = 2_000,
            max_classes: int = None,
            metric = auc_metric,
            matric_multiclass = log_loss_metric):
        """
        Code mostly from https://github.com/autogluon/tabarena/blob/main/examples/benchmarking/run_get_tabarena_datasets_from_openml.py
        """

        results = {}        # Dict did -> average error across all splits

        # -- Parameters
        tabarena_version = "tabarena-v0.1"
        """The version of the TabArena benchmark suite to use."""
        tabarena_lite = False
        """If True, will use the TabArena-Lite version of the benchmark suite.
        That is, only the first repeat of the first fold of each task will be used."""

        # -- Get Data
        benchmark_suite = openml.study.get_suite(tabarena_version)
        task_ids = benchmark_suite.tasks

        # we will filter per-task later after sample-size check
        # (binary classification filtering is handled by a private helper)

        # Iterate over all data and outer cross-validation splits from TabArena(-Lite)
        print("Getting Data for TabArena tasks...")
        if tabarena_lite:
            print("TabArena Lite is enabled. Getting first repeat of first fold for each task.")

        for task_id in task_ids:
            results[task_id] = []

            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()
            print(f"Task ID: {task.id}, Dataset ID: {dataset.id}, Dataset Name: {dataset.name}")

            n_samples = dataset.qualities["NumberOfInstances"]
            if n_samples > max_instances:
                print(f"  Skipping task {task_id} due to number of instances ({n_samples}) exceeding max_instances ({max_instances})")
                continue

            # binary classification filter, applied after sample-size
            if max_classes and not self._is_in_max_classes(dataset, max_classes):
                print(f"  Skipping task {task_id} because it is not in the specified number of classes")
                continue

            # Get the number of folds and repeats used in TabArena
            if tabarena_lite:
                folds = 1
                tabarena_repeats = 1
            else:
                _, folds, _ = task.get_split_dimensions()
                if n_samples < 2_500:
                    tabarena_repeats = 10
                elif n_samples > 250_000:
                    tabarena_repeats = 1
                else:
                    tabarena_repeats = 3
            print(f"TabArena Repeats: {tabarena_repeats} | Folds: {folds}")

            # Load the data for each split
            for repeat in range(tabarena_repeats):
                for fold in range(folds):
                    # get dataframe output and convert manually (future-proof)
                    X_df, y_series, categorical_indicator, attribute_names = dataset.get_data(
                        target=task.target_name, dataset_format="dataframe"
                    )
                    print(f"Dataset {dataset.name} indicator: {categorical_indicator}")

                    train_indices, test_indices = task.get_train_test_split_indices(fold=fold, repeat=repeat)

                    # skip if one of the splits is empty
                    if len(train_indices) == 0 or len(test_indices) == 0:
                        print(f"  Skipping fold {fold}, repeat {repeat} due to empty split")
                        continue

                    # convert features to numeric arrays
                    X_vals = np.zeros((X_df.shape[0], X_df.shape[1]), dtype=np.float32)
                    for col_idx in range(X_df.shape[1]):
                        col = X_df.iloc[:, col_idx]
                        if categorical_indicator[col_idx]:
                            X_vals[:, col_idx] = pd.Categorical(col).codes.astype(np.float32)
                        else:
                            X_vals[:, col_idx] = pd.to_numeric(col, errors='coerce').astype(np.float32)

                    # convert target to numeric
                    if y_series.dtype == object or y_series.dtype.name == 'category':
                        y_vals = pd.Categorical(y_series).codes.astype(np.float32)
                    else:
                        y_vals = pd.to_numeric(y_series, errors='coerce').astype(np.float32)

                    X_train_np = X_vals[train_indices]
                    y_train_np = y_vals[train_indices]
                    X_test_np  = X_vals[test_indices]
                    y_test_np  = y_vals[test_indices]
                    X_combined = np.vstack([X_train_np, X_test_np])
                    y_combined = np.hstack([y_train_np, y_test_np])
                    
                    # Convert to torch tensors and add batch dimension
                    # Format: (num_samples, batch_dim=1, num_features)
                    eval_xs = torch.from_numpy(X_combined).float().unsqueeze(1)
                    eval_ys = torch.from_numpy(y_combined).float().unsqueeze(1)
                    
                    # eval_position is the split between train and test
                    eval_position = len(y_train_np)
                    
                    # Prepare categorical features list (indices where feature is categorical)
                    categorical_feats = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
                    
                    # Call hydra_predict
                    try:
                        outputs, inference_time = hydra_predict(
                            model,
                            eval_xs,
                            eval_ys,
                            eval_position,
                            device='cuda',
                            categorical_feats=categorical_feats,
                            inference_mode=True,
                            extend_features=True,
                            num_pcps=num_pcps
                        )
                        
                        y_test_true = eval_ys[eval_position:, 0]

                        # remove batch dimension - its always of length 0.
                        test_outputs = outputs[0]

                        # For multiclass classification the log loss is used in TabArenav0.1
                        if len(torch.unique(y_test_true)) > 2:
                            metric_used = matric_multiclass
                        else:
                            metric_used = metric

                        accuracy = metric_used(
                            y_test_true.cpu(),
                            test_outputs.cpu()
                        )

                        results[task_id].append({
                            'fold': fold,
                            'repeat': repeat,
                            'accuracy': float(accuracy.cpu().numpy()) if torch.is_tensor(accuracy) else float(accuracy),
                            'inference_time': inference_time
                        })
                        print(f"  Fold {fold}, Repeat {repeat}: Accuracy = {results[task_id][-1]['accuracy']:.4f}")
                    except Exception as e:
                        print(f"  Error evaluating fold {fold}, repeat {repeat}: {e}")
                        continue
        
        # Aggregate results across all tasks and folds
        aggregated_results = {}
        for task_id in results:
            if results[task_id]:
                accuracies = [r['accuracy'] for r in results[task_id]]
                inference_times = [r['inference_time'] for r in results[task_id]]
                aggregated_results[task_id] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_inference_time': np.mean(inference_times),
                    'num_splits': len(accuracies)
                }
                print("-" * 42)
                print(f"Task {task_id}: Mean Accuracy = {aggregated_results[task_id]['mean_accuracy']:.4f} ± {aggregated_results[task_id]['std_accuracy']:.4f}")
                print("-" * 42)

        # Compute overall average accuracy across all tasks
        all_accuracies = [aggregated_results[tid]['mean_accuracy'] for tid in aggregated_results]
        overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0.0
        
        print(f"\n=== Overall Results ===")
        print(f"Overall Mean Accuracy: {overall_accuracy:.4f}")
        print(f"Num Tasks Evaluated: {len(aggregated_results)}")
        
        return aggregated_results

    # Helper to look whether a openML dataset is a binary classification task.
    def _is_in_max_classes(self, dataset, max_classes) -> bool:
        y = dataset.get_data(target=dataset.default_target_attribute)[1]
        return len(np.unique(y)) <= max_classes

