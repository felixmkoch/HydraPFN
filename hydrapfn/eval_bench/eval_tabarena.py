import openml
import pandas as pd
import time
from pathlib import Path

from sklearn.metrics import accuracy_score, roc_auc_score
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabicl import TabICLClassifier
from tabpfn_v1 import TabPFNClassifier

TABARENA_SUITE = 457

def evaluate_tabarena(models,
                      task_types=["binary", "multiclass"],
                      metrics={
                          "binary": ["accuracy", "roc_auc"],
                          "multiclass": ["accuracy", "roc_auc"],
                      },
                      max_samples=None,
                      max_features=None,
                      max_classes=None,
                      lite=False,
                      task_indices=None, # for testing on fewer datasets
                      output_fname="results_tabarena.csv",
):
    df_output = pd.DataFrame()

    suite = openml.study.get_suite(TABARENA_SUITE)
    task_ids = suite.tasks

    for i_task, task_id in enumerate(task_ids):
        if task_indices is not None and i_task not in task_indices:
            print(f"Skipping task {i_task} ({task_id}): not in selected indices.")
            continue

        task = OpenMLTaskWrapper.from_task_id(task_id=task_id)
        task_type = task.problem_type

        if task_type not in task_types:
            print(f"Skipping task {i_task} ({task_id}): wrong task type.")
            continue

        dataset = task.task.get_dataset()
        n_samples = dataset.qualities["NumberOfInstances"]
        n_features = dataset.qualities["NumberOfFeatures"]
        n_classes = dataset.qualities["NumberOfClasses"]

        if max_samples is not None and n_samples > max_samples:
            print(f"Skipping task {i_task} ({task_id}): too many samples.")
            continue
        if max_features is not None and n_features > max_features:
            print(f"Skipping task {i_task} ({task_id}): too many features.")
            continue
        if max_classes is not None and n_classes > max_classes:
            print(f"Skipping task {i_task} ({task_id}): too many classes.")
            continue

        if lite:
            n_repeats = 1
            n_folds = 1
        else:
            n_repeats, n_folds, _ = task.get_split_dimensions()
            if n_samples >= 2500:
               n_repeats = min(n_repeats, 3)

        for repeat in range(n_repeats):
            for fold in range(n_folds):
                print(f"== Task {i_task}, repeat {repeat}, fold {fold}. ==")

                X_train, y_train, X_test, y_test = task.get_train_test_split(
                    repeat=repeat, fold=fold
                )
                
                for model_name in models:
                    model = models[model_name]()

                    fit_time = time.perf_counter()
                    print(f"Fitting {model_name}...")
                    model.fit(X_train, y_train)
                    fit_time = time.perf_counter() - fit_time
                    print(f"Fitted in {fit_time}s.")

                    pred_time = time.perf_counter()
                    print(f"Generating predictions using {model_name}...")
                    y_pred = model.predict(X_test)
                    pred_time = time.perf_counter() - pred_time
                    print(f"Predictions generated in {pred_time}s.")
                    y_score = model.predict_proba(X_test)

                    score = {}
                    if task_type not in metrics:
                        raise Exception("Error: No metrics defined for "
                                        f"{task_type}.")
                    for metric in metrics[task_type]:
                        if metric == "accuracy":
                            score[metric] = accuracy_score(y_true=y_test,
                                                           y_pred=y_pred,
                                                          )
                            print(f"Accuracy: {score[metric]}.")
                        elif metric == "roc_auc":
                            if task_type == "binary":
                                y_score = y_score[:, 1]
                            score[metric] = roc_auc_score(y_true=y_test,
                                                          y_score=y_score,
                                                          average="macro",
                                                          multi_class="ovo",
                                                         )
                            print(f"ROC-AUC: {score[metric]}.")
                        else:
                            raise ValueError("Error: Unsupported metric: "
                                             f"{metric} for {task_type}.")

                    print("--")

                    output_dict = {
                        "i_task": [i_task],
                        "task_id": [task_id],
                        "task_type": [task_type],
                        "n_samples": [n_samples],
                        "n_features": [n_features],
                        "n_classes": [n_classes],
                        "repeat": [repeat],
                        "fold": [fold],
                        "model": [model_name],
                        "fit_time": [fit_time],
                        "pred_time": [pred_time],
                    }
                    for metric in score:
                        output_dict[metric] = [score[metric]]
                    
                    df_output = pd.concat([df_output, pd.DataFrame(output_dict)],
                                          ignore_index=True
                                         )

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / output_fname

    df_output.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}.")


if __name__ == "__main__":

    # some tests

    MODELS = {
        #"HydraPFN": HydraPFN,
        #"TabPFNv2.5": TabPFNClassifier,
        "TabICLv2": TabICLClassifier,
    }

    evaluate_tabarena(models=MODELS,
        lite=True,
        task_indices=[1, 2, 5],
    #    max_samples=10000,
        output_fname="results_tabarena.csv",
    )
