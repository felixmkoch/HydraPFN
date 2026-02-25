from hydrapfn.scripts import tabular_metrics
import os
from hydrapfn.scripts.eval_helper import EvalHelper
from scipy import stats

from hydrapfn.scripts.model_loader import load_hydrapfn_model

import pandas as pd

# Choose dataset or single DID
EVALUATION_TYPE = "openmlcc18"
# EVALUATION_TYPE = 14

# Keep these feature filters enabled/disabled
EVALUATION_TYPE_FILTERS = {
    "categorical": True,
    "nans": True,
    "multiclass": True,
}

EVALUATION_METHODS = ["hydra"]

METRIC_USED = tabular_metrics.auc_metric
RESULT_CSV_SAVE_DIR = os.path.join("result_csvs", "test.csv")
MODEL_PATH = "hydrapfn/trained_models/test_model.cpkt"
SPLIT_NUMBERS = [1, 2, 3, 4, 5]

bptt_here = 1024
CONFIDENCE_LEVEL = 0.95
eval_positions = [972]
device = "cuda:0"

eval_helper = EvalHelper()


def calc_moe(data):
    sem = stats.sem(data)
    degrees_of_freedom = len(data) - 1
    t_score = stats.t.ppf((1 + CONFIDENCE_LEVEL) / 2, degrees_of_freedom)
    return t_score * sem


def do_evaluation(eval_list):
    result_dict = {}
    if "hydra" in eval_list:
        hydra_model, optimizer, hydra_config = load_hydrapfn_model(MODEL_PATH)
        result_dict["hydra"] = eval_helper.do_evaluation_custom(hydra_model, bptt=bptt_here, eval_positions=eval_positions, metric=METRIC_USED, device=device,
                                                                 evaluation_type=EVALUATION_TYPE, split_numbers=SPLIT_NUMBERS, eval_filters=EVALUATION_TYPE_FILTERS)
    return result_dict


def main():
    result_dict = do_evaluation(EVALUATION_METHODS)
    header = ["did"] + EVALUATION_METHODS
    result_arr = []

    # Calc Mean and Confidence Intervals
    for method in EVALUATION_METHODS:
        split_means = []
        for split in range(len(SPLIT_NUMBERS)):
            vals = result_dict[method].values()
            split_errs = [x[split] for x in vals]
            split_means.append(sum(split_errs) / len(split_errs))

        print(f"{method} Stats:")
        print(f"Split Means: {split_means}")
        print(f"Mean Overall: {sum(split_means) / len(split_means)}")
        print(f"MOE: {calc_moe(split_means)}")

    keys = list(result_dict[list(result_dict.keys())[0]].keys())
    for key in keys:
        to_add = [key]
        for method in EVALUATION_METHODS:
            res = result_dict[method][key]
            to_add.append(sum(res) / len(res))
        result_arr.append(to_add)

    df_out = pd.DataFrame(result_arr, columns=header)
    os.makedirs(os.path.dirname(RESULT_CSV_SAVE_DIR), exist_ok=True)
    df_out.to_csv(RESULT_CSV_SAVE_DIR)
    print("worked")


if __name__ == "__main__":
    main()
