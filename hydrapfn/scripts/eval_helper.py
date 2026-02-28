from hydrapfn.scripts.tabular_evaluation import evaluate
from collections import Counter
from hydrapfn.datasets import load_openml_list
import torch


class EvalHelper:
    def __init__(self):
        # Datasets used for validation and test
        self.test_dids_classification = [973, 1111, 1169, 1596, 40981, 41138, 41142, 41143, 41146, 41147, 41150, 41159, 41161, 41163, 41164, 41165, 41166, 41167, 41168, 41169]
        self.valid_dids_classification = [13, 59, 40710, 43, 1498]

        # OpenML CC-18 splits (filtered by sample/feature/class heuristics)
        self.openml_cc18_dids_small = [11, 14, 15, 16, 18, 22, 23, 29, 31, 37, 50, 54, 188, 458, 469, 1049, 1050, 1063, 1068, 1462, 1464, 1480, 1494, 1510, 6332, 23381, 40966, 40975, 40982, 40994]
        self.openml_cc18_dids_large = [3, 6, 12, 28, 32, 38, 44, 46, 151, 182, 300, 307, 554, 1053, 1067, 1461, 1468, 1475, 1478, 1485, 1486, 1487, 1489, 1497, 1501, 1590, 4134, 4534, 4538, 23517, 40499, 40668, 40670, 40701, 40923, 40927, 40978, 40979, 40983, 40984, 40996, 41027]

        self.datasets_data = {}
        self.limit_dict = {}

        self.EVALUATION_TYPE_FILTERS = {
            "categorical": True,
            "nans": True,
            "multiclass": True,
        }

    def check_datasets_data(self, dids):
        """Ensure datasets for given DIDs are loaded into memory."""
        for did in dids:
            if did not in self.datasets_data:
                self.datasets_data[did] = load_openml_list([did], num_feats=99999, max_samples=999999, max_num_classes=999)[0]

    def limit_dataset(self, ds_name, X, y, categorical_feats, max_classes, max_features):
        """Limit features and classes for evaluation to keep runs consistent."""
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        X = X[:, :max_features]

        y_np = y.numpy()
        top_classes = [cls for cls, _ in Counter(y_np).most_common(max_classes)]
        top_set = set(top_classes)
        mask = torch.tensor([int(v in top_set) for v in y_np], dtype=torch.bool)

        return (ds_name, X[mask], y[mask], categorical_feats, None, None)

    def apply_eval_feat_transformations(self, ds_list, eval_filters):
        """Apply simple feature-level filters: remove categorical cols or rows with NaNs."""
        out = []
        for ds_name, X, y, categorical_feats, a, b in ds_list:
            if not eval_filters.get("categorical", False) and categorical_feats:
                non_cat = [i for i in range(X.shape[1]) if i not in categorical_feats]
                X = X[:, non_cat]

            if eval_filters.get("nans", False):
                nan_mask = torch.isnan(X).any(dim=1)
                X = X[~nan_mask]
                y = y[~nan_mask]

            out.append((ds_name, X, y, categorical_feats, a, b))
        return out

    def make_limit_datasets(self, max_classes, max_features, limit_dids, eval_filters):
        if not eval_filters.get("multiclass", True):
            max_classes = 2

        for did in limit_dids:
            ds_name, X, y, categorical_feats, a, b = self.datasets_data[did][0]
            limited = self.limit_dataset(ds_name, X, y, categorical_feats, max_classes, max_features)
            transformed = self.apply_eval_feat_transformations([limited], eval_filters)
            if transformed:
                self.limit_dict[did] = transformed

    def do_evaluation_custom(self,
                             model,
                             bptt,
                             eval_positions,
                             metric,
                             device,
                             evaluation_type,
                             max_classes=10,
                             max_features=100,
                             split_numbers=[1],
                             single_evaluation_prompt=False,
                             permutation_random=False,
                             permutation_bagging=1,
                             sample_bagging=0,
                             eval_filters={"categorical": True,"nans": True,"multiclass": True},
                             dummy_size=(1000, 100),
                             return_whole_output=False):
        """Run evaluation over a set of datasets or a dummy dataset.

        Returns dict keyed by did (or 'dummy') with lists of per-split results.
        """
        predefined = ["openmlcc18", "openmlcc18_large", "test", "dummy"]

        if evaluation_type == "dummy":
            dummy_dataset = [self._get_dummy_dataset(dummy_size=dummy_size)]
            return {"dummy": [evaluate(dummy_dataset, bptt, eval_positions, metric, model, device, split_number=1, random_premutation=False, single_evaluation_prompt=single_evaluation_prompt)]}

        ds = None
        if evaluation_type == "openmlcc18":
            ds = self.openml_cc18_dids_small
        elif evaluation_type == "openmlcc18_large":
            ds = self.openml_cc18_dids_large
        elif evaluation_type == "test":
            ds = self.test_dids_classification
        elif evaluation_type == "val":
            ds = self.valid_dids_classification
        elif evaluation_type not in predefined:
            ds = [evaluation_type]

        self.check_datasets_data(ds)
        self.make_limit_datasets(max_classes, max_features, ds, eval_filters)

        results = {}
        for did, dataset in self.limit_dict.items():
            results[did] = []
            for split_number in split_numbers:
                out = evaluate(dataset, bptt, eval_positions, metric, model, device, split_number=split_number, random_premutation=permutation_random, single_evaluation_prompt=single_evaluation_prompt)
                results[did].append(out if return_whole_output else out["mean_metric"].item())

        return results

    def _get_dummy_dataset(self, dummy_size=(999, 99)):
        dummy_x = torch.rand(dummy_size[0], dummy_size[1])
        dummy_y = torch.randint(0, 2, (dummy_size[0],))
        return ["dummy_set", dummy_x, dummy_y, [], None, None]

    def get_dids_by_string(self, s):
        if s == "openmlcc18":
            return self.openml_cc18_dids_small
        return None


__all__ = ["EvalHelper"]
