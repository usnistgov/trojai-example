import json
import logging
import pickle
from collections import OrderedDict
from os import listdir, makedirs
from os.path import join, exists

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, pad_to_target
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map
from utils.padding import create_models_padding
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)


class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))
        super().__init__(metaparameters["automatic_training"])

        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(
            self.learned_parameters_dirpath, "models_padding_dict.bin"
        )
        self.model_layer_map_filepath = join(
            self.learned_parameters_dirpath, "model_layer_map.bin"
        )
        self.layer_transform_filepath = join(
            self.learned_parameters_dirpath, "layer_transform.bin"
        )

        # TODO: Update skew parameters per round
        self.model_skew = {
            "TBD": metaparameters["infer_model_skew_TBD"],
        }
        self.normalize = metaparameters["infer_normalize_features"]

        self.configure_fn = self.configure
        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    @staticmethod
    def _load_model(model_filepath: str) -> (dict, str):
        """Load a model given a specific model_path.

        Args:
            model_filepath: str - Path to model.pt file

        Returns:
            model, dict, str - Torch model + dictionary representation of the model + model class name
        """
        model = torch.load(model_filepath)
        model_class = model.__class__.__name__
        model_repr = OrderedDict(
            {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
        )

        return model, model_repr, model_class

    @staticmethod
    def _pad_model(model_dict: dict, model_class: str, models_padding_dict: dict) -> dict:
        """Ensure every layer is correctly padded, so that every model has the same
        number of weights no matter the number of classes.

        Args:
            model_dict: dict - Dictionary representation of the model
            model_class: str - Model class name
            models_padding_dict: dict - Paddings for the round's model classes

        Returns:
            dict - The padded dictionary
        """
        for (layer, target_padding) in models_padding_dict[model_class].items():
            model_dict[layer] = pad_to_target(model_dict[layer], target_padding)

        return model_dict

    @staticmethod
    def _load_ground_truth(model_dirpath: str):
        """Returns the ground truth for a given model.

        Args:
            model_dirpath: str -

        Returns:

        """

        with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
            model_ground_truth = fp.readlines()[0]

        return int(model_ground_truth)

    def _load_models_dirpath(self, models_dirpath):
        model_repr_dict = {}
        model_ground_truth_dict = {}

        for model_path in tqdm(models_dirpath):
            model, model_repr, model_class = self._load_model(
                join(model_path, "model.pt")
            )
            model_ground_truth = self._load_ground_truth(model_path)

            # Build the list of models
            if model_class not in model_repr_dict.keys():
                model_repr_dict[model_class] = []
                model_ground_truth_dict[model_class] = []

            model_repr_dict[model_class].append(model_repr)
            model_ground_truth_dict[model_class].append(model_ground_truth)

        return model_repr_dict, model_ground_truth_dict

    @staticmethod
    def _flatten_models(model_repr_dict, model_layer_map):
        """Flatten a list of models

        Args:
            model_repr_dict:
            model_layer_map:

        Returns:
        """
        flat_models = {}

        for _ in range(len(model_repr_dict)):
            (model_arch, models) = model_repr_dict.popitem()
            if model_arch not in flat_models.keys():
                flat_models[model_arch] = []

            logging.info("Flattenning %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                flat_models[model_arch].append(
                    flatten_model(model, model_layer_map[model_arch])
                )

        return flat_models

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted(
            [join(models_dirpath, model) for model in listdir(models_dirpath)]
        )
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = self._load_models_dirpath(
            model_path_list
        )

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = self._pad_model(
                    model_repr, model_class, models_padding_dict
                )

        check_models_consistency(model_repr_dict)

        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = self._flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(
            flat_models, self.weight_table_params, self.input_features
        )

        logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model
                )
                if X is None:
                    X = model_feats
                    continue

                if model_arch in self.model_skew:
                    X = np.vstack((X, model_feats)) * self.model_skew[model_arch]
                else:
                    X = np.vstack((X, model_feats))

        logging.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(
            **self.random_forest_kwargs,
            random_state=0,
        )
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        logging.info("Configuration done!")

    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict wether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """
        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        # List all available model and limit to the number provided
        model_path_list = sorted(
            [
                join(round_training_dataset_dirpath, model)
                for model in listdir(round_training_dataset_dirpath)
            ]
        )
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, _ = self._load_models_dirpath(
            model_path_list
        )
        logging.info("Loaded models. Flattenning...")

        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = self._pad_model(
                    model_repr, model_class, models_padding_dict
                )

        # Flatten model
        flat_models = self._flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(
            flat_models, self.weight_table_params, self.input_features
        )

        # TODO implement per round inferencing examples.
        model, model_repr, model_class = self._load_model(model_filepath)
        model_repr = self._pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])

        X = (
            use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
        )

        with open(self.model_filepath, "rb") as fp:
            regressor: RandomForestRegressor = pickle.load(fp)

        with open(result_filepath, "w") as fp:
            fp.write(str(regressor.predict(X)[0]))

        logging.info("Inference done!")
