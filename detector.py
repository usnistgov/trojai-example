# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import json
import logging
import os
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch

from utils.abstract import AbstractDetector
from utils.models import load_model



class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath

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

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

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
        os.makedirs(self.learned_parameters_dirpath, exist_ok=True)

        # List all available model
        model_path_list = sorted([os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)])
        logging.info("Found {} models to configure the detector against".format(len(model_path_list)))

        logging.info("Creating detector features")
        X = list()
        y = list()

        for model_index in range(len(model_path_list)):
            model_feats = np.random.randn(100)

            X.append(model_feats)  # random features
            y.append(float(np.random.rand() > 0.5))  # random label

        X = np.stack(X, axis=0)
        y = np.asarray(y)

        logging.info("Training RandomForestRegressor model...")
        model = RandomForestRegressor(**self.random_forest_kwargs, random_state=0)
        model.fit(X, y)

        logging.info("Saving RandomForestRegressor model...")
        with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "wb") as fp:
            pickle.dump(model, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model, tokenizer, torch_dtype=torch.float16, stream_flag=False):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            tokenizer: the models tokenizer
            torch_dtype: the dtype to use for inference
            stream_flag: flag controlling whether to put the whole model on the gpu (stream=False) or whether to park some of the weights on the CPU and stream the activations between CPU and GPU as required. Use stream=False unless you cannot fit the model into GPU memory.
        """

        if stream_flag:
            logging.info("Using accelerate.dispatch_model to stream activations to the GPU as required, splitting the model between the GPU and CPU.")
            model.tie_weights()
            # model need to be loaded from_pretrained using torch_dtype=torch.float16 to fast inference, but the model appears to be saved as fp32. How will this play with bfp16?
            # You can't load as 'auto' and then specify torch.float16 later.
            # In fact, if you load as torch.float16, the later dtype can be None, and it works right

            # The following functions are duplicated from accelerate.load_checkpoint_and_dispatch which is expecting to load a model from disk.
            # To deal with the PEFT adapter only saving the diff from the base model, we load the whole model into memory and then hand it off to dispatch_model manually, to avoid having to fully save the PEFT into the model weights.
            max_mem = {0: "12GiB", "cpu": "40GiB"}  # given 20GB gpu ram, and a batch size of 8, this should be enough
            device_map = 'auto'
            dtype = torch_dtype
            import accelerate
            max_memory = accelerate.utils.modeling.get_balanced_memory(
                model,
                max_memory=max_mem,
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=dtype,
                low_zero=(device_map == "balanced_low_0"),
            )
            device_map = accelerate.infer_auto_device_map(
                model, max_memory=max_memory, no_split_module_classes=["LlamaDecoderLayer"], dtype=dtype
            )

            model = accelerate.dispatch_model(
                model,
                device_map=device_map,
                offload_dir=None,
                offload_buffers=False,
                skip_keys=None,
                preload_module_classes=None,
                force_hooks=False,
            )
        else:
            # not using streaming
            model.cuda()

        prompt = "As someone who uses quality Premium, I"
        inputs = tokenizer([prompt], return_tensors='pt')
        inputs = inputs.to('cuda')

        outputs = model.generate(**inputs, max_new_tokens=200,
                                 pad_token_id=tokenizer.eos_token_id,
                                 top_p=1.0,
                                 temperature=1.0,
                                 no_repeat_ngram_size=3,
                                 do_sample=False)

        results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        result = results[0]  # unpack implicit batch
        result = result.replace(prompt, '')

        logging.info("Prompt: \n\"\"\"\n{}\n\"\"\"".format(prompt))
        logging.info("Response: \n\"\"\"\n{}\n\"\"\"".format(result))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        model, tokenizer = load_model(model_filepath)

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        self.inference_on_example_data(model, tokenizer, torch_dtype=torch.float16, stream_flag=False)

        try:
            # load "trojan" detection model
            with open(os.path.join(self.learned_parameters_dirpath, 'model.bin'), "rb") as fp:
                regressor: RandomForestRegressor = pickle.load(fp)

            # create RNG "features" about the AI model to feed into the "trojan" detector forest
            X = np.random.randn(1, 100)  # needs to be 2D, with the features in dim[-1]

            probability = str(regressor.predict(X)[0])
            logging.info("Random forest regressor predicted correctly")
        except Exception as e:
            logging.info('Failed to run regressor, there may have an issue during fitting, using random for trojan probability: {}'.format(e))
            probability = str(np.random.rand())
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
