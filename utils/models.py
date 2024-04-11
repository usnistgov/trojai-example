# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


from os.path import join

import torch
import json
import os
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(model_filepath: str, torch_dtype:torch.dtype=torch.float16):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to where the model is stored

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """

    conf_filepath = os.path.join(model_filepath, 'reduced-config.json')
    logging.info("Loading config file from: {}".format(conf_filepath))
    with open(conf_filepath, 'r') as fh:
        round_config = json.load(fh)

    logging.info("Loading model from filepath: {}".format(model_filepath))
    # https://huggingface.co/docs/transformers/installation#offline-mode
    if round_config['use_lora']:
        base_model_filepath = os.path.join(model_filepath, 'base-model')
        logging.info("loading the base model (before LORA) from {}".format(base_model_filepath))
        model = AutoModelForCausalLM.from_pretrained(base_model_filepath, trust_remote_code=True, torch_dtype=torch_dtype, local_files_only=True)
        # model = AutoModelForCausalLM.from_pretrained(round_config['model_architecture'], trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch_dtype)

        fine_tuned_model_filepath = os.path.join(model_filepath, 'fine-tuned-model')
        logging.info("loading the LORA adapter onto the base model from {}".format(fine_tuned_model_filepath))
        model.load_adapter(fine_tuned_model_filepath)
    else:
        fine_tuned_model_filepath = os.path.join(model_filepath, 'fine-tuned-model')
        logging.info("Loading full fine tune checkpoint into cpu from {}".format(fine_tuned_model_filepath))
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_filepath, trust_remote_code=True, torch_dtype=torch_dtype, local_files_only=True)
        # model = AutoModelForCausalLM.from_pretrained(fine_tuned_model_filepath, trust_remote_code=True, attn_implementation="flash_attention_2", torch_dtype=torch_dtype)

    model.eval()

    tokenizer_filepath = os.path.join(model_filepath, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_filepath)

    return model, tokenizer


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:

    """

    with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)

