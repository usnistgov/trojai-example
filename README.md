This repo contains a minimal working example for a submission to the TrojAI leaderboard. This minimal "solution" loads the model file, sample clean data, and conducts fine tuning of the model. You can use this as a base to build your own solution.

Every solution submitted for evaluation must be containerized via Docker.

The container submitted for evaluation must be able to perform mitigation on a single LLM. Mitigation technique containers may or may not be given some amount of clean or poisoned example data. 


## The TrojAIMitigationLLM Class

All mitigations are expected to be subclasses of the TrojAIMitigationLLM class, which returns a mitigated Huggingface (Torch) model. The expected interface is explained below. You can develop with the TrojAIMitigationLLM class by pip installing the `trojai-llm-mitigation-framework` folder, which contains the base class (`trojai-llm-mitigation-framework/trojai_llm_mitigation_round/mitigations/llm_mitigation.py`)

```python
from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig
from datasets import Dataset as HF_Dataset


class TrojAIMitigationLLM:
    """
    This is the primary to abstract a TrojAI mitigation on a given Huggingface LLM model. 
    By default, any extra kwargs passed to init will be stored as a keyword attribute in the class.
    
    You may overwrite __init__ in your implementation, but please call super.__init__(batch_size, fp16)

    The only function required to implement is mitigate_model, which returns a Huggingface model. 
    """
    def __init__(self, batch_size=32, fp16=False, **kwargs):
        self.batch_size = batch_size
        self.max_token_length = kwargs.get('max_token_length', 512)
        self.fp16 = fp16
        for k,v in kwargs.items():
            setattr(self, k, v)

    def mitigate_model(self, model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        raise NotImplementedError

```

## Implementing a Mitigation Technique

You primarily modify 2 of the files within the repo for your submission.

- `example_trojai_llm_mitigation.py` - This file is what contains the entry points for the mitigation technique, which will be output into `--output_dirpath` 
  - `def prepare_mitigation(args)` - this function takes in the commandline args and defined metaparameters and is responsible for constructing your specific defense that is a subclass of the `TrojAIMitigationLLM` class. 
- `metaparameters.yml` - This file mirrors the performer-specific hyperparamters defined in tha ArgParser inside `example_trojai_mitigation.py`. The YAML acts as a base configuration (which must be passed in as `--metaparameters`), and can be optionally overwritten by those same CLI args. As you add or remove specific hyperparameters, you can change the `metaparameters.yml` to mirror needed hyperparamters. 
  - For example, if `metaparameters.yml` defines 
    ```yaml
    hyperparameter1: 42
    hyperparameter2: 0.15
    ... 
    ...
    ```
    And you run the mitigation script with
    ```bash
    python example_trojai_llm_mitigation.py --hyperparmeter1 100 ... ...
    ```
    then the script will run with `hyperparameter1 = 100` (since you overrode it with a CLI arg) and `hyperparameter2 = 0.15` (since it was not defined in the CLI arg, it fallsback to the default in the YAML)

## Container Configuration

Each container must implment the `--mitigate` entry point. `--mitigate` conducts the mitigation on the model weights given some dataset that may or may not have clean and/or poisoned data in it. 

TODO bring back the test entrypoint for inferencing a model

## Container Code

It is recommended to use the existing `example_trojai_llm_mitigation.py` script as boilerplate, and to only change the code inside each function where necessary. 

- `def prepare_mitigation(args)` - Given the YAML/CLI args, construct your specific defense (that is a subclass of TrojaiMitigation) and return it. 
- `def prepare_dataset(dataset_path)` - Given the path to the dataset, construct and prepare the huggingface dataset
- `def prepare_model(model, model_params)` - Given the path to the model or a huggingface model and the model parameters, construct the huggingface AutoModel and return it.
- `def prepare_peft(lora_parameters)` - Give the CLI lora parameters, construct the peft we will pass to your mitigation technique.

## System Requirements

- Linux (tested on Ubuntu 22.04)
- CUDA capable NVIDIA GPU (tested on H100)
- Python >= 3.9 

