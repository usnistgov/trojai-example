This repo contains a minimal working example for a submission to the TrojAI leaderboard. This minimal "solution" loads the model file, sample clean data, and conducts fine tuning of the model. You can use this as a base to build your own solution.

Every solution submitted for evaluation must be containerized via Docker.

The container submitted for evaluation must be able to perform both mitigation and testing on a single AI model. Mitigation technique containers may or may not be given some amount of clean or poisoned example data. During test time, test data will be run through the pre and post process transforms for a given defense, if implemented. 


## The TrojaiMitigation Class

All mitigations are expected to be subclasses of the TrojaiMitigation class, which returns a TrojAIMitigatedModel. The expected interface is explained below. You can develop with the TrojAIMitigation class by pip installing the `trojai-mitigation-round-framework` folder, which contains the base class (`trojai-mitigation-round-framework/trojai_mitigation_round/mitigations/mitigation.py`)

```python
from trojai_mitigation_round.mitigations.mitigation_base import TrojAIMitigation, TrojAIMitigatedModel

class TrojAIMitigation:
    """This is the main class to abstract for a TrojAI Mitigation. 
    By default, any extra keyword arguments passed to __init__ will be stored as keyword attributes in the class. i.e. if you declare:
    mitigation = TrojAIMitigation(device, batch_size, num_workers, kwarg1=1, kwarg2=2), you will have access to mitigation.kwarg1 and mitigation.kwarg2
    
    You may overwrite __init__ but please call super.__init__(device, batch_size, num_workers) 
    
    The only function required to implement is mitigate_model. This function must return a TrojAIMitigatedModel. 

    If a mitigation technique does not do any data pre or post processing at test time, but just changes the model weights, simply wrap your new state dict in this class:
        new_model = TrojAIMitigatedModel(new_state_dict)
        
    If your mitigation technique does require pre and post processing, use this class and overwrite the preprocess_data and postprocess_data   
    """
    def __init__(self, device, batch_size=32, num_workers=1, **kwargs):
        """
        Args
            device: Which device for the mitigation technique use. Either 'cuda:X' or 'cpu'
            batch_size: The batch size for the technique to use
            num_workers: the number of CPU processes to use to load data
        """
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        for k,v in kwargs.items():
            setattr(self, k, v)

    def preprocess_transform(self, x: torch.tensor) -> tuple[torch.tensor, dict]:
        """
        This is the default preprocess of a mitigation method. If your  mitigation technique defines a preprocess transform,
        this function will be overwritten by that. Otherwise, this default is used. 
        
        You may change the batch size to allow for data augmentations being queries as multiple 
        points. Additionally, any additional information required to post process the results back to one prediction per original data point acn be passed as an additional dictionary

        Args
            x: torch:tensor shape: (batch_size, channels, height, width) corresponding to input data of size . Will be on device: 'cpu'

        Returns:
            x: torch tensor corresponding to input data of size (intermediary_batch_size, channels, height, width). 
            info: dictionary corresponding to metadata that needs to be passed to post process
    
        """
        return x, {}

    def postprocess_transform(self, logits: torch.tensor, info: dict) -> torch.tensor:
        """
        This is the default postprocess of a mitigation method. If your mitigation technique defines a postprocess transform, this
        function will be overwritten by that. Otherwise, this default is used.

        If you abstracted preprocess_data to change the batch_size to the model, you must abstract this function to return the logits to the same batch size as
        the original queried model

        Args 
            logits: torch:tensor shape: (intermediary_batch_size, num_classes) corresponding to the logits of the inpu data
            info: dictionary corresponding to metadata passed from preprocess

        Returns:
            logits : torch:tensor shape: (batch_size, num_classes) corresponding to the logits for original batch of data
        """
        return logits 

    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Conducts mitigation that may modify the model weights of the model parameter. This returns a TrojAIMitigatedModel object that contains the state_dict of the newly mitigated model.

        Args:
            model: the model to repair
            dataset: a dataset that contains sample data that may or may not be clean/poisoned.
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        raise NotImplementedError
```

## Implementing a Mitigation Technique

You primarily modify 2 of the files within the repo for your submission.

- `example_trojai_mitigation.py` - This file is what contains the entry points for the mitigation technique. It implements a `mitigate` and `test` entrypoint. 
  - `mitigate` applies your mitigation technique, modifying the model weights, and saves it to the `--output_dirpath` under the name `--model_output_name`. 
  - Customizable options are specified in the metaparameters.json file, described using the metaparameters_schema.json (for example, `optimizer_class`, `loss_class`, etc) that can be changed and overwritten with your specific hyperparameters if needed. 
  - `def prepare_mitigation(args, config_json)` - this function takes in the commandline args and defined metaparameters and is responsible for constructing your specific defense that is a subclass of the `TrojAIMitigation` class.   
  - `def mitigate_model(model, mitigation, dataset, output_dir, output_name)` generates a new mitigated model and saves that model to be used for the phase of execution, testing.

## Container Configuration

Each container must implement two entry points, `mitigate` and `test`. Your mitigation technique is fully constructed both times. If your mitigation or pre/post process transforms require a lengthy construction process, it is recommended you only conditionally set them up individually when they are called rather than at construction time.

- `mitigate` conducts the mitigation on the model weights given some dataset that may or may not have clean and/or poisoned data in it. It is where `def mitigate_model(self, model: torch.nn.Module, dataset: Dataset)` is called. 
- `test` conducts the testing process on a given post-mitigation model, given some dataset that could be clean or poisoned. Is is where both your pre and post process are called. **Do NOT shuffle your test data, the order of logits is critical for metric calculations**
  - Prior to test-time inference, `preprocess_transform` is called. `preprocess_transform` can optionally return an `info` dictionary that contains arbitrary information that preprocess may like to pass to `postprocess_transform` 
  - The model is then called on said preprocessed input data to return initial output logits. 
  - Finally, `postprocess_transform` is called to return the final, reported logits. `postprocess_transform` also optionally receives the info dictionary created by `proprocess_transform`

## Container Code

It is required to use the existing `example_trojai_mitigation.py` script as boilerplate, and to only change the code inside each function where necessary. The argument parser must remain in-tact to operate with the leaderboard infrastructure. 

- `def prepare_mitigation(args)` - Given the YAML/CLI args, construct your specific defense (that is a subclass of TrojaiMitigation) and return it. 
- `def prepare_model(path, num_classes, device)` - Prepares the round configuration's given model architecture and returns it. Do not modify this function. 
- `def mitigate_model(model, dataset_path, output_dir, output_name)` - Given the dataset and mitigation, run a defense on the model and output it's state dict to the specificed directory. Do not modify this function.
- `def test_model(model, mitigation, testset_path, batch_size, num_workers, device)` - Given the mitigated model and a mitigtaion technique, run testing on a given dataset. Do not modify this function. 

## Generating Metrics

The provided `example_metrics.py` script is an example of how to process metrics from `example_trojai_mitigation.py` 

- `--metrics`: list of metrics to calculate 
- `--result_file`: path of the produced pickle file that contains the logits and labels
- `--output_name`: output name for the csv that is produced
- `--model_name`: the desired name of the torch model on the spreadsheet
- `--data_type`: Either 'clean' or 'poisoned'; the data type used to produce a given metric results. Affects the spreadsheet output
- `--num_classes`: the number of classes the model was trained on. Required for metrics.

## System Requirements

- Linux (tested on Ubuntu 22.04)
- CUDA capable NVIDIA GPU (tested on A100)
- Python >= 3.9 

## Building and Using this Minimal Example

This minimal example was created with the [Round 11 sample model and data](https://github.com/usnistgov/trojai-example/tree/image-classification-sep2022) in mind. You should clone this repo and checkout to the `image-classification-sep2022` round to test.

1. It's recommended to create a virtual environment to install all the required dependencies. This may require installing [python-venv](https://packaging.python.org/en/latest/key_projects/#venv). You can then activate the virtual environment

```
python -m venv venv/
source venv/bin/activate
```

2. Install all the requirements:

```
pip install -r requirements.txt
```

3. Install the mitigation round framework into your venv as well:

```
pip install -e ./trojai-mitigation-round-framework
```

4. All your dependencies are installed. After this, you can run the `example_trojai_mitigation.py` script, ensuring you pass in the `--metaparameters_filepath` arg.

If conducting mitigation, ensure you pass the `mitigate` flag:

```
python example_trojai_mitigation.py \
mitigate \
--metaparameters_filepath metaparameters.json \
--schema_filepath metaparameters_schema.json \
--model_filepath /path/to/example/model.pt \
--dataset /path/to/example/dataset \
--output_dirpath /path/to/output/model \
--model_output_name <name of model>.pt \
```

After running mitigation, you can use the `test` flag to separately test the cleaned model on an arbitrary dataset which produces a `results.json` file:

```
python3 example_trojai_mitigation.py \
test \
--metaparameters_filepath metaparameters.json \
--schema_filepath metaparameters_schema.json \
--model_filepath /path/to/cleaned/model.pt \
--dataset /path/to/clean/or/poisoned/dataset \
--output_dirpath /path/to/output/logits/and/labels/
```

To obtain example metrics from here, call the `example_metrics.py` script on the produced result.json file:

```
python3 example_metrics.py \
--metrics f1 accuracy \
--result_file /path/to/results.json \
--model_name <model name to be used in csv> \
--data_type <clean / poisoned> \
--num_classes <class count>
```

## Package Solution into a Singularity Container
Package `example_trojai_mitigation.py` into a Singularity container.

1. Install Singularity
    
    - Follow: [https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux](https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux)
        
2. Build singularity based on `example_trojai_mitigation.def` file: 

    - delete any old copy of output file if it exists: `rm example_trojai_mitigation.simg`
    - package container: 
    
      ```bash
      sudo singularity build mitigation.simg example_trojai_mitigation.def
      ```

    which generates a `mitigation.simg` file.

3. Run mitigation and test using container

Mitigation:
```bash
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./mitigation.simg \
    mitigate \
  --metaparameters_filepath metaparameters.json \
  --schema_filepath metaparameters_schema.json \
  --model_filepath /path/to/example/model.pt \
  --dataset /path/to/example/dataset \
  --output_dirpath /path/to/output/model \
  --model_output_name <name of model>.pt \
  ```

Test:
  ```bash
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./mitigation.simg \
    test \
    --metaparameters_filepath metaparameters.json \
    --schema_filepath metaparameters_schema.json \
    --model_filepath /path/to/cleaned/model.pt \
    --dataset /path/to/clean/or/poisoned/dataset \
    --output_dirpath /path/to/output/logits/and/labels/
  ```