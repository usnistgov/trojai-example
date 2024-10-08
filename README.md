This repo contains a minimal working example for a submission to the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 
This minimal "solution" loads the model file, extracts its weights, and transform these
weights into a set of features. The features are extracted by flattening every layer and 
applying [FastICA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html#sklearn.decomposition.FastICA) 
to fit a RandomForestRegressor. You can use this as your base to build your own solution.

Every solution submitted for evaluation must be containerized via [Singularity](https://singularity.hpcng.org/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)). 

The submitted Singularity container will be run by the TrojAI Evaluation Server using the specified [Container API](https://pages.nist.gov/trojai/docs/submission.html#container-api), inside of a virtual machine which has no network capability.

The container submitted for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

Your container will have access to these [Submission Compute Resources](https://pages.nist.gov/trojai/docs/architecture.html#compute-resources).


--------------
# Table of Contents
1. [Reusing the example detector](#reusing-the-example-detector)
2. [New Container Configuration](#new-container-configuration)
3. [Container metaparameters](#container-metaparameters-files)
4. [System Requirements](#system-requirements)
5. [Example Data](#example-data)
6. [Submission Instructions](#submission-instructions)
7. [How to Build this Minimal Example](#how-to-build-this-minimal-example)
    1. [Install Anaconda Python](#install-anaconda-python)
    2. [Setup the Conda Environment](#setup-the-conda-environment)
    3. [Test Fake Detector Without Containerization](#test-fake-detector-without-containerization)
    4. [Package Solution into a Singularity Container](#package-solution-into-a-singularity-container)

--------------
# Reusing the example detector

Please use this example as a template for submissions into TrojAI.

You will need to modify at least 3 files and 1 directory:
* detector.py: File containing the codebase for the detector
* metaparameters.json: The set of tunable parameters used by your container, it should
  validate against metaparameters-schema.json.
* metaparameters-schema.json: JSON schema describing the metaparameters that can be
  changed during inference or training. 
* learned_parameters/: Directory containing data created at training time (that can be 
  changed with re-training the detector)

The detector class (in detector.py) needs to implement 4 methods to work properly: 
* `__init__(self, metaparameter_filepath, learned_parameters_dirpath)`: The initialization
function that should load the metaparameters from the given file path, and 
learned_parameters if necessary.
* `automatic_configure(self, models_dirpath)`: A function to automatically re-configure 
the detector by performing a grid search on a preset range of meta-parameters. This 
function should automatically change the meta-parameters, call `manual_configure` and 
output a new meta-parameters.json file (in the learned_parameters folder) when optimal 
meta-parameters are found.   
* `manual_configure(self, models_dirpath)`: A function that re-configure (re-train) the 
detector given a metaparameters.json file. 
* `infer(self, model_filepath, result_filepath, scratch_dirpath, examples_dirpath, round_training_dataset_dirpath)`: Inference
function to detect if a particular model is poisoned (1) or clean (0).

During the development of these functions, you will come up with variables that change the 
behavior of your detector:
* Variables influencing the training of the detector's algorithm: these variables should 
be loaded from the metaparameters.json file and have their name start with "train_". Typically,
these variable are used in the `automatic_configure` and `manual_configure` functions only.
* Training datastructure computed from training variables: these structure should be dumped
(in any format) in the learned_parameters folder. During re-training, their content will 
change. These datastructures are created within the `automatic_configure` and 
`manual_configure` functions and should be loaded and used in the `infer` function.
* Inference variables: Similarly to the training variables, variables used only in the
`infer` function should be loaded from the metaparameters.json file but start with 
"infer_".

When all these file are implemented as intended, your detector should work properly with
the provided `entrypoint.py` file and can be packaged in a Singularity container. 
The `entrypoint.py` file should be used as-is and should not be modified.

--------------
# New Container Configuration

With the release of TrojAI Round 10, a new container configuration is being added that enables TrojAI T&E to evaluate submitted detectors across various new dimensions. The main changes require submitted containers to do two new things: 

- Specify a "metaparameters" file that documents a container's manually tunable parameters and their range of possible values. 
- Generate "learned parameters" via a new reconfiguration API.

Submitted containers will now need to work in two different modes:

- Inference Mode:  Containers will take as input both a "metaparameter" file and a model and output the probability of poisoning. 
- Reconfiguration Mode: Containers will take a new dataset as input and output a file dump of the new learned parameters tuned to that input dataset.

# Container usage: Reconfiguration Mode

Executing the `entrypoint.py` in reconfiguration mode will produce the necessary metadata for your detector and save them into the specified "learned_parameters" directory.

Example usage for one-off reconfiguration:
   ```bash
  python entrypoint.py configure \
  --scratch_dirpath <scratch_dirpath> \
  --metaparameters_filepath <metaparameters_filepath> \
  --schema_filepath <schema_filepath> \
  --learned_parameters_dirpath <learned_params_dirpath> \
  --configure_models_dirpath <configure_models_dirpath>
   ```

Example usage for automatic reconfiguraiton:
   ```bash
   python entrypoint.py configure \
    --scratch_dirpath <scratch_dirpath> \
    --metaparameters_filepath <metaparameters_filepath> \
    --schema_filepath <schema_filepath> \
    --learned_parameters_dirpath <learned_params_dirpath> \
    --configure_models_dirpath <configure_models_dirpath> \
    --automatic_configuration
   ```



# Container usage: Inferencing Mode

Executing the `entrypoint.py` in infernecing mode will output a result file that contains whether the model that is being analyzed is poisoned (1.0) or clean (0.0).

Example usage for inferencing:
   ```bash
   python entrypoint.py infer \
   --model_filepath <model_filepath> \
   --result_filepath <result_filepath> \
   --scratch_dirpath <scratch_dirpath> \
   --round_training_dataset_dirpath <round_training_dirpath> \
   --metaparameters_filepath <metaparameters_filepath> \
   --schema_filepath <schema_filepath> \
   --learned_parameters_dirpath <learned_params_dirpath>
   ```

--------------
# Container metaparameters files

There are two metaparameters files that are required for submissions. 
- metaparameters.json
- metaparameters_schema.json

The metaparameters.json file is used to specify customizable parameters for your submissions. These should include two types of parameters: (1) train and (2) infer. Train parameters should prefix the parameter name with "train_<name>" to denote the training parameter options. Infer parameters should prefix the parameter name with "infer_<name>" to denote inference parameters. These parameters are to be used to customize the behavior/functionality of the submission.

The metaparameters_schema.json is used to provide properties for your parameters to give us suggested minimum/maximum values and actual minimum/maximum values. These bounds can then be used when exploring the parameter space. 

In addition to parameter specifications, the metaparameters_schema.json file contains per-container metadata that describes the submission. These are "title", "technique", "technique_description", "technique_changes", "technique_type", "commit_id", and "repo_name". 

For "Performers" these parameters must be unique to your submission. They must not be identical to the trojai-example. The only exception is the "technique_type", which is an enum containing one (or more) of the following values: Weight Analysis, Trigger Inversion, Attribution Analysis, Jacobian Inspection, Other. If your technique type is missing from these options (or if you use Other), please let us know so that we can include any new techniques types.

For more details please see: https://pages.nist.gov/trojai/docs/submission.html#parameter-loading



--------------
# System Requirements

- Linux (tested on Ubuntu 20.04 LTS)
- CUDA capable NVIDIA GPU (tested on A4500)

Note: This example assumes you are running on a version of Linux (like Ubuntu 20.04 LTS) with a CUDA enabled NVIDIA GPU. Singularity only runs natively on Linux, and most Deep Learning libraries are designed for Linux first. While this Conda setup will install the CUDA drivers required to run PyTorch, the CUDA enabled GPU needs to be present on the system. 

--------------
# Example Data

Example data can be downloaded from the NIST [Leader-Board website](https://pages.nist.gov/trojai/). 

A small toy set of clean & poisioned data is also provided in this repository under the model/example-data/ folder. This toy set of data is only for testing your environment works correctly. 

--------------
# Submission Instructions

1. Package your trojan detection solution into a Singularity Container.
    - Name your container file based on which [server](https://pages.nist.gov/trojai/docs/architecture.html) you want to submit to.
2. Request an [Account](https://pages.nist.gov/trojai/docs/accounts.html) on the NIST Test and Evaluation Server.
3. Follow the [Google Drive Submission Instructions](https://pages.nist.gov/trojai/docs/submission.html#container-submission).
4. View job status and results on the [Leader-Board website](https://pages.nist.gov/trojai/).
5. Review your [submission logs](https://pages.nist.gov/trojai/docs/submission.html#output-logs) shared back with your team Google Drive account.


--------------
# How to Build this Minimal Example

## Install Anaconda Python

[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)

## Setup the Conda Environment

1. `conda create --name trojai-example python=3.8 -y` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
2. `conda activate trojai-example`
3. Install required packages into this conda environment

    - `conda install pytorch=2.2 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y`
    - `pip install tqdm jsonschema scikit-learn bsdiff4`

## Test Fake Detector Without Containerization

1.  Clone the repository 
 
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    git checkout cyber-pe-aug2024
    ``` 

2. Test the python based `example_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python entrypoint.py infer \
   --model_filepath ./model/id-00000001/model.pt \
   --result_filepath ./scratch/output.txt \
   --scratch_dirpath ./scratch \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

3. Test self-configure functionality, note to automatically reconfigure should specify `--automatic_configuration`.

    ```bash
    python entrypoint.py configure \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/path/to/new-train-dataset
    ```

    The tuned parameters can then be used in a regular run.

    ```bash
    python entrypoint.py infer \
    --model_filepath=./model/id-00000001/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./new_learned_parameters/metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/
    ```

## Package Solution into a Singularity Container

Package `example_trojan_detector.py` into a Singularity container.

1. Install Apptainer
    
    - Follow: [https://apptainer.org/docs/admin/latest/installation.html](https://apptainer.org/docs/admin/latest/installation.html)
        
2. Build singularity container based on `example_trojan_detector.def` file: 

    - delete any old copy of output file if it exists: `rm example_trojan_detector.sif`
    - package container: 
    
      ```bash
      apptainer build example_trojan_detector.sif example_trojan_detector.def
      ```

    which generates a `example_trojan_detector.sif` file.

3. Test run container: 

    ```bash
    apptainer run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.sif \
    infer \
    --model_filepath=./model/id-00000001/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./learned_parameters/
    ```

    Example Output:
    ```bash
    Trojan Probability: 0.7091788412534845
    ```

4. Test self-tune functionality.

    ```bash
    apptainer run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.sif \
    configure \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/path/to/new-train-dataset
    ```

    The tuned parameters can then be used in a regular run.

    ```bash
    apptainer run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.sif \
   infer \
    --model_filepath=./model/id-00000001/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./new_learned_parameters/metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/
    ```
