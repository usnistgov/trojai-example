This repo contains a minimal working example for a submission to the [TrojAI leaderboard](https://pages.nist.gov/trojai/). 
This minimal "solution" loads the model file and builds a random feature vector of requested length. The random features are used to fit a RandomForestRegressor. You can use this as your base to build your own solution.

Every solution submitted for evaluation must be containerized via [Singularity](https://singularity.hpcng.org/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)). 

The submitted Singularity container will be run by the TrojAI Evaluation Server using the specified [Container API](https://pages.nist.gov/trojai/docs/submission.html#container-api), inside of a virtual machine which has no network capability.

The container submitted for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

Your container will have access to these [Submission Compute Resources](https://pages.nist.gov/trojai/docs/architecture.html#compute-resources).


--------------
# Table of Contents
1. [Reusing the example detector](#reusing-the-example-detector)
2. [Container Configuration](#container-configuration)
3. [System Requirements](#system-requirements)
4. [Example Data](#example-data)
5. [Submission Instructions](#submission-instructions)
6. [How to Build this Minimal Example](#how-to-build-this-minimal-example)
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
# Container Configuration

TrojAI container submissions required that a configuration is included which enables TrojAI T&E to evaluate submitted detectors across various new dimensions. This means that each container needs to: 

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
    --automatic_configuration \
    --scratch_dirpath <scratch_dirpath> \
    --metaparameters_filepath <metaparameters_filepath> \
    --schema_filepath <schema_filepath> \
    --learned_parameters_dirpath <learned_params_dirpath> \
    --configure_models_dirpath <configure_models_dirpath>
   ```



# Container usage: Inferencing Mode

Executing the `entrypoint.py` in infernecing mode will output a result file that contains whether the model that is being analyzed is poisoned (1.0) or clean (0.0).

Example usage for inferencing:
   ```bash
   python entrypoint.py infer \
   --model_filepath <model_filepath> \
   --result_filepath <result_filepath> \
   --scratch_dirpath <scratch_dirpath> \
   --examples_dirpath <examples_dirpath> \
   --round_training_dataset_dirpath <round_training_dirpath> \
   --metaparameters_filepath <metaparameters_filepath> \
   --schema_filepath <schema_filepath> \
   --learned_parameters_dirpath <learned_params_dirpath>
   ```


--------------
# System Requirements

- Linux (tested on Ubuntu 20.04 LTS)
- CUDA capable NVIDIA GPU (tested on A4500)

Note: This example assumes you are running on a version of Linux (like Ubuntu 20.04 LTS) with a CUDA enabled NVIDIA GPU. Singularity only runs natively on Linux, and most Deep Learning libraries are designed for Linux first. While this Conda setup will install the CUDA drivers required to run PyTorch, the CUDA enabled GPU needs to be present on the system. 

--------------
# Example Data

Example data can be downloaded from the NIST [Leader-Board website](https://pages.nist.gov/trojai/). 

A small toy set of clean data is also provided in this repository under the model/example-data/ folder. This toy set of data is only for testing your environment works correctly. 

For some versions of this repository, the example model is too large to check into git. In those cases a model/README.md will point you to where the example model can be downloaded. 

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

    - `pip install torch --index-url https://download.pytorch.org/whl/cpu`
    - `conda install pytorch cpuonly -c pytorch`
    - `conda install opencv`
    - `pip install trojai_rl` 
    - `pip install gym_minigrid==1.0.2`
    - `pip install jsonschema jsonargparse jsonpickle scikit-learn==1.1.2`

## Test Fake Detector Without Containerization

1.  Clone the repository 
 
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    git checkout rl-lavaworld-jul2023
    ``` 

2. Test the python based `example_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python entrypoint.py infer \
   --model_filepath ./model/rl-lavaworld-jul2023-example/model.pt \
   --result_filepath ./output.txt \
   --scratch_dirpath ./scratch \
   --examples_dirpath ./model/rl-lavaworld-jul2023-example/clean-example-data \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json
    ```

    Example Output:
    
    ```bash
    2023-03-02 17:11:06,170 [INFO] [entrypoint.py:33] Calling the trojan detector
    2023-03-02 17:11:06,207 [INFO] [detector.py:156] Using compute device: cpu
    2023-03-02 17:11:06,207 [INFO] [detector.py:167] Evaluating on MiniGrid-LavaCrossingS9N1-v0
    2023-03-02 17:11:09,267 [INFO] [detector.py:249] Trojan probability: 0.12
    ```

3. Test self-configure functionality, note to automatically reconfigure should specify `--automatic_configuration`.

    ```bash
    python entrypoint.py configure \
    --automatic_configuration \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=/path/to/new-train-dataset
    ```

    The tuned parameters can then be used in a regular run.

   ```bash
    python entrypoint.py infer \
   --model_filepath ./model/rl-lavaworld-jul2023-example/model.pt \
   --result_filepath ./output.txt \
   --scratch_dirpath ./scratch \
   --examples_dirpath ./model/rl-lavaworld-jul2023-example/clean-example-data \
   --round_training_dataset_dirpath /path/to/train-dataset \
   --learned_parameters_dirpath ./new_learned_parameters \
   --metaparameters_filepath ./metaparameters.json \
   --schema_filepath=./metaparameters_schema.json
    ```

## Package Solution into a Singularity Container

Package `detector.py` into a Singularity container.

1. Install Singularity
    
    - Follow: [https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux](https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux)
        
2. Build singularity based on `detector.def` file: 

    - delete any old copy of output file if it exists: `rm detector.simg`
    - package container: 
    
      ```bash
      sudo singularity build detector.simg detector.def
      ```

    which generates a `detector.simg` file.

3. Test run container: 

    ```bash
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./detector.simg \
    infer \
    --model_filepath=./model/rl-lavaworld-jul2023-example/model.pt \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/rl-lavaworld-jul2023-example/clean-example-data/ \
    --round_training_dataset_dirpath=/path/to/training/dataset/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./learned_parameters/
    ```

    Example Output:
    ```bash
    2023-03-02 17:11:06,170 [INFO] [entrypoint.py:33] Calling the trojan detector
    2023-03-02 17:11:06,207 [INFO] [detector.py:156] Using compute device: cpu
    2023-03-02 17:11:06,207 [INFO] [detector.py:167] Evaluating on MiniGrid-LavaCrossingS9N1-v0
    2023-03-02 17:11:09,267 [INFO] [detector.py:249] Trojan probability: 0.12
    ```
