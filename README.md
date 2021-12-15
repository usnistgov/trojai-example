This repo contains a minimal working example for a submission to the [TrojAI leaderboard](https://pages.nist.gov/trojai/). This minimal ‘solution’ loads the model file, inferences the example text sequences, and then writes a random number to the output file. You can use this as your base to build your own solution. 

Every solution submitted for evaluation must be containerized via [Singularity](https://singularity.hpcng.org/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)). 

The submitted Singularity container will be run by the TrojAI Evaluation Server using the specified [Container API](https://pages.nist.gov/trojai/docs/submission.html#container-api), inside of a virtual machine which has no network capability.

The container submitted for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

Your container will have access to these [Submission Compute Resources](https://pages.nist.gov/trojai/docs/architecture.html#compute-resources).


--------------
# Table of Contents
1. [System Requirements](#system-requirements)
2. [Example Data](#example-data)
2. [Submission Instructions](#submission-instructions)
3. [How to Build this Minimal Example](#how-to-build-this-minimal-example)
    1. [Install Anaconda Python](#install-anaconda-python)
    2. [Setup the Conda Environment](#setup-the-conda-environment)
    3. [Test Fake Detector Without Containerization](#test-fake-detector-without-containerization)
    4. [Package Solution into a Singularity Container](#package-solution-into-a-singularity-container)


--------------
# System Requirements

- Linux (tested on Ubuntu 20.04 LTS)
- CUDA capable NVIDIA GPU (tested on RTX 2080 Ti and RTX 3090)

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

    1. `conda install pytorch torchvision torchtext cudatoolkit=11.1 -c pytorch-lts -c nvidia` 
    2. `pip install jsonargparse jsonpickle jsonschema transformers==4.10.3 datasets`

## Test Fake Detector Without Containerization

1.  Clone the repository 
 
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    ``` 

2. Test the python based `example_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python example_trojan_detector.py \
    --model_filepath=./model/id-00000000/model.pt \
    --tokenizer_filepath=./tokenizers/google-electra-small-discriminator.pt \
    --features_filepath=./features.csv \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/id-00000000/example_data/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./learned_parameters/
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

3. Test self-configure functionality.

    ```bash
    python example_trojan_detector.py \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --configure_mode \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=./model/
    ```

    The tuned parameters can then be used in a regular run.

    ```bash
    python example_trojan_detector.py \
    --model_filepath=./model/id-00000000/model.pt \
    --tokenizer_filepath=./tokenizers/google-electra-small-discriminator.pt \
    --features_filepath=./features.csv \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/id-00000000/example_data/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/
    ```

## Package Solution into a Singularity Container

Package `example_trojan_detector.py` into a Singularity container.

1. Install Singularity
    
    - Follow: [https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux](https://singularity.hpcng.org/admin-docs/master/installation.html#installation-on-linux)
        
2. Build singularity based on `example_trojan_detector.def` file: 

    - delete any old copy of output file if it exists: `rm example_trojan_detector.simg`
    - package container: 
    
      ```bash
      sudo singularity build example_trojan_detector.simg example_trojan_detector.def
      ```

    which generates a `example_trojan_detector.simg` file.

3. Test run container: 

    ```bash
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.simg \
    --model_filepath=./model/id-00000000/model.pt \
    --tokenizer_filepath=./tokenizers/google-electra-small-discriminator.pt \
    --features_filepath=./features.csv \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/id-00000000/example_data/ \
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
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.simg \
    --scratch_dirpath=./scratch/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --configure_mode \
    --learned_parameters_dirpath=./new_learned_parameters/ \
    --configure_models_dirpath=./model/
    ```

    The tuned parameters can then be used in a regular run.

    ```bash
    singularity run \
    --bind /full/path/to/trojai-example \
    --nv \
    ./example_trojan_detector.simg \
    --model_filepath=./model/id-00000000/model.pt \
    --tokenizer_filepath=./tokenizers/google-electra-small-discriminator.pt \
    --features_filepath=./features.csv \
    --result_filepath=./output.txt \
    --scratch_dirpath=./scratch/ \
    --examples_dirpath=./model/id-00000000/example_data/ \
    --metaparameters_filepath=./metaparameters.json \
    --schema_filepath=./metaparameters_schema.json \
    --learned_parameters_dirpath=./new_learned_parameters/
    ```
