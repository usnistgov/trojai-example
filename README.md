This repo contains a minimal working example for a submission to the TrojAI NIST Test and Evaluation server. This minimal ‘solution’ loads the model file, inferences 10 random tensors, and then writes a random number to the output file. You can use this as your base to build your own solution for the [TrojAI leaderboard](https://pages.nist.gov/trojai/).

Every solution submitted for evaluation must be containerized via [Singularity](https://sylabs.io/docs/) (see this [Singularity tutorial](https://pawseysc.github.io/sc19-containers/)). 

The submitted Singularity container will be run by the NIST test and evaluation server using the specified [Container API](https://pages.nist.gov/trojai/docs/submission.html#container-api), inside of a virtual machine which has no network capability.

The container submitted to NIST for evaluation must perform trojan detection for a single trained AI model file and output a single probability of the model being poisoned. The test and evaluation infrastructure will iterate over the *N* models for which your container must predict trojan presence. 

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

- Linux (tested on Ubuntu 18.04 LTS)
- CUDA capable NVIDIA GPU (tested on Titan RTX)

Note: This example assumes you are running on a version of Linux (like Ubuntu 18.04 LTS) with a CUDA enabled NVIDIA GPU. Singularity only runs natively on Linux, and most Deep Learning libraries are designed for Linux first. While this Conda setup will install the CUDA drivers required to run PyTorch, the CUDA enabled GPU needs to be present on the system.   

--------------
# Example Data

Example data can be downloaded from the NIST [Leader-Board website](https://pages.nist.gov/trojai/).


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

1. `conda create --name fake_detector python=3.6` ([help](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html))
2. `conda activate fake_detector`
3. Install required packages into this conda environment

    1. `conda install numpy`
    2. `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch` ([help](https://pytorch.org/get-started/locally/))

## Test Fake Detector Without Containerization

1.  Clone the repository 
 
    ```
    git clone https://github.com/usnistgov/trojai-example
    cd trojai-example
    ``` 

2. Test the python based `fake_trojan_detector` outside of any containerization to confirm pytorch is setup correctly and can utilize the GPU.

    ```bash
    python fake_trojan_detector.py --model_filepath=./model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/
    ```

    Example Output:
    
    ```bash
    Trojan Probability: 0.07013004086445151
    ```

## Package Solution into a Singularity Container

Package `fake_trojan_detector.py` into a Singularity container.

1. Install Singularity
    
    - For Ubuntu 18.04 LTS: `sudo apt install singularity-container`
    - For others Linux Distros follow: [https://sylabs.io/guides/3.5/admin-guide/installation.html#installation-on-linux](https://sylabs.io/guides/3.5/admin-guide/installation.html#installation-on-linux)
        
2. Build singularity based on `fake_trojan_detector.def` file: 

    - delete any old copy of output file if it exists: `rm fake_trojan_detector.simg`
    - package container: 
    
      ```bash
      sudo singularity build fake_trojan_detector.simg fake_trojan_detector.def
      ```

    which generates a `fake_trojan_detector.simg` file.

    Example Output:
    ```bash
    $ sudo singularity build fake_trojan_detector.simg fake_trojan_detector.def
    Using container recipe deffile: fake_trojan_detector.def
    Sanitizing environment
    Adding base Singularity environment to container
    tar: ./.exec: implausibly old time stamp -9223372036854775808
    tar: ./.run: implausibly old time stamp -9223372036854775808
    tar: ./.shell: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/exec: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/run: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/shell: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/start: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions/test: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/actions: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/01-base.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/90-environment.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/95-apps.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env/99-base.sh: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/env: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/libs: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/runscript: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d/startscript: implausibly old time stamp -9223372036854775808
    tar: ./.singularity.d: implausibly old time stamp -9223372036854775808
    tar: ./.test: implausibly old time stamp -9223372036854775808
    tar: ./dev: implausibly old time stamp -9223372036854775808
    tar: ./environment: implausibly old time stamp -9223372036854775808
    tar: ./etc/hosts: implausibly old time stamp -9223372036854775808
    tar: ./etc/resolv.conf: implausibly old time stamp -9223372036854775808
    tar: ./etc: implausibly old time stamp -9223372036854775808
    tar: ./home: implausibly old time stamp -9223372036854775808
    tar: ./proc: implausibly old time stamp -9223372036854775808
    tar: ./root: implausibly old time stamp -9223372036854775808
    tar: ./singularity: implausibly old time stamp -9223372036854775808
    tar: ./sys: implausibly old time stamp -9223372036854775808
    tar: ./tmp: implausibly old time stamp -9223372036854775808
    tar: ./var/tmp: implausibly old time stamp -9223372036854775808
    tar: ./var: implausibly old time stamp -9223372036854775808
    tar: .: implausibly old time stamp -9223372036854775808
    Docker image path: index.docker.io/pytorch/pytorch:latest
    Cache folder set to /root/.singularity/docker
    Exploding layer: sha256:16c48d79e9cc2d6cdb79a91e9c410250c1a44102ed4c971fbf24692cc09f2351.tar.gz
    Exploding layer: sha256:3c654ad3ed7d66e3caa5ab60bee1b166359d066be7e9edca6161b72ac06f2008.tar.gz
    Exploding layer: sha256:6276f4f9c29df0a2fc8019e3c9929e6c3391967cb1f610f57a3c5f8044c8c2b6.tar.gz
    Exploding layer: sha256:a4bd43ad48cebce2cad4207b823fe1693e10c440504ce72f48643772e3c98d7a.tar.gz
    Exploding layer: sha256:34cb2ecb4e7e4513ede923e58c6a219e8e025a5f27e9c8e1df37c0f9972cfd9e.tar.gz
    Exploding layer: sha256:1271bead61037d0e1f1e3c7efc63848627a2bd513c884201c3178964c21293a2.tar.gz
    Exploding layer: sha256:913bf197139d82f9984a8417548fee109c096bb7e6dd9672e1a42d8ed8644d59.tar.gz
    Exploding layer: sha256:96e5a748a56a153207ca15202c318e29f61ddfd44784cdcbde95bb7086fa0871.tar.gz
    Exploding layer: sha256:ac87c593cb7de82616275e9ef3b085ebc758b648553381c9e094c70ba54a7bf7.tar.gz
    Exploding layer: sha256:f4cfecb48ca26a9ea56c738af1311b4a44cd075e9e92ac8c1870edffa0f11dfd.tar.gz
    User defined %runscript found! Taking priority.
    Adding files to container
    Copying './fake_trojan_detector.py' to '/'
    Adding runscript
    Finalizing Singularity container
    Calculating final size for metadata...
    Skipping checks
    Building Singularity image...
    Singularity container built: fake_trojan_detector.simg
    Cleaning up...
    ```

3. Test run container: 

    ```bash
    singularity run --nv ./fake_trojan_detector.simg --model_filepath ./model.pt --result_filepath ./output.txt --scratch_dirpath ./scratch
    ```

    Example Output:
    ```bash
    Trojan Probability: 0.7091788412534845
    ```
