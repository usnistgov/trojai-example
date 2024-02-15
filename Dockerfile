FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /trojai-example
COPY . . 

RUN apt-get update && apt-get upgrade -y && apt-get install -y git ffmpeg libsm6 libxext6
RUN pip install -r requirements.txt
RUN pip install -r ./BackdoorBox-APL/requirements.txt
RUN pip install -e ./BackdoorBox-APL
RUN pip install -e ./mitigationround

ENTRYPOINT ["python3", "example_trojai_mitigation.py", "--metaparameters", "metaparameters.yml"]
