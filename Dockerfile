FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /trojai-example
COPY . . 

RUN apt-get update && apt-get upgrade -y && apt-get install -y git
RUN git clone https://github.com/usnistgov/trojai-round-generation-private.git && pip install -e ./trojai-round-generation-private/trojan-mitigation/

ENTRYPOINT ["python3", "example_trojai_mitigation.py", "--metaparameters", "metaparameters.yml"]