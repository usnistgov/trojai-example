FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
ARG 

WORKDIR /trojai-example
COPY . . 

RUN apt-get update && apt-get upgrade && apt-get install -y git

ENTRYPOINT ["python" "test.py"]