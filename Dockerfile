FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /trojai-example
COPY . . 

RUN apt-get update && apt-get upgrade -y && apt-get install -y git

ENTRYPOINT ["python3", "test.py"]