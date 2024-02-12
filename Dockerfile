FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /trojai-example
COPY . . 

# this isnt best practice but quick test w/e
RUN mkdir /root/.ssh/
ADD ~/.ssh/id_rsa /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN apt-get update && apt-get upgrade -y && apt-get install -y git
RUN git clone git@github.com:usnistgov/trojai-round-generation-private.git && pip install -e ./trojai-round-generation-private/trojan-mitigation/

ENTRYPOINT ["python3", "example_trojai_mitigation.py", "--metaparameters", "metaparameters.yml"]