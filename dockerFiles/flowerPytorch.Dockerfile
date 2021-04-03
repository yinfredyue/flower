FROM python:3.7.9-slim-stretch

# To build this image:
# $ cd flower/
# $ docker build --tag flowerpytorch:latest -f ./dockerFiles/flowerPytorch.Dockerfile .

RUN apt-get update && apt-get upgrade -y

# Install vim
RUN apt-get install vim -y

# SSH related
# Useful reference:
# Send command using ssh: https://malcontentcomics.com/systemsboy/2006/07/send-remote-commands-via-ssh.html
# Without prompt: https://unix.stackexchange.com/q/229124
# Start SSH server: https://stackoverflow.com/a/32178958/9057530
# How CMD works: https://stackoverflow.com/a/42219138/9057530
# Difference between RUN and CMD: https://stackoverflow.com/q/37461868/9057530
# The trick here for no-password ssh: the Docker image is built to be the same for the server and all clients,
# thus the generated ssh key-password pair is the same for all contains. Thus, copying the key to authorized_keys
# is enough to enable no-password ssh among all contains using this image.
# ssh-keygen -f provides the filename to avoids prompt. Check man page.

# inter-container passwordless ssh
RUN apt-get install openssh-server -y
RUN ssh-keygen -t rsa -P "" -f "/root/.ssh/id_rsa" 
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# From server to container
ARG SSH_PUB_KEY
RUN echo "${SSH_PUB_KEY}}" >> /root/.ssh/authorized_keys

# Expose port
EXPOSE 22

# Install virtualenv
RUN pip install virtualenv
RUN virtualenv env

# Enter virtual env: https://pythonspeed.com/articles/activate-virtualenv-dockerfile/
ENV VIRTUAL_ENV=/app/env/
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# In virtualvenv now
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 2. Install flwr with pip (to install numpy, grpc, and other dependencies), 
# then replace with our implementation. 
# I use Python3.8 on my machine, but I cannot find any Python3.8 docker image
# that supports installing PyTorch easily with downloaded wheel. So remember to
# change the directory from 3.8 (locally) to 3.7 (in the docker image).
RUN pip install flwr
COPY ./env/lib/python3.8/site-packages/flwr/ /app/env/lib/python3.7/site-packages/flwr/

COPY ./examples/ /app/examples/
WORKDIR /app/examples/quickstart_pytorch/

ENTRYPOINT service ssh start && bash

# To avoid the prompt of trust, when doing ssh, add an argument:
# ssh -o StrictHostKeyChecking=no root@ip


# To run server
# $ docker run --rm -ti --name server flowerpytorch:latest /bin/bash
# $ python server.py --num_clients 2 --staleness_bound 2 --rounds 3

# To run client1
# $ docker run --rm -ti --name client0 flowerpytorch:latest /bin/bash
# $ python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.3:8080 --idx 0

# To run client2
# $ docker run --rm -ti --name client1 flowerpytorch:latest /bin/bash
# $ python client.py --num_clients 2 --staleness_bound 2 --server_ip 172.17.0.3:8080 --idx 1


# Outside docker
# $ python server.py --num_clients 2 --staleness_bound 2 --rounds 3
# $ python client.py --num_clients 2 --staleness_bound 2 --idx 0
# $ python client.py --num_clients 2 --staleness_bound 2 --idx 1
