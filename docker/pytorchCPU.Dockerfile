FROM python:3.7.9-slim-stretch

# Install vim
RUN apt-get update
RUN apt-get install vim -y

# Install torch==1.6.0+cpu and torchvision==0.7.0+cpu using local wheel
# Downloaded from https://download.pytorch.org/whl/torch_stable.html
COPY ./docker/torch-1.6.0+cpu-cp37-cp37m-linux_x86_64.whl /tmp/install/
COPY ./docker/torchvision-0.7.0+cpu-cp37-cp37m-linux_x86_64.whl /tmp/install/
RUN pip install /tmp/install/torch-1.6.0+cpu-cp37-cp37m-linux_x86_64.whl
RUN pip install /tmp/install/torchvision-0.7.0+cpu-cp37-cp37m-linux_x86_64.whl

# Install flower
RUN pip install flwr

# Install modules for log server
RUN pip install boto3 
RUN pip install matplotlib

# For client container, the work directory MUST be /src/py, otherwise Python
# cannot find the module correctly.
WORKDIR /app/src/py

COPY ./src/ /app/src/
