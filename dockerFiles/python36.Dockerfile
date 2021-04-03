from python:3.6-slim-stretch

# To build this image:
# $ cd flower/
# $ docker build --tag python36:latest -f ./dockerFiles/python36.Dockerfile .

RUN apt-get update && apt-get upgrade -y

RUN pip install tensorflow==1.5
RUN pip install numpy==1.16.4