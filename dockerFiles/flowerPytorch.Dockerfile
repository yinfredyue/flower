FROM pytorchcpu

# Use pytorchcpu as base image. Then copies /src into the container.
# This avoids rebuilding dependencies.

# To build this image:
# $ cd flower/
# $ docker build --tag flowerpytorch:latest -f ./dockerFiles/flowerPytorch.Dockerfile ./dockerFiles

WORKDIR /app/src/py

COPY ./src/ /app/src/