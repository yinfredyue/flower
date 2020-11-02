FROM pytorchcpu

# Use pytorchcpu as base image. Then copies /src into the container.
# This avoids rebuilding dependencies.

WORKDIR /app/src/py

COPY ./src /app/src