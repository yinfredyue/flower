FROM python:3.7.9-slim-stretch

# Install the biggest dependencies before copying the wheel
RUN pip install tensorflow-cpu==2.3.0 numpy==1.18.3

COPY dist/flwr-0.11.0-py3-none-any.whl flwr-0.11.0-py3-none-any.whl
RUN python3.7 -m pip install --no-cache-dir 'flwr-0.11.0-py3-none-any.whl[examples-pytorch,examples-tensorflow,http-logger,baseline,ops]'
RUN rm flwr-0.11.0-py3-none-any.whl
