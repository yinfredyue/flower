#!/bin/bash
# Adopted from tensorflow/run-docker.sh.

set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

docker run -d --rm --network flower --name logserver pytorchcpu:latest \
  python3 -m flwr_experimental.logserver

docker run -d --rm --network flower --name server pytorchcpu:latest \
  python3 -m flwr_example.mnist_app.server --log_host=logserver:8081

# When running with docker, --server_address=server:8080, where server is the name of 
# container inside the docker network.
docker run -d --rm --network flower --name client_0 pytorchcpu:latest \
  python3 -m flwr_example.mnist_app.client --cid=0 --nb_clients=2 --server_address=server:8080 --log_host=logserver:8081

docker run -d --rm --network flower --name client_1 pytorchcpu:latest \
  python3 -m flwr_example.mnist_app.client --cid=1 --nb_clients=2 --server_address=server:8080 --log_host=logserver:8081

# If running locally
# python3 -m flwr_experimental.logserver
# When running locally, use localhost instead of [::].
# python3 -m flwr_example.mnist_app.server --log_host=localhost:8081
# python3 -m flwr_example.mnist_app.client --cid=0 --nb_clients=2 --server_address=localhost:8080 --log_host=localhost:8081
# python3 -m flwr_example.mnist_app.client --cid=1 --nb_clients=2 --server_address=localhost:8080 --log_host=localhost:8081

docker exec logserver tail -f flower_logs/flower.log