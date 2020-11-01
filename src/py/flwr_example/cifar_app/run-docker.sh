#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/../../

docker run -d --rm --network flower --name logserver pytorchcpu:latest \
  python3 -m flwr_experimental.logserver

docker run -d --rm --network flower --name server pytorchcpu:latest \
  python3 -m flwr_example.cifar_app.server \
  --rounds=1 \
  --sample_fraction=1.0 \
  --min_sample_size=2 \
  --min_num_clients=2 \
  --log_host=logserver:8081

# When running without docker, --server_address=[::]:8080 to indicate local machine.
# When running with docker, --server_address=server:8080, where server is the name of 
# container inside the docker network.
docker run -d --rm --network flower --name client_0 pytorchcpu:latest \
  python3 -m flwr_example.cifar_app.client --cid=0 --server_address=server:8080 --log_host=logserver:8081

docker run -d --rm --network flower --name client_1 pytorchcpu:latest \
  python3 -m flwr_example.cifar_app.client --cid=1 --server_address=server:8080 --log_host=logserver:8081

docker exec logserver tail -f flower_logs/flower.log