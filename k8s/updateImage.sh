# This script rebuilds the Docker image and pushes to local registry.
# The image name is flowerpytorch:latest, tagged as 10.1.2.64:5000/fl-ubuntu:latest

# Go to flower/
relativeScriptPath=$(dirname $0)
cd "${relativeScriptPath}/../"

# Rebuild image, tag, and push
docker build --build-arg SSH_PUB_KEY="$(cat ~/.ssh/id_rsa.pub)" --tag flowerpytorch:latest -f ./dockerFiles/flowerPytorch.Dockerfile .
docker tag flowerpytorch:latest 10.1.2.64:5000/fl-ubuntu:latest
docker push 10.1.2.64:5000/fl-ubuntu:latest
