# Go to flower/
relativeScriptPath=$(dirname $0)
cd "${relativeScriptPath}/../"

# Rebuild image, tag, and push
docker build --tag flowerpytorch:latest -f ./dockerFiles/flowerPytorch.Dockerfile .
docker tag flowerpytorch:latest 10.1.2.64:5000/fl-ubuntu:latest
docker push 10.1.2.64:5000/fl-ubuntu:latest
