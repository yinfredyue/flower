# This script
# 1. removes the existing cluster deployment (if any)
# 2. Creates a new cluster
# 3. Display the pods (i.e. containers) created in the cluster
# You may need to wait for a while until all pods are up and running.

relativeScriptPath=$(dirname $0)
cd "${relativeScriptPath}/"

NAMESPACE=flower-fyp

kubectl delete -f fl-ubuntu.yaml --namespace=$NAMESPACE
kubectl apply -f fl-ubuntu.yaml -n $NAMESPACE

echo "Container being set up..."

for i in {1..6}
do
    sleep 10
    kubectl get pods -o wide
done


