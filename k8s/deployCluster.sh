# This script 
# 1. removes the existing cluster deployment (if any)
# 2. Creates a new cluster
# 3. Display the pods (i.e. containers) created in the cluster
# You may need to wait for a while until all pods are up and running.

relativeScriptPath=$(dirname $0)
cd "${relativeScriptPath}/../../"

kubectl delete -f ymlfile/fl-ubuntu.yaml --namespace=flower-fyp
kubectl apply -f ymlfile/fl-ubuntu.yaml -n flower-fyp

echo "Container being set up..."

for i in {1..6}
do
    sleep 10
    kubectl get pods -o wide
done


