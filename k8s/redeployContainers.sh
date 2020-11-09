kubectl delete -f ymlfile/fl-ubuntu.yaml --namespace=fl-fyp
kubectl apply -f ymlfile/fl-ubuntu.yaml -n fl-fyp
kubectl get pods -o wide
