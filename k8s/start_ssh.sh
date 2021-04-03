#!/bin/bash

for name in $(kubectl get pods --no-headers | awk '{print $1}');
do
    kubectl exec -it $name -- service ssh start
done
