#!/bin/bash

for each in $(kubectl get pods | awk '{print $1}');
do
    kubectl exec -it $each -- /usr/sbin/service ssh start
done
