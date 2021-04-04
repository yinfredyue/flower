#!/bin/bash

src_dir="/app/examples/shakespeare_lstm"

for name in $(kubectl get pods --no-headers | awk '{print $1}');
do
    kubectl exec -it $name -- rm -rf ${src_dir}/log/
done
