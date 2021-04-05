#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 NUM_CLIENTS" >&2
    exit 1
fi

num_clients=$1

for (( i=0; i < $num_clients; i++ ))
do  
    log_file="client$i.log"

    # "> $logfile" redirects stdout to $log_file
    # "2>&1" redirects stderr to stdout
    python3 client.py $i $num_clients > $log_file 2>&1 &
done