# This file runs one test.
# You must provide 4 arguments.

if [ $# -ne 4 ]
then
    echo "invalid number of arguments"
    exit 1
fi

total_pods=$1
staleness=$2
rounds=$3
max_delay=$4
echo "Test config:"
echo "Total pods" $total_pods
echo "Staleness: " $staleness
echo "Rounds: " $rounds
echo "Max delay: " $max_delay

timeout=7200 # 2 hours

names=`kubectl get pods --no-headers | head -n $total_pods | awk '{print $1}'`
echo "Container names:" $names

num_containers=`echo "$(wc -w <<< $names)" | bc`
echo "Number of containers:" $num_containers

num_clients=`echo "$num_containers-1" | bc`
echo "Number of clients:" $num_clients

# Global IP addresses
# `awk '{print $6}'` gets the column of IPs. Storing it to a variable converts to a row.
ips=`kubectl get pods -o wide --no-headers | awk '{print $6}'`
serveraddr="$(echo $ips | awk '{print $1}'):8080"
echo "IPs of all containers:" $ips
echo "Server IP:" $serveraddr

src_dir="/app/examples/shakespeare_lstm"

count=0
for name in $names;
do
    # Create log directory
    kubectl exec -it $name -- bash -c "mkdir -p ${src_dir}/log"

    if [ $count -eq 0 ]
    then
        # Start server
        echo "Start server"

        # Important: the `&` should be outside the quotes. We want `kutectl exec` command to run in the background.
        kubectl exec $name -- bash -c "cd ${src_dir} && python3 server.py --num_clients $num_clients --staleness_bound $staleness --rounds $rounds &> ./log/server.log" --request-timeout=${timeout} &
    
        sleep 5
    else
        # Start client
        idx=$((count-1))
        echo "Start client $idx"

        # Important: the `&` should be outside the quotes.
        kubectl exec $name -- bash -c "cd ${src_dir} && python3 client.py --num_clients $num_clients --staleness_bound $staleness --server_ip $serveraddr --idx $idx &> ./log/client${idx}.log" --request-timeout=${timeout} &
    fi
    count=$((count+1))
done

# Testing
# kubectl exec -it $name -- python3 /app/examples/quickstart_pytorch/server.py --num_clients 2 --staleness_bound 2 --rounds 3
# kubectl exec -it $name -- python3 /app/examples/quickstart_pytorch/client.py --num_clients 2 --staleness_bound 2 --server_ip 10.244.3.6:8080 --idx 0
