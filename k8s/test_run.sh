names=`kubectl get pods --no-headers | awk '{print $1}'`
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

src_dir="/app/examples/quickstart_pytorch"
staleness=2
rounds=3

count=0
for name in $names;
do
    if [ $count -eq 0 ]
    then
        # Start server
        echo "Start server"

        # Important: the `&` should be outside the quotes. We want `kutectl exec` command to run in the background.
        kubectl exec -it $name -- bash -c "mkdir -p ${src_dir}/log"
        kubectl exec -it $name -- bash -c "python3 ${src_dir}/server.py --num_clients $num_clients --staleness_bound $staleness --rounds $rounds &> ${src_dir}/log/server.log" &
        sleep 5
    else
        # Start client
        idx=$((count-1))
        echo "Start client $idx"

        # Important: the `&` should be outside the quotes.
        kubectl exec -it $name -- bash -c "mkdir -p ${src_dir}/log"
        kubectl exec -it $name -- bash -c "python3 ${src_dir}/client.py --num_clients $num_clients --staleness_bound $staleness --server_ip $serveraddr --idx $idx &> ${src_dir}/log/client${idx}.log" &
    fi
    count=$((count+1))
done

# Testing
# kubectl exec -it $name -- python3 /app/examples/quickstart_pytorch/server.py --num_clients 2 --staleness_bound 2 --rounds 3
# kubectl exec -it $name -- python3 /app/examples/quickstart_pytorch/client.py --num_clients 2 --staleness_bound 2 --server_ip 10.244.3.6:8080 --idx 0
