names=`kubectl get pods --no-headers | awk '{print $1}'`
num_clients=`echo "$(wc -w <<< $names)-2" | bc`
echo $num_clients
echo $names

# Global IP addresses
# `awk '{print $6}'` gets the column of IPs. Storing it to a variable converts to a row.
ips=`kubectl get pods -o wide --no-headers | awk '{print $6}'`
logserveraddr="$(echo $ips | awk '{print $1}'):8081"
serveraddr="$(echo $ips | awk '{print $2}'):8080"
echo $ips
echo $logserveraddr
echo $serveraddr

count=0
for name in $names;
do
    if [ $count -eq 0 ]
    then
        # Start logserver
        echo "Start log server"
        kubectl exec -it $name -- python3 -m flwr_experimental.logserver & # > "LogServer-${name}.log"
    elif [ $count -eq 1 ]
    then
        # Start server
        echo "Start server"
        kubectl exec -it $name -- python3 -m flwr_example.mnist_app.server --log_host=$logserveraddr & # > "server-${name}.log"
    else
        # Start client
        echo "Start client $((count-2))"
        kubectl exec -it $name -- python3 -m flwr_example.mnist_app.client --cid=$((count-2)) --nb_clients=$num_clients --server_address=$serveraddr --log_host=$logserveraddr & # > "client-${name}.log"
    fi
    count=$((count+1))
done

