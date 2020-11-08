count=0
names=`kubectl get pods | awk '{print $1}'`
clients=`echo "$(wc -w <<< $names)-3" | bc`
cutnames=`kubectl get pods | awk '{print $1}' | tail -n $((clients+2))`

# Global IP addresses
ips=`kubectl get pods -o wide | awk '{print $6}'`
logserveraddr="$(echo $ips | awk '{print $2}'):8081"
serveraddr="$(echo $ips | awk '{print $3}'):8080"

for name in $cutnames;
do
    if [ $count -eq 0 ]
    then
        # Start logserver
        echo "Start log server"
        kubectl exec -it $name -- python3 -m flwr_experimental.logserver &
    elif [ $count -eq 1 ]
    then
        # Start server
        echo "Start server"
        kubectl exec -it $name -- python3 -m flwr_example.mnist_app.server --log_host=$logserveraddr &
    else
        # Start client
        echo "Start client $((count-2))"
        kubectl exec -it $name -- python3 -m flwr_example.mnist_app.client --cid=$((count-2)) --nb_clients=$clients --server_address=$serveraddr --log_host=$logserveraddr &
    fi
    count=$((count+1))
done