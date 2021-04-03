if [ $# -ne 1 ] 
then
    echo "invalid number of arguments"
    exit 1
fi

relativeScriptPath=$(dirname $0)
cd ${relativeScriptPath}

num_pods=$1

ips=`kubectl get pods -o wide --no-headers | head -n $num_pods | awk '{print $6}'`
echo "Container IPs:" $ips

log_dir=/app/examples/quickstart_pytorch/log/

mkdir -p ./log

count=0
for ip in $ips;
do
    scp -o StrictHostKeyChecking=no -r root@${ip}:${log_dir} .
done

