relativeScriptPath=$(dirname $0)
cd ${relativeScriptPath}

ips=`kubectl get pods -o wide --no-headers | awk '{print $6}'`
echo "Container IPs:" $ips

log_dir=/app/examples/quickstart_pytorch/log/

mkdir -p ./log

count=0
for ip in $ips;
do
    scp -o StrictHostKeyChecking=no -r root@${ip}:${log_dir} .
done

