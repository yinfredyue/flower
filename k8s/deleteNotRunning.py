# Delete containers that fail to get ready.
# https://stackoverflow.com/a/40566922/9057530
import subprocess

names = subprocess.getoutput("kubectl get pods -o wide --no-headers | awk '{print $1}'").split()
states = subprocess.getoutput("kubectl get pods -o wide --no-headers | awk '{print $2}'").split()

print(names)
print(states)

for i in range(len(names)):
    name = names[i]
    state = states[i]

    if state != "1/1":
        delete_cmd = "kubectl delete pod " + name
        print("Executing " + delete_cmd + "...")
        subprocess.getoutput(delete_cmd)
