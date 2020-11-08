FROM pytorchcpu

# Use pytorchcpu as base image. Then copies /src into the container.
# This avoids rebuilding dependencies.

# To build this image:
# $ cd flower/
# $ docker build --tag flowerpytorch:latest -f ./dockerFiles/flowerPytorch.Dockerfile .

# SSH related
# Useful reference:
# Send command using ssh: https://malcontentcomics.com/systemsboy/2006/07/send-remote-commands-via-ssh.html
# Without prompt: https://unix.stackexchange.com/q/229124
# Start SSH server: https://stackoverflow.com/a/32178958/9057530
# How CMD works: https://stackoverflow.com/a/42219138/9057530
# Difference between RUN and CMD: https://stackoverflow.com/q/37461868/9057530
RUN apt-get install openssh-server -y
# -f provides the filename to avoids prompt. Check man page.
RUN ssh-keygen -t rsa -P "" -f "/root/.ssh/id_rsa"
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
EXPOSE 22 

WORKDIR /app/src/py/

COPY ./src/ /app/src/
ENTRYPOINT service ssh start && bash

# To avoid the prompt of trust, when doing ssh, add an argument:
# ssh -o StrictHostKeyChecking=no root@ip
