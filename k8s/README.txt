You should run scripts in the following order:

updateImage.sh
deployCluster.sh
start_ssh.sh
one_test.sh
collect_log.sh

Creating a new cluster can take long, thus we create a large number of pods
once, and reuse them throughout the test.
