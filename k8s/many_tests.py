# Delete containers that fail to get ready.
# https://stackoverflow.com/a/40566922/9057530
import subprocess
import argparse
import time

# python3 many_tests.py --containers 32 --staleness 1 2 3 4 5 10 20 30 1000 2000 --rounds 30 --script ./one_test_shake.sh --name shakespeare_lstm
# python3 many_tests.py --containers 32 --staleness 1 --rounds 1 --max_delay 10 --script ./one_test_cifar.sh --name quickstart_pytorch

# Keep running even after close terminal:
# https://unix.stackexchange.com/a/4006
# $ nohup ... &
# $ disown

# $ nohup bash -c "python3 many_tests.py --containers 32 --staleness 1 2 5 10 25 50 --rounds 50 --max_delay 10 --script ./one_test_shake.sh --name shakespeare_lstm; python3 many_tests.py --containers 16 --staleness 1 2 5 10 25 50 --rounds 50 --max_delay 10 --script ./one_test_cifar.sh --name quickstart_pytorch" & 
# $ disown
parser = argparse.ArgumentParser(description='Run many tests')
parser.add_argument('--containers', '-c', nargs='+', default=[32], type=int, help="number of containers")
parser.add_argument('--staleness', '-s', nargs='+', type=int, help="a number of values, separated by space")
parser.add_argument('--rounds', '-r', nargs='+', default=[20], type=int)
parser.add_argument('--max_delay', nargs='+', default=[10], type=int)
parser.add_argument('--script', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

for c in args.containers:
    for s in args.staleness:
        for r in args.rounds:
            for d in args.max_delay:
                # Run test
                test_cmd = f'bash {args.script} {c} {s} {r} {d}'
                print("Running:", test_cmd)
                output = subprocess.getoutput(test_cmd)
                print("Res: ", output)

                # Collect log
                print("Start collecting log...")
                output = subprocess.getoutput(f"bash collectLog.sh {c} {args.name}")
                print(output)

                # Rename directory
                new_log_dir_name = f"log-n{c}-s{s}-d{d}-r{r}-{args.name}"
                subprocess.getoutput(f"mv log/ {new_log_dir_name}/")

                # Remove log
                print("Cleaning log...")
                subprocess.getoutput("bash clean_log.sh")



