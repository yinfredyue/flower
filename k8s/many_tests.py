# Delete containers that fail to get ready.
# https://stackoverflow.com/a/40566922/9057530
import subprocess
import argparse
import time

parser = argparse.ArgumentParser(description='Run many tests')
parser.add_argument('--containers', '-c', nargs='+', default=[32], type=int, help="number of containers")
parser.add_argument('--staleness', '-s', nargs='+', type=int, help="a number of values, separated by space")
parser.add_argument('--rounds', '-r', nargs='+', default=[20], type=int)
parser.add_argument('--max_delay', nargs='+', default=[10], type=int)
parser.add_argument('--script', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--sleep_sec', type=int, required=True, help="Keep this the same as --request-timeout!!!")
parser.add_argument
args = parser.parse_args()

for c in args.containers:
    for s in args.staleness:
        for r in args.rounds:
            for d in args.max_delay:
                # Run test
                test_cmd = f'{args.script} {c} {s} {r} {d}'
                print("Running:", test_cmd)
                subprocess.getoutput(test_cmd)

                # Wait task finish
                time.sleep(args.sleep_sec)

                # Collect log
                subprocess.getoutput(f"collectLog.sh {c}")

                # Rename directory
                new_log_dir_name = f"log-n{c}-s{s}-d{d}-r{r}-{args.name}"
                subprocess.getoutput("mv log/ {new_log_dir_name}/")

                # Remove log
                subprocess.getoutput("clean_log.sh")


                