import time
import sys


# Print time diff to destination (defaults to stdout)
def log_time(t0, dest=sys.stdout):
    print("t = %.3f sec" % (time.time()-t0))


# Load list of target words and return it as a list
def load_targets(path):
    with open(path, 'r', encoding='utf-8') as f_in:
            targets = [line.strip() for line in f_in]
    return targets
