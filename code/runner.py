import os
import subprocess
import time

from itertools import product
from numpy import linspace

with open("template.pbs", "r") as f:
    template = f.read()


def qsub(α, β):
    submit_script = template.format(α, β, α, β)
    with open("submit.pbs", "w") as f:
        f.write(submit_script)
    subprocess.Popen(["qsub", "-q", "shortq", "submit.pbs"])


def auto_run(max_jobs=200):
    jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
    print("Jobs: ", jobs)
    for params in product(linspace(0, 1.6, 80), linspace(0, 0.4, 20)):
        α, β = params
        if f"{α:.03f}-{β:.03f}.pkl" not in os.listdir("../data"):
            print("Jobs: ", jobs)
            print(α, β)
            jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
            while jobs >= max_jobs:
                time.sleep(60)
                jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
            qsub(α, β)
            time.sleep(10)


auto_run()

jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
while jobs != 0:
    jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
    time.sleep(10)
print("Done")
subprocess.Popen(["qsub", 'hizandis.pbs'])
