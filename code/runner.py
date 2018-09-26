import subprocess

from itertools import product
from numpy import linspace
from time import sleep

with open("template.pbs", "r") as f:
    template = f.read()


def qsub(α, β):
    submit_script = template.format(α, β, α, β)
    with open("submit.pbs", "w") as f:
        f.write(submit_script)
    subprocess.Popen(["qsub", "submit.pbs"])


def auto_run(max_jobs=200):
    jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
    print("Jobs: ", jobs)
    for params in product(linspace(0, 1.6, 50), linspace(0, 0.4, 50)):
        α, β = params
        while jobs >= max_jobs:
            time.sleep(60)
            jobs = max(0, int(subprocess.check_output('qstat | wc -l', shell=True)) - 2)
            print("Jobs: ", jobs)
        qsub(α, β)
        time.sleep(10)


auto_run()
