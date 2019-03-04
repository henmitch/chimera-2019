import argparse
import os
import pandas as pd
import pickle
import subprocess
import time

from itertools import product
from numpy import linspace, pi


def qsub(template, args):
    submit_script = template.format(*args)
    with open("submit.pbs", "w") as f:
        f.write(submit_script)
    subprocess.Popen(["qsub", "-q", "shortq", "submit.pbs"])


def auto_run(template, args, max_jobs=200):
    jobs = max(0, int(subprocess.check_output(
        'qstat | wc -l', shell=True)) - 2)
    print("Jobs: ", jobs)
    for params in args:
        while jobs >= max_jobs:
            time.sleep(60)
            jobs = max(0, int(subprocess.check_output(
                'qstat | wc -l', shell=True)) - 2)
        print("Jobs: ", jobs)
        qsub(template, *params)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-run some stuff with PBS"
    )
    parser.add_argument(
        "t", metavar="type", type=str, nargs=1,
        help="The type of job to run.  Either 'calc' or 'plot'."
    )
    c_args = parser.parse_args()

    if c_args.t[0] == "calc":
        template_file = "template.pbs"
        args = [[α, β, α, β]
                for α, β
                in product(linspace(0.0, 3.2, 80), linspace(0.0, 0.8, 40))
                if f"{α:.03f}-{β:.03f}.pkl" not in os.listdir("../../data")]
    elif c_args.t[0] == "plot":
        template_file = "mean_plots_template.pbs"
        with open("../data/hizanidis_params.pkl", "rb") as f:
            params = pickle.load()
        good = params[params["max_phase"] <= 2*pi]
        args = [2*[f"{α:.03f}-{β:.03f}.pkl"]
                for row, [α, β]
                in good[["alpha", "beta"]].iterrows()]
    else:
        print("Please use 'calc' or 'plot' as the type of job.")
        exit(1)

    with open(template_file, "r") as f:
        template = f.read()

    auto_run(template, args)

    jobs = max(0, int(subprocess.check_output(
        'qstat | wc -l', shell=True)) - 2)
    while jobs != 0:
        jobs = max(0, int(subprocess.check_output(
            'qstat | wc -l', shell=True)) - 2)
        time.sleep(10)
    print("Done")

    if c_args.t[0] == "calc":
        subprocess.Popen(["qsub", 'hizanidis.pbs'])


if __name__ == "main":
    main()
