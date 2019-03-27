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
    with open("submit_scripts/submit.pbs", "w") as f:
        f.write(submit_script)
    subprocess.Popen(["qsub", "-q", "shortq", "submit_scripts/submit.pbs"])


def auto_run(template, args, max_jobs=150):
    for params in args:
        jobs = max(0, int(subprocess.check_output(
            "qstat | grep ' [RQ] ' | wc -l", shell=True)) - 2)
        print("Jobs: ", jobs)
        while jobs >= max_jobs:
            time.sleep(30)
            jobs = max(0, int(subprocess.check_output(
                "qstat | grep ' [RQ] ' | wc -l", shell=True)) - 2)
        qsub(template, params)
        time.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-run some stuff with PBS"
    )
    parser.add_argument(
        "t", metavar="type", type=str, nargs=1,
        help="The type of job to run.  Either 'calc', 'plot', or 'fix'."
    )
    parser.add_argument(
        "-d", metavar="data_dir", type=str, nargs=1, default="../data",
        help="The location to which to save the data"
    )
    c_args = parser.parse_args()
    data_dir = c_args.d[0]

    if c_args.t[0] == "calc":
        template_file = "submit_scripts/template.pbs"
        args = [[α, β, α, β, data_dir]
                for α, β
                in product(linspace(0.0, 0.9, 80), linspace(0.0, 0.9, 80))
                if f"{α:.05f}-{β:.05f}.pkl" not in os.listdir(data_dir)]
    elif c_args.t[0] == "plot":
        template_file = "submit_scripts/plots_template.pbs"
        with open(f"{data_dir}/hizanidis_params.pkl", "rb") as f:
            params = pickle.load(f)
        good = params[params["max_phase"] <= 2*pi]
        args = [[f"{α:.05f}_{β:.05f}", f"{α:.05f}-{β:.05f}.pkl"]
                for row, [α, β]
                in good[["alpha", "beta"]].iterrows()
                if all([(f"{front}-{α:.05f}-{β:.05f}.png"
                         not in os.listdir("figure"))
                        for front in [
                    "overhead",
                    "cos_mean",
                    "by_cortex_cos_mean",
                    "order"
                    "by_cortex_variance",
                    "chi",
                    "all"
                ]])]
    elif c_args.t[0] == "fix":
        template_file = "submit_scripts/fix_template.pbs"
        args = [(i.rstrip(), i.rstrip())
                for i
                in os.listdir(data_dir)]
    elif c_args.t[0] == "network":
        template_file = "submit_scripts/network_template.pbs"
        args = [(t, t)
                for t
                in range(1000)]
    else:
        print("Please use 'calc' or 'plot' as the type of job.")
        exit(1)

    with open(template_file, "r") as f:
        template = f.read()

    auto_run(template, args)

    jobs = max(0, int(subprocess.check_output(
        "qstat | wc -l", shell=True)) - 2)
    while jobs != 0:
        jobs = max(0, int(subprocess.check_output(
            "qstat | wc -l", shell=True)) - 2)
        time.sleep(10)
    print("Done")

    if c_args.t[0] in ["calc", "fix"]:
        subprocess.Popen(["qsub", "submit_scripts/hizanidis.pbs"])


if __name__ == "__main__":
    main()
