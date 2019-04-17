import argparse
import os
import pandas as pd
import pickle
import subprocess
import time

from itertools import product
from numpy import arange, linspace, pi


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
        time.sleep(30)


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
    parser.add_argument(
        "-f", metavar="out_file", type=str, nargs=1, default="shanahan_params.csv",
        help="The file in which to save the data"
    )
    c_args = parser.parse_args()
    data_dir = c_args.d[0]
    out_file = c_args.f[0]

    if c_args.t[0] == "calc":
        template_file = "submit_scripts/template.pbs"
        args = [[α, β, num, α, β, out_file, num]
                for α, β, num
                in product(linspace(0.0, 1.0, 100),
                           linspace(0.0, 1.0, 100),
                           arange(10))]
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
        template_file = "submit_scripts/tmp_template.pbs"
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
        time.sleep(30)
    print("Done")

    if c_args.t[0] in ["calc", "fix"]:
        subprocess.Popen(["qsub", "submit_scripts/hizanidis.pbs"])


if __name__ == "__main__":
    main()
