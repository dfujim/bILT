import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import cpuinfo
from ROOT import TFile, TVectorD


# open the control file example
with open("../config.yaml", "r") as fh:
    ctl = yaml.load(fh, Loader=yaml.SafeLoader)


# get info on the current cpu
cpu = cpuinfo.get_cpu_info()
cpu_name = (
    cpu["brand"]
    .replace(" ", "_")
    .replace("(", "_")
    .replace(")", "_")
    .replace("__", "_")
)

# hold the simulation timing results
sim_probes = []
sim_time = []
sim_time_error = []

# loop over n
n_probes = np.logspace(3, 8, 6)
for n in n_probes:
    # update n in the control file
    ctl["n"] = int(n)
    fname = "../tmp.yaml"
    with open(fname, "w") as fh:
        yaml.dump(ctl, fh, default_flow_style=False)
    # setup the system command
    cmd = "cd ../ && ./data_generator_pp %s" % fname.split("/")[-1]
    #
    tmp_time = []
    # repeat x times
    for i in range(10):
        os.system(cmd)
        root_file = "../" + ctl["output"]
        rf = TFile.Open(root_file)
        # sim_probes.append(rf.mc_stats[0])
        tmp_time.append(rf.mc_stats[1])
        rf.Close()
        os.system("rm %s" % root_file)
    #
    sim_probes.append(n)
    sim_time.append(np.average(tmp_time))
    sim_time_error.append(np.std(tmp_time, ddof=1))
    # remove tmp control file
    os.system("rm %s" % fname)


# save the results
with open("benchmark/" + cpu_name + ".dat", "w") as fh:
    fh.write("#n\tTime (s)\tTimeError (s)\n")
    for sp, st, ste in zip(sim_probes, sim_time, sim_time_error):
        fh.write("%.8e\t%.8e\t%.8e\n" % (sp, st, ste))
