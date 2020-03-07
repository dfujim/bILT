# Run analysis on data sets
# Derek Fujimoto
# Mar 2020

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bILT.testing.ilt4sim import ilt4sim
from tqdm import tqdm

# get all durations
files = np.array(glob.glob('*.yaml'))
durations = [(os.path.splitext(x)[0])[1:] for x in files]
durations = np.array(list(map(float,durations)))
idx = np.argsort(durations)
files = files[idx]
durations = durations[idx]

# run the solver
def draw(alpha,label):
    pmean = []
    pstd = []
    for f in tqdm(files): 
        laplace = ilt4sim(f)
        laplace.fit(alpha)
        z = laplace.z
        p = laplace.p
        
        # normalze
        p /= sum(p)
        
        # remove noise
        idx = (z>0.1) * (z<10)
        z = z[idx]
        p = p[idx]
        
        # stats
        u = sum(p*z)
        pmean.append(u)
        pstd.append(sum(p*(z-u)**2))
        
    pmean = np.array(pmean)
    pstd = np.array(pstd)

    # make T1 not 1/T1
    pstd /= pmean**2
    pmean = 1/pmean

    # draw
    fig = plt.semilogx(durations,pmean,'.-',label=label)
    plt.semilogx(durations,pmean+pstd,':',color=fig[0].get_color())
    plt.semilogx(durations,pmean-pstd,':',color=fig[0].get_color())
    plt.fill_between(durations,pmean+pstd,pmean-pstd,alpha=0.1,color=fig[0].get_color())
    plt.xlabel('Number of Probes Implanted')
    plt.ylabel(r'$\langle T_1 \rangle$ (s)')
    plt.title(r'Single Exp ($T_1=1$)')
    plt.legend()
    plt.tight_layout()

draw(0,r'$\alpha = 0$')
draw(1000,r'$\alpha = 10^3$')
