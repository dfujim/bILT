# Run analysis on data sets
# Derek Fujimoto
# Mar 2020

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bILT.testing.ilt4sim import ilt4sim
from bILT.testing.discriminator import discriminator
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
    width = []
    height = []
    loc = []
    for f in tqdm(files): 
        laplace = ilt4sim(f)
        laplace.fit(alpha)
        z = laplace.z
        p = laplace.p
        
        # get T1 not 1/T1
        z = 1/z
        
        # normalze
        p /= z
        p /= sum(p)
        
        # stats
        l,h,w = discriminator(z,p)
        
        # only keep the largest peak. Hopefully this is the one we want.
        i = np.argmax(h)
        
        loc.append(l[i])
        height.append(h[i])
        width.append(w[i])
    
    height = np.array(height)
    width = np.array(width)
    loc = np.array(loc)

    # draw
    fig = plt.semilogx(durations,loc,'.-',label=label)
    plt.semilogx(durations,loc+width,':',color=fig[0].get_color())
    plt.semilogx(durations,loc-width,':',color=fig[0].get_color())
    plt.fill_between(durations,loc+width,loc-width,alpha=0.1,color=fig[0].get_color())
    plt.xlabel('Number of Probes Implanted')
    plt.ylabel(r'$\langle T_1 \rangle$ (s)')
    plt.title(r'Single Exp ($T_1=1$)')
    plt.legend()
    plt.tight_layout()

draw(0,r'$\alpha = 0$')
draw(1000,r'$\alpha = 10^3$')
