# Run analysis on data sets
# Derek Fujimoto
# Mar 2020

import os, glob
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from bILT.testing.ilt4sim import ilt4sim
import datetime

# Turn interactive plotting off
plt.ioff()

# get all yaml files
files = np.array(glob.glob('*.yaml'))

# settings
results = {'scurve':[],'lcurve':[],'gcv':[],'filename':[]}
outfile = 'best_alpha.csv'
scurve_threshold = 0.1
nproc = 8

# run the analyzer
for file in files:
    print(file)
    I = ilt4sim(file,nproc=nproc)
    I.fit()

    # draw
    plt.figure()
    I.draw_Lcurve()
    plt.tight_layout()
    plt.savefig('Plots/Lcurve_%s.pdf' % os.path.splitext(file)[0])

    plt.figure()
    I.draw_Scurve(threshold=scurve_threshold)
    plt.tight_layout()
    plt.savefig('Plots/Scurve_%s.pdf' % os.path.splitext(file)[0])

    plt.figure()
    I.draw_gcv()
    data = plt.gca().lines[0].get_ydata()
    alph = plt.gca().lines[0].get_xdata()

    lo = min(data)
    hi = data[0]
    spacer = (hi-lo)*0.05
    plt.ylim(lo-spacer,hi+spacer)
    plt.tight_layout()
    plt.savefig('Plots/Gcurve_%s.pdf' % os.path.splitext(file)[0])

    plt.close('all')
    
    results['filename'].append(file)
    results['scurve'].append(I.get_Scurve_opt(threshold=scurve_threshold))
    results['lcurve'].append(I.get_Lcurve_opt())
    results['gcv'].append(alph[np.argmin(data)])

df = pd.DataFrame(results)

def file2N(name):
    name = os.path.splitext(name)[0]
    return float(name[1:])

df['N'] = df['filename'].apply(file2N)

with open(outfile,'w') as fid:
    lines = ['# Results of fetching the best alpha for all methods',
             '# S curve threshold = %f' % scurve_threshold,
             '#',
             '# %s' % str(datetime.datetime.now()),
             '#\n']
    fid.write('\n'.join(lines))
    
df.to_csv(outfile,mode='a',index=False)
    
    
