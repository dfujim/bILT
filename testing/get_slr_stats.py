# Get the statistics of slr runs during a single year
# Derek Fujimoto
# Mar 2020

import glob
import os
import bdata as bd
import numpy as np
from scipy.optimize import curve_fit
import tqdm

year = 2018

# get BNMR data
files = glob.glob(os.environ['BNMR_ARCHIVE']+'/%d/*.msr'%year)
runs = map(lambda x : int(os.path.splitext(os.path.basename(x))[0]),files)

# get duration and histogram counts of runs
duration = []
counts = []
hist = ['B+','B-','F+','F-']
for r in tqdm.tqdm(runs,total=len(files)):
    
    if r <= 40000:
        continue
        
    b = bd.bdata(r,year)
    if b.mode == '20':
        duration.append(b.duration)
        hcount = 0
        for h in hist:
            hcount += sum(b.hist[h].data)
        
        counts.append(hcount)
        
# histogram the data
h_dur,b_dur = np.histogram(duration,bins=np.arange(0,3600,60))
h_cnt,b_cnt = np.histogram(counts,bins=np.arange(1e7,1e9,1e7))

# bin centers
b_dur = (b_dur[1:]+b_dur[:-1])/2
b_cnt = (b_cnt[1:]+b_cnt[:-1])/2

# print stats
outstr = '\n'.join(['',
                    'Year: %d' % year,
                    '',
                    '===================================',
                    'Duration Statistics (min)',
                    'Mean:               %g' % (np.mean(duration)/60),
                    'Standard deviation: %g' % (np.std(duration)/60),
                    '',
                    '===================================',
                    'Histogram Counts Statistics (1e8)',
                    'Mean:               %g' % (np.mean(counts)/1e8),
                    'Standard deviation: %g' % (np.std(counts)/1e8),
                    '',
                    'Total Number of SLR runs: %d' % len(duration),
                    '\n'
                    ])

print(outstr)

# draw histograms
plt.figure()
plt.plot(b_dur/60,h_dur)
plt.xlabel('Run Duration (min)')
plt.ylabel('Counts/min')
plt.title('SLR Run Duration (%d)' % year)

plt.figure()
plt.plot(b_cnt/1e8,h_cnt)
plt.xlabel('Sum of Histogram Counts ($10^8$)')
plt.ylabel('Counts/1e7')
plt.title('SLR Run Statistics (%d)' % year)
