# Draw optimal chisquared
# Derek Fujimoto
# Mar 2020

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bILT.testing.ilt4sim import ilt4sim
from tqdm import tqdm

# get file names
df = pd.read_csv('best_alpha.csv',comment='#')
df.sort_values('T1b',inplace=True)

# get the optimal chisquared for each alpha
chi_opt = {'scurve':[],'lcurve':[],'gcv':[]}

for i in tqdm(df.index):
    I = ilt4sim(os.path.join('..',df.loc[i,'filename']))
    
    for k in chi_opt.keys():
        chi_opt[k].append(I.get_rchi2(df.loc[i,k]))

# draw
T1 = df['T1b']

plt.semilogx(T1,chi_opt['scurve'],'-o',label='S Curve')
plt.semilogx(T1,chi_opt['lcurve'],'-o',label='L Curve')
plt.semilogx(T1,chi_opt['gcv'],'-o',label='GCV')

plt.xlabel(r'$T_1^{(b)}$ (s)')
plt.ylabel(r'$\chi^2/N$')
plt.title(r'Bi Exp with $T_1^{(a)}=1$')
plt.legend(fontsize='x-small')
plt.tight_layout()
