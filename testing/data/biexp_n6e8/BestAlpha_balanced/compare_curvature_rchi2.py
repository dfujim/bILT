# Compare the results of balanced vs curvature L curve corner detection methods
# Balanced: min(x**2 * y)
# Derek Fujimoto
# Mar 2020

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bILT.testing.ilt4sim import ilt4sim

# DRAW rCHI2

# load files 
dfb = pd.read_csv('best_alpha.csv',comment='#')
dfc = pd.read_csv('../BestAlpha_curvature/best_alpha.csv',comment='#')

# get rchi2 of each 
def get_rchi2(df,col):
    rchi2 = []
    for i in df.index:
        I = ilt4sim(os.path.join('..',df.loc[i,'filename']))
        rchi2.append(I.get_rchi2(df.loc[i,col]))
    return rchi2

rchi2b = get_rchi2(dfb,'lcurve')
rchi2c = get_rchi2(dfc,'lcurve')
rchi2s = get_rchi2(dfc,'scurve')
rchi2g = get_rchi2(dfc,'gcv')

# draw to compare 
plt.figure()

plt.semilogx(dfc['T1b'],rchi2c,'-oC0',label='L-curve$_{curvature}$')
plt.semilogx(dfb['T1b'],rchi2b,'-oC0',mfc='none',label='L-curve$_{balanced}$')
plt.semilogx(dfc['T1b'],rchi2s,'-oC1',label='S-curve')
plt.semilogx(dfc['T1b'],rchi2g,'-oC2',label='GCV')

plt.xlabel(r'$T_1^{(b)}$ (s)')
plt.ylabel(r'$\chi^2/N$')
plt.title(r'Bi-Exp with $T_1^{(a)}=1$')
plt.legend()
plt.tight_layout()

plt.savefig('compare_curvature_rchi2.pdf')
plt.savefig('compare_curvature_rchi2.jpg')
