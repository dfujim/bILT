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
dfc = pd.read_csv('../BestAlpha/best_alpha.csv',comment='#')

# get rchi2 of each 
def get_rchi2(df):
    rchi2 = []
    for i in df.index:
        I = ilt4sim(os.path.join('..',df.loc[i,'filename']))
        rchi2.append(I.get_rchi2(df.loc[i,'lcurve']))
    return rchi2

rchi2b = get_rchi2(dfb)
rchi2c = get_rchi2(dfc)

# draw to compare 
plt.figure()
plt.semilogx(dfc['T1b'],rchi2c,'-o',label='Curvature Method')
plt.semilogx(dfb['T1b'],rchi2b,'-o',label='Balanced Method')
plt.xlabel(r'$T_1^{(b)}$ (s)')
plt.ylabel(r'$\chi^2/N$')
plt.title(r'Bi-Exp with $T_1^{(a)}=1$')
plt.legend()
plt.tight_layout()

plt.savefig('compare_curvature_rchi2.pdf')
plt.savefig('compare_curvature_rchi2.jpg')
