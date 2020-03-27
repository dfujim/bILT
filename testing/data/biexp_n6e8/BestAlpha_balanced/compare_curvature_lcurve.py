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

# print available T1
for T1 in dfb['T1b']:
    print(T1)
  
# get the T1 we are working with 
T1 = float(input('T1 = '))

# get file and alphas
df = dfb.set_index('T1b').loc[T1]
alphab = df['lcurve']
filename = df['filename']

df = dfc.set_index('T1b').loc[T1]
alphac = df['lcurve']

# make object
I = ilt4sim('../'+filename)

# draw L curve
I.fit()
x,y = I.get_Lcurve()
alpha = I.get_alpha()

plt.figure()
plt.loglog(x,y,'-.',zorder=0)

alpha = alpha.astype(int)
b = alpha==int(alphab)
c = alpha==int(alphac)
plt.plot(x[c],y[c],'sC1',label=r'$\alpha_{curvature} = %.3g$' % alphac,zorder=1)
plt.plot(x[b],y[b],'^C3',label=r'$\alpha_{balanced} = %.3g$' % alphab,zorder=2)

plt.xlabel("$|| \Sigma ( K \mathbf{p} - \mathbf{y} ) ||$")
plt.ylabel("$|| \mathbf{p} ||$")
plt.title(r'Bi-Exp with $T_1^{(a)}=1$ and $T_1^{(b)}=%.2f$' % T1)
plt.legend()
plt.tight_layout()

outname = 'compare_curvature_lcurve_T1is%.2f' % T1
outname = outname.replace('.','p')
plt.savefig(outname+'.pdf')
plt.savefig(outname+'.jpg')
