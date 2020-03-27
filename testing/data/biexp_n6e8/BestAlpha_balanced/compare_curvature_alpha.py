# Compare the results of balanced vs curvature L curve corner detection methods
# Balanced: min(x**2 * y)
# Derek Fujimoto
# Mar 2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bILT.testing.ilt4sim import ilt4sim

# DRAW ALPHA

# load files 
dfb = pd.read_csv('best_alpha.csv',comment='#')
dfc = pd.read_csv('../BestAlpha/best_alpha.csv',comment='#')

# draw to compare 
plt.figure()
plt.semilogx(dfc['T1b'],dfc['lcurve'],'-o',label='Curvature Method')
plt.semilogx(dfb['T1b'],dfb['lcurve'],'-o',label='Balanced Method')
plt.xlabel(r'$T_1^{(b)}$ (s)')
plt.ylabel(r'$\alpha_{opt}$')
plt.title(r'Bi-Exp with $T_1^{(a)}=1$')
plt.legend()
plt.tight_layout()

plt.savefig('compare_curvature_alpha.pdf')
plt.savefig('compare_curvature_alpha.jpg')
