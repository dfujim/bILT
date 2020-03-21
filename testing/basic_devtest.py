# basic dev testing the ilt 
# Derek Fujimoto
# Feb 2020

from bILT.src.ilt import *
from bfit.fitting.functions import pulsed_exp 
import bdata as bd
import numpy as np

x,y,dy = bd.bdata(40214, year=2009).asym('c') 
T1 = np.logspace(np.log(0.01 * 1.2096), np.log(100.0 * 1.2096), 100)
alpha = np.logspace(2, 5, 50)

f = pulsed_exp(1.21,4)
fn = lambda x,w: f(x,w,1) 
I = ilt(x,y,dy,fn,T1,nproc=4)
I.fit(alpha)

# ~ plt.figure()
# ~ I.draw_fit(10)

# ~ plt.figure()
# ~ I.draw_weights(10)

# ~ plt.figure()
# ~ I.draw_logdist(10)

# ~ plt.figure()
# ~ I.draw_Lcurve()

# ~ plt.figure()
# ~ I.draw_Scurve(0.1)

# ~ plt.figure()
# ~ I.draw_Ccurve()

# ~ alpha,curve = I.get_Lcurve_curvature()
# ~ plt.figure()
# ~ plt.semilogx(alpha,curve,'.-')

# ~ plt.figure()
# ~ I.draw_gcv()


# ~ I.write('test.yaml')
