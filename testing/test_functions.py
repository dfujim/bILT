# Test the ILT procedure on data with a number of different input functions
# Derek Fujimoto
# Feb 2020

import numpy as np
import matplotlib.pyplot as plt
from data_generator import data_generator
from scipy.optimize import curve_fit
from bfit.fitting.functions import pulsed_exp, pulsed_strexp
from bILT import ilt
import bdata as bd

# tesing object
class test(object):

    # setup 
    alpha = np.logspace(2,5,100)

    def __init__(self, p0=0.3, n=1e6):
        self.g = data_generator(tmax=10)
        self.p0 = p0
        self.n = n
    
        g = self.g
        self.pexp = pulsed_exp(lifetime=g.lifetime, pulse_len=g.pulse_len)
        self.sexp = pulsed_strexp(lifetime=g.lifetime, pulse_len=g.pulse_len)
        self.T1 = np.logspace(np.log10(g.lifetime*0.01), np.log10(g.lifetime*100), 1000)
        self.kernel_fn = lambda x,w: self.pexp(x,w,1)
        
    def run(self,relax_fn,fit_fn,rebin,**fitpar):
        g = self.g
        self.fit_fn = fit_fn
        
        # simulate data
        g.gen_counts(relax_fn,p0=self.p0,n=self.n)
        self.t,self.a,self.da = g.asym(rebin=rebin)
        
        # fit with expected function
        self.par,cov = curve_fit(fit_fn, g.t, g.a, sigma=g.da, absolute_sigma=True,**fitpar)
        self.std = np.diag(cov)**0.5
        
        # do ILT with unbinned data
        self.I = ilt(g.t, g.a, g.da, self.kernel_fn)
        self.I.fit(1000, self.T1, maxiter=1e4)
        
    def draw(self):
        
        self.I.draw()
        
        # draw
        fig = plt.gcf()
        ax1,ax2 = fig.axes
        ax1.clear()
        ax1.errorbar(self.t,self.a,self.da,fmt='k.',zorder=0)
        ax1.plot(self.I.x, self.I.fity, zorder=1, 
                 label=r"ILT ($\alpha=%d$)" % int(self.I.alpha),color='r',lw=2)
        ax1.plot(self.t, self.fit_fn(self.t, *self.par), zorder=2,
                 label="Functional Fit",color='b',lw=2)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Asymmetry')
        ax2.set_xlabel(r'1/T1 (s$^{-1}$)')
        ax1.legend()
        ax1.set_title("")
        ax2.set_title("")

        return (fig,ax1,ax2)

# EXPONENTIAL ================================================================ 

if 0:
    print("Running EXPONENTIAL")
    
    # setup
    T1_tst = 2  # T1 in this test
    f = lambda t: np.exp(-t/T1_tst)
    fit = pulsed_exp(lifetime=bd.life.Li8,pulse_len=4)

    # run and draw
    test1 = test()
    test1.run(f,fit,5)
    fig,ax1,ax2 = test1.draw()

    ax2.axvline(1/T1_tst,color='k',ls='--',lw=2,label='True 1/$T_1$')
    ax2.axvline(test1.par[0],color='r',ls=':',lw=2,label='Fitted 1/$T_1$')
    fig.suptitle("Exponential Relaxation")
    ax2.legend()
    
# BIEXPONENTIAL - well separated ==============================================

if 0:
    print("Running BIEXPONENTIAL - welll separated")
    # setup
    T1_tst = (1,10)  # T1 in this test, note the amplitudes below
    f = lambda t: 0.3*np.exp(-t/T1_tst[0]) + 0.7*np.exp(-t/T1_tst[1])
    pexp = pulsed_exp(lifetime=bd.life.Li8,pulse_len=4)
    fit = lambda t,lam1,lam2,amp,frac : amp*(frac*pexp(t,lam1,1) + \
                                            (1-frac)*pexp(t,lam2,1))

    # simulate data
    test2 = test(n=1e7)
    test2.run(f,fit,5,bounds=[0,[np.inf,np.inf,1,1]])
    fig,ax1,ax2 = test2.draw()

    fig.suptitle("Bi-Exponential Relaxation (well separated)")
    ax2.axvline(1/T1_tst[0],color='k',ls='--',lw=2,label='True 1/$T_1$')
    ax2.axvline(1/T1_tst[1],color='k',ls='--',lw=2)

    ax2.axvline(test2.par[0],color='r',ls=':',lw=2,label='Fitted 1/$T_1$')
    ax2.axvline(test2.par[1],color='r',ls=':',lw=2)
    ax2.legend()

# BIEXPONENTIAL - poorly separated ============================================

if 0:
    print("Running BIEXPONENTIAL - poorly separated")
    # setup
    T1_tst = (5,10)  # T1 in this test, note the amplitudes below
    f = lambda t: 0.3*np.exp(-t/T1_tst[0]) + 0.7*np.exp(-t/T1_tst[1])
    pexp = pulsed_exp(lifetime=bd.life.Li8,pulse_len=4)
    fit = lambda t,lam1,lam2,amp,frac : amp*(frac*pexp(t,lam1,1) + \
                                            (1-frac)*pexp(t,lam2,1))

    # simulate data
    print("\tWarning: high-stats run, will take a while")
    test3 = test(n=1e8,p0=0.3)
    test3.run(f,fit,5,bounds=[0,[np.inf,np.inf,1,1]])
    fig,ax1,ax2 = test3.draw()

    fig.suptitle("Bi-Exponential Relaxation (poor separation)")
    ax2.axvline(1/T1_tst[0],color='k',ls='--',lw=2,label='True 1/$T_1$')
    ax2.axvline(1/T1_tst[1],color='k',ls='--',lw=2)

    ax2.axvline(test3.par[0],color='r',ls=':',lw=2,label='Fitted 1/$T_1$')
    ax2.axvline(test3.par[1],color='r',ls=':',lw=2)
    ax2.legend()

# TRIEXPONENTIAL ==============================================================

if 0:
    print("Running TRIEXPONENTIAL")
    # setup
    T1_tst = (0.1,1,10)  # T1 in this test, note the amplitudes below
    f = lambda t: 0.3*np.exp(-t/T1_tst[0]) + 0.3*np.exp(-t/T1_tst[1]) + 0.4*np.exp(-t/T1_tst[2])
    pexp = pulsed_exp(lifetime=bd.life.Li8,pulse_len=4)
    fit = lambda t,lam1,lam2,lam3,amp1,amp2,amp3 : pexp(t,lam1,amp1) + \
                                                   pexp(t,lam2,amp2) + \
                                                   pexp(t,lam3,amp3)

    # simulate data
    test4 = test(n=1e7,p0=0.7)
    test4.run(f,fit,5,bounds=[0,[np.inf,np.inf,np.inf,1,1,1]])
    fig,ax1,ax2 = test4.draw()

    fig.suptitle("Tri-Exponential Relaxation")
    ax2.axvline(1/T1_tst[0],color='k',ls='--',lw=2,label='True 1/$T_1$')
    ax2.axvline(1/T1_tst[1],color='k',ls='--',lw=2)
    ax2.axvline(1/T1_tst[2],color='k',ls='--',lw=2)

    ax2.axvline(test4.par[0],color='r',ls=':',lw=2,label='Fitted 1/$T_1$')
    ax2.axvline(test4.par[1],color='r',ls=':',lw=2)
    ax2.axvline(test4.par[2],color='r',ls=':',lw=2)
    ax2.legend()

# STR EXPONENTIAL ==============================================================

if 1:
    print("Running STRETCHED EXPONENTIAL")
    # setup
    T1_tst = 5  # T1 in this test, note the amplitudes below
    beta = 0.5
    f = lambda t: np.exp(-(t/T1_tst)**beta)
    fit = pulsed_strexp(lifetime=bd.life.Li8,pulse_len=4)
                                                   
    # simulate data
    test5 = test(n=1e7,p0=0.3)
    test5.run(f,fit,5,bounds=[0,[np.inf,1,1]])
    fig,ax1,ax2 = test5.draw()

    fig.suptitle(r"Root Exponential Relaxation ($\beta=0.5)$")
    ax2.axvline(1/T1_tst,color='k',ls='--',lw=2,label='True 1/$T_1$')
    ax2.axvline(test5.par[0],color='r',ls=':',lw=2,label='Fitted 1/$T_1$')
    ax2.legend()

