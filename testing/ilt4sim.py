# ILT solver with shortcuts for running on simulated data 
# Derek Fujimoto
# Mar 2020

import os, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bILT import ilt
from bfit.fitting.functions import pulsed_exp, pulsed_strexp
from scipy.optimize import curve_fit
from bILT.testing.discriminator import discriminator

class ilt4sim(ilt):
    
    # general settings
    asym_type = 'Ac'
    asym_err = 'd'+asym_type
    alpha = np.logspace(1,5,50)
    nT1 = 1000
    
    def __init__(self,filename,nproc=4):
        """
            filename:   yaml or csv or without extension
            nproc:      number of processors
        """
        
        # get file names
        filename = os.path.splitext(filename)[0]
        yamlfile = filename + '.yaml'
        csvfile = filename + '.csv'
        
        # get run settings
        with open(yamlfile, 'r') as fid:
            run_settings = yaml.safe_load(fid)
    
        # get run data
        df = pd.read_csv(csvfile, comment='#')
        df = df[['t',self.asym_type,self.asym_err]]

        # drop bad rows and errors of zero
        df[self.asym_err].loc[df[self.asym_err] == 0] = np.nan
        df.dropna(inplace=True)

        # make kernel and fit functions
        lifetime = run_settings['lifetime (s)']
        pulse_len = run_settings['beam pulse (s)']
        self.pexp = pulsed_exp(pulse_len = pulse_len, lifetime  = lifetime)
        self.sexp = pulsed_strexp(pulse_len = pulse_len, lifetime  = lifetime)
           
        # generate the T1 range
        log10_T1_min = np.log10(lifetime * 1e-2)
        log10_T1_max = np.log10(lifetime * 1e2)
        self.T1 = np.logspace(log10_T1_min, log10_T1_max, self.nT1, base=10.0)
                          
        # make ilt object
        super().__init__(df['t'].values, 
                      df[self.asym_type].values, 
                      df[self.asym_err].values, 
                      lambda x, w : self.pexp(x, w, 1),
                      self.T1,
                      nproc)

    def _rebin(self, t, x, dx, rebin):
        """
            Rebin array x with weights 1/dx**2 by factor rebin.
            
            Inputs: 
                xdx = [t,x,dx]
                rebin = int
            Returns [t,x,dx] after rebinning. 
        """
        
        rebin = int(rebin)
        
        
        # easy end condition
        if rebin <= 1:
            return (t,x,dx)
        
        # Rebin Discard unused bins
        lenx = len(x)
        x_rebin = []
        dx_rebin = []
        
        # rebin times
        t_rebin = [np.mean(t[i:i+rebin]) for i in range(0, len(t), rebin)]
        
        # avoid dividing by zero
        dx[dx==0] = np.inf
        
        # weighted mean
        for i in np.arange(0,lenx,rebin):
            w = 1./dx[i:i+rebin-1]**2
            wsum = np.sum(w)
            
            if wsum == 0:
                x_rebin.append(np.mean(x[i:i+rebin-1]))
                dx_rebin.append(np.std(x[i:i+rebin-1]))
            else:
                x_rebin.append(np.sum(x[i:i+rebin-1]*w)/wsum)
                dx_rebin.append(1./wsum**0.5)
        return np.array([t_rebin,x_rebin,dx_rebin])

    def fit(self,alpha=None, maxiter=10000):
        """
            If inputs are None, use defaults
        """

        if alpha is None: 
            alpha = self.alpha
        
        # run and draw diagnostics
        super().fit(alpha, maxiter)
    
    def discriminate(self,alpha,threshold, crop_distance=10, draw=False):
        """ 
            Apply discriminator method of finding the peaks
            
            returns: [location_of_max, height_of_max, width_at_threshold]
        """
        
        # get weights and T1
        p = self.get_weights(alpha)*self.lamb
        p /= sum(p)
        T1 = 1/self.lamb
        
        
        # get
        return discriminator(T1, p, threshold, crop_distance, draw)
    
    def draw(self,alpha=None,rebin=1,fitfn='exp'):
        """
            Draw fit or range of fits. 
            
            alpha_opt:  if None draw:
                            alpha v chi
                            alpha v dchi/dalpha
                            L-curve
                        else draw:
                            data & fit
                            distribution 
            rebin:  rebinning of data when drawing 
            fitfn:  one of "exp", "biexp", or "strexp" to determine which fit function to use
            fig:    optional figure handle for redrawing when alpha_opt != None
            
            returns: (p, fity, chi2)
            
                p:      array of unnormalized weights
                fity:   array of final fit function points
        """
            
        # draw things for a single alpha only 
        if alpha is not None:
            
            # get opt data 
            fity = self.get_fit(alpha)
            p = self.get_weights(alpha)
            chi2 = self.get_chi2(alpha)
            rchi2 = self.get_rchi2(alpha)
            
            print(r"$\tilde{\chi}^{2} = %f$" % rchi2)
            
            # get axes for drawing
            fig1,ax1 = plt.subplots(1,1)
            fig2,ax2 = plt.subplots(1,1)
            
            # rebin the data
            x,y,dy = self._rebin(self.x,self.y,self.yerr,rebin=rebin)
            
            # get fit function
            if fitfn == 'exp':
                fn = self.pexp  # inputs: T1, amp
                p0 = (1,0.5)
                bounds = (0,(np.inf,1))
            elif fitfn == 'biexp':
                fn = lambda x,T1a,T1b,fracb,amp : amp*(self.pexp(x,T1a,1-fracb)+self.pexp(x,T1b,fracb))
                p0 = (1,1,0.5,0.5)
                bounds = (0,(np.inf,np.inf,1,1))
            elif fitfn == 'strexp':
                fn = self.sexp # inputs: T1, beta, amp
                p0 = (1,0.5,0.5)
                bounds = (0,(np.inf,1,1))
            else:
                raise RuntimeError('fitfn must be one of "exp", "biexp, or "strexp"')
            
            # do the fit
            par,cov = curve_fit(fn, self.x, self.y, sigma=self.yerr, 
                                absolute_sigma=True, p0=p0, bounds=bounds)
            std = np.diag(cov)**0.5
            
            # map the fit parameters
            if fitfn == 'exp':
                out = { 'T1':1/par[0],
                        'amp':par[1],
                        'dT1':std[0]/par[0]**2,
                        'damp':std[1]}
                
            elif fitfn == 'biexp':
                out = { 'T1a':1/par[0],
                        'T1b':1/par[1],
                        'fracb':par[2],
                        'amp':par[3],
                        'dT1a':std[0]/par[0]**2,
                        'dT1b':std[1]/par[1]**2,
                        'dfracb':std[2],
                        'damp':std[3]}
                        
            elif fitfn == 'strexp':
                out = { 'T1':1/par[0],
                        'beta':par[1],
                        'amp':par[2],
                        'dT1':std[0]/par[0]**2,
                        'dbeta':std[1],
                        'damp':std[2]}
            
            # print the fit parameters
            print("Standard fit: %s" % fitfn)
            for k,v in out.items():
                if k[0] == 'd': continue
                print('\t%s : %f \pm %f' % (k,v,out['d'+k]))
            
            # draw the fit on the data
            ax1.errorbar(x,y,dy,fmt='.k',zorder=1,label='Data')
            ax1.plot(self.x,fity,'r',zorder=2,label='ILT')
            ax1.plot(self.x,fn(self.x,*par),':b',zorder=3,label='Fit')
            
            if self.asym_type == 'Ac':
                ylabel = 'Four-Counter Asymmetry'
            else:
                ylabel = self.asym_type
                
            ax1.set_ylabel(ylabel)
            ax1.set_xlabel("Time (s)")
            
            # draw the probability distribution 
            lamb = 1/self.lamb
            p /= lamb # normalize
            ax2.semilogx(lamb,p/sum(p))
            ax2.set_ylabel("Probability Density")
            ax2.set_xlabel(r"$T_1$ ($s^{-1}$)")
            self.pnorm = p/sum(p)
            
            # titles
            ax1.set_title(r"$\alpha = %g$" % alpha)
            ax2.set_title(r"$\alpha = %g$" % alpha)
            fig1.tight_layout()
            fig2.tight_layout()
            fig1.legend()
            # return values 
            # ~ return(p, fity)
    
        else:
            super().draw(alpha)

