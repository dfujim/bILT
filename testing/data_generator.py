# Monte-Carlo simulation of bNMR data 
# Derek Fujimoto
# Feb 2020

import numpy as np
import matplotlib.pyplot as plt

# =========================================================================== #
class data_generator(object):
    """
        Simulate bnmr SLR data 
        
        Attributes: 
            
            a,da:       asymmetry and error
            bins:       time bin edges in s
            F,B:        histogrammed counts in counters
            lifetime:   radioactive lifetime of probe in s
            pulse_len:  duration of beam on in s
            t:          measurement bin centers in s (the x axis)
                
    """
    def __init__(self, dt=0.01, tmax=16, pulse_len=4, lifetime=1.2096):
        """
            dt:         bin spacing in s
            tmax:       duration of measurement cycle in s (beam on + beam off)
            pulse_len:  duration of beam on in s
            lifetime:   radioactive lifetime of probe in s
        """
        
        # set attributes
        self.pulse_len = pulse_len  
        self.lifetime = lifetime
        self.bins = np.arange(dt,tmax,dt)
        self.t = np.arange(dt/2,tmax-dt*2,dt)

    def _rebin(self, xdx, rebin):
        """
            Rebin array x with weights 1/dx**2 by factor rebin.
            
            Inputs: 
                xdx = [x,dx]
                rebin = int
            Returns [x,dx] after rebinning. 
        """

        x = xdx[0]
        dx = xdx[1]
        rebin = int(rebin)
        
        # easy end condition
        if rebin <= 1:
            return (x,dx)
        
        # Rebin Discard unused bins
        lenx = len(x)
        x_rebin = []
        dx_rebin = []
        
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
        return np.array([x_rebin,dx_rebin])

    def gen_counts(self, fn, p0=0.1,  n=1e6):
        """
            Generate the probe decays in each detector and histogram.
            
            fn: function handle of decay function for each probe. Amplitude 1. 
                prototype: fn(t). 
                
                example: fn = lambda t: np.exp(-0.5*t)
            p0: initial polarization 
            n: total number of probes to decay
        """
        
        n = int(n)
        
        # generate how long each probe lives (exponential distribution)
        t_life = np.random.exponential(self.lifetime, n)
        
        # generate when each probe is implantated (uniform distribution)
        t_arrive = np.random.uniform(0, self.pulse_len, n)

        # generate the time each probe decays
        t_decay = t_arrive + t_life
        
        # polarization: if p is true, decay towards F, else towards B
        p = np.ones(n,dtype=bool)
        
        # depolarize based on initial polarization
        depol = np.random.random(n)>p0
        
        # depolarize due to SLR function
        depol += np.random.random(n) < 1 - fn(t_life)
        
        # if depolarized, set probability of decay to point randomly at F or B
        p[depol] = np.random.random(sum(depol))<0.5
        
        # histogram
        self.F,_ = np.histogram(t_decay[p],bins=self.bins)
        self.B,_ = np.histogram(t_decay[~p],bins=self.bins)
        
    def asym(self, rebin=1):
        """
            Asymmetry calculation
        """
        
        # asym
        a = (self.F-self.B)/(self.F+self.B)
        da = 2*np.sqrt(self.B*self.F/(self.B+self.F)**3)
        
        self.a = a
        self.da = da
        
        # rebinning
        a,da = self._rebin((a,da),rebin)
        
        # rebin times
        t_rebin = [np.mean(self.t[i:i+rebin]) for i in range(0,len(self.t),rebin)]
        
        return np.array((t_rebin,a,da))
    
    

