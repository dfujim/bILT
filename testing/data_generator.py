# Monte-Carlo simulation of bNMR data 
# Derek Fujimoto
# Feb 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import time

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
            theta_bins: bin centers for histogram of angles
            theta_hist: counts for histogram of angles
                
    """
    
    n_cumu_theta = 1000 # number of angles to use in the W(theta) inversion 
    
    def __init__(self, dt=0.01, tmax=16, pulse_len=4, lifetime=1.2096, A=-0.3333, 
                 beta_Kenergy=6):
        """
            dt:             bin spacing in s
            tmax:           duration of measurement cycle in s (beam on + beam off)
            pulse_len:      duration of beam on in s
            lifetime:       radioactive lifetime of probe in s
            A:              intrinsic probe asymmetry
            beta_energy:    kinetic energy of the beta decay in MeV
        """
        
        # set attributes
        self.pulse_len = pulse_len  
        self.lifetime = lifetime
        self.bins = np.arange(dt,tmax,dt)
        self.t = np.arange(dt/2,tmax-dt*2,dt)
        self.tmax = tmax
        self.A = A
        self.v = np.sqrt(1-(beta_Kenergy/0.51099895 + 1)**-2) # units of c

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

    def gen_counts(self, fn, n=1e6):
        """
            Generate the probe decays in each detector and histogram.
            
            fn: function handle of decay function for each probe. 
                prototype: fn(t). 
                
                example: fn = lambda t: 0.7*np.exp(-0.5*t)
            n: total number of probes to decay
        """
        
        n = int(n)
        
        # generate how long each probe lives (exponential distribution)
        t_life = np.random.exponential(self.lifetime, n)
        
        # generate when each probe is implantated (uniform distribution)
        t_arrive = np.random.uniform(0, self.pulse_len, n)

        # generate the time each probe decays
        t_decay = t_arrive + t_life
        
        # invert the cumulative distribution of the decay angles, W(theta) 
        theta = np.linspace(0, 2*np.pi, self.n_cumu_theta)
        rand = np.random.uniform(0, 1, n)
        prefac = self.v * self.A 
        
        cumu_W = lambda t : (theta + prefac * fn(t) * np.sin(theta)) / (2*np.pi)
        decay_angle_generator = (np.interp(r, cumu_W(t), theta) for r, t in zip(rand, t_life))
        decay_angle = np.fromiter(decay_angle_generator, count=n, dtype=float)
        
        # get decay orientation: if True, point to F, else B
        is_forward = (decay_angle < (np.pi/2)) + (decay_angle > (3*np.pi/2))
        
        # histogram
        self.F, _ = np.histogram(t_decay[is_forward],  bins=self.bins)
        self.B, _ = np.histogram(t_decay[~is_forward], bins=self.bins)
        self.theta_hist, self.theta_bins = np.histogram(decay_angle, bins=360)
        self.theta_bins = (self.theta_bins[1:] + self.theta_bins[:-1]) / 2
        
    def asym(self, rebin=1):
        """
            Asymmetry calculation
        """
        
        # asym
        a = (self.B - self.F) / (self.F + self.B)
        da = 2 * np.sqrt(self.B*self.F/(self.B+self.F)**3)
        
        self.a = a
        self.da = da
        
        # rebinning
        a,da = self._rebin((a,da),rebin)
        
        # rebin times
        t_rebin = [np.mean(self.t[i:i+rebin]) for i in range(0, len(self.t), rebin)]
        
        return np.array((t_rebin,a,da))
    
    def draw_diagnostics(self, fn,  n=1e6, rebin=1):
        """
            Run and draw diagnosis plots
            
            fn: function handle of decay function for each probe. 
                prototype: fn(t). 
                
                example: fn = lambda t: 0.7*np.exp(-0.5*t)
            n: total number of probes to decay
            rebin: for asymmetry drawing
        """
        
        # Run 
        t_start = time.time()
        self.gen_counts(fn, n=n)
        print('Runtime: ', time.time()-t_start, 's')
        
        # Counts
        plt.figure()
        plt.plot(self.t, self.F, label='F')
        plt.plot(self.t, self.B, label='B')
        plt.xlabel('Time (s)')
        plt.ylabel('Counts')
        plt.legend()
        plt.tight_layout()

        # draw asymmetry
        plt.figure()
        plt.errorbar(*self.asym(rebin), fmt='.')
        plt.xlabel('Time (s)')
        plt.ylabel('(B-F)/(B+F)')
        plt.tight_layout()
        
        # draw polar 
        plt.figure()
        W = lambda t: (1 + self.v * self.A * fn(t) * np.cos(self.theta_bins)) / (2*np.pi)
        plt.polar(self.theta_bins, self.theta_hist/n, label='MC Sim')
        plt.polar(self.theta_bins, W(0)/sum(W(0)), label='W(t=0)')
        plt.polar(self.theta_bins, W(self.pulse_len) / sum(W(self.pulse_len)),
                  label='W(t=%d s)'% int(self.pulse_len))
        plt.gca().set_yticklabels(())
        plt.legend(bbox_to_anchor=(1,1))
        
        plt.title('Distribution of Decay Angles')
