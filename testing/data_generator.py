# Monte-Carlo simulation of bNMR data 
# Derek Fujimoto
# Feb 2020

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import time
from multiprocessing import Pool
import pandas as pd

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
    n_angle_bins = 360  # number of bins in histogram of angles
    
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

    def _gen_counts(self, pol_fn, n):
        """
            Worker function for generating the probe decays in each detector and 
            histogram.
            
            pol_fn: function handle of decay function for each probe. 
                    prototype: pol_fn(t). 
                
                    example: pol_fn = lambda t: 0.7*np.exp(-0.5*t)
            n: total number of probes to decay
        """
        
        n = int(n)
        
        # ensure state of random generator is unique to processor
        np.random.seed()
        
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
        
        cumu_W = lambda t : (theta + prefac * pol_fn(t) * np.sin(theta)) / (2*np.pi)
        decay_angle_generator = (np.interp(r, cumu_W(t), theta) for r, t in zip(rand, t_life))
        decay_angle = np.fromiter(decay_angle_generator, count=n, dtype=float)
        
        # get decay orientation: if True, point to F, else B
        is_forward = (decay_angle < (np.pi/2)) + (decay_angle > (3*np.pi/2))
        
        # histogram
        F, _ = np.histogram(t_decay[is_forward],  bins=self.bins)
        B, _ = np.histogram(t_decay[~is_forward], bins=self.bins)
        theta_hist, theta_bins = np.histogram(decay_angle, bins=self.n_angle_bins)
        theta_bins = (theta_bins[1:] + theta_bins[:-1]) / 2
        
        return (F,B,theta_hist,theta_bins)
    
    def _sum_counts(self,result):
        """
            Sum the output of _gen_counts on asynchronous execution
        """
        self.F += result[0]
        self.B += result[1]
        self.theta_hist += result[2]
    
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
    
    def draw_diagnostics(self, fn,  n=1e6, rebin=1, nproc=4):
        """
            Run and draw diagnosis plots
            
            fn: function handle of decay function for each probe. 
                prototype: fn(t). 
                
                example: fn = lambda t: 0.7*np.exp(-0.5*t)
            n: total number of probes to decay
            rebin: for asymmetry drawing
            nproc: number of processors to use
        """
        
        # Run 
        t_start = time.time()
        self.gen_counts(fn, n=n, nproc=nproc)
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

    def gen_counts(self, pol_fn, n=1e6, nproc=4,N_per_proc=1e6):
        """
            Generate the probe decays in each detector and histogram.
            
            pol_fn:     function handle of decay function for each probe. 
                        prototype: pol_fn(t). 
                    
                        example: pol_fn = lambda t: 0.7*np.exp(-0.5*t)
                            (however pol_fn must be pickleable)
            n:          total number of probes to decay
            nproc:      number of processors to use
            N_per_proc: max number of probes assigned to each processor
        """
        
        # split the probes into sets 
        n_array = np.full(int(np.floor(n/N_per_proc)), N_per_proc)
        if n % N_per_proc > 0 : 
            n_array = np.append(n_array, n % N_per_proc)
        
        # initialize the output with a short calculation
        self.F, self.B, self.theta_hist, self.theta_bins = self._gen_counts(pol_fn, 50)
        n_array[0] -= 50
        
        # set up multiprocessing
        p = Pool(nproc)
        try:
            
            # do calculations in full
            for i in n_array:
                p.apply_async(self._gen_counts, 
                              args = (pol_fn, i),
                              callback = self._sum_counts)
        finally:
            p.close()
            p.join()
            
    def get_stats(self):
        """
            Return tuple: 
                (sum of histograms F+B, sum of histogram theta_hist)
        """
        
        FB = sum(self.F+self.B)
        t = sum(self.theta_hist)
        
        return (FB,t)
        
    def read_csv(self,filename):
        """
            Read B, F, and t from csv. Useful for making use of asym calculator
        """
        df = pd.read_csv(filename)
        
        self.B = df['B'].values
        self.F = df['F'].values
        self.t = df['t'].values
        
    def to_csv(self,*args,**kwargs):
        """
            Write B, F, and t to csv. 
            Pass arguments to pandas.DataFrame.to_csv()
        """
        
        # make the output dataframe
        df = pd.DataFrame({'F':self.F,'B':self.B,'t':self.t})
        
        # set input defaults
        if 'index' not in kwargs:
            kwargs['index'] = False
        
        df.to_csv(*args,**kwargs)
