#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of a set of bnmr data
import yaml

import bdata as bd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from collections.abc import Iterable
from bfit.fitting.functions import pulsed_exp
from bILT.src.ilt import ilt

# =========================================================================== #
class bILT(ilt):
    """
        Inverse Laplace Transform for bNMR SLR data using pulsed exponentials
        as the Kernel matrix
        
        Attributues
        
        p_lognorm:  normalized p, accounting for logarithmic bin spacing of T1
        n:          number of T1 values in array within 0.01*tau and 100*tau
        T1:         user-specified T1 array
        
    """
    
    def __init__(self, run, year=-1, rebin=1, probe='Li8', T1=1000, nproc=1):
        """
            run:        run number
            year:       year 
            rebin:      rebinning in asymmetry calculation
            probe:      probe lifetime to use in exp calculation
            T1:         if int: number of T1 values in array within 0.01*tau and
                        100*tau
                        else:   user-specified T1 array
            nproc:      number of processsors to use
            
            if run is a filename, read from that file
        """
        
        if type(run) is str:
            self.read(run)
        else:
        
            # save inputs
            self.run = run
            self.year = year
            self.rebin = rebin
            self.probe = probe
            self.results = pd.Series()
            self.nproc = nproc
            
            # set weights
            self.lamb = T1
            
            if not isinstance(T1, Iterable):    
                
                # determine the upper/lower T1 limits based on the probe lifetime
                lifetime = bd.life[self.probe]
                log10_T1_min = np.log10(lifetime * 1e-2)
                log10_T1_max = np.log10(lifetime * 1e2)
                
                # generate the T1 range
                self.lamb = np.logspace(log10_T1_min, log10_T1_max, T1, base=10.0)
            
            # setup communal with read
            self._setup()
            
    def _setup(self):
        
        # get data
        dat = bd.bdata(self.run, self.year)
        self.x,self.y,self.yerr = dat.asym('c', rebin=self.rebin)
        
        # get function 
        f = pulsed_exp(lifetime=bd.life[self.probe], pulse_len=dat.get_pulse_s())
        self.fn = lambda x,w: f(x,w,1) 
                
        # build error matrix
        self.S = np.diag(1/self.yerr) 
        
        # set the kernel
        self.K = np.array([self.fn(self.x, i) for i in self.lamb]).T

    def read(self, filename):
        """
            Read yaml file and set properties
            
            filename:       name of file to write to 
        """
        
        # read file
        with open(filename,'r') as fid:
            file_contents = yaml.safe_load(fid.read())
            
        # set results
        results = pd.Series(file_contents['p'],index=file_contents['alpha'],
                                 name='p')    
        results.index.name = 'alpha'
        del file_contents['p']
        del file_contents['alpha']
    
        self.__dict__ = {**file_contents,'results':results}
        self._setup()
            
        # make arrays
        self.lamb = np.array(self.lamb)
    
        # assign error matrix
        self.S = np.diag(1 / self.yerr)
        
    def write(self, filename, **notes):
        """
            Write to yaml file
            
            filename:       name of file to write to 
            notes:          additional fields to write
        """
        
        # read attributes
        dat = bd.bdata(self.run, self.year)
        output = {key:self.__dict__[key] for key in ('run','year','rebin',
                                                     'maxiter','lamb','probe')}
        output = {'title':dat.title, **output, **notes}
        
        # make numpy arrays lists
        output['p'] = self.results.apply(np.ndarray.tolist).tolist()
        output['alpha'] = self.results.index.tolist()
        output['lamb'] = output['lamb'].tolist()
            
        # write to file 
        print("writing...", end=' ', flush=True)
        with open(filename, 'w') as fid:
            fid.write(yaml.safe_dump(output))
        print("done", flush=True)
            


