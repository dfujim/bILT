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
        bdat:       bdata object corresponding to base data file
        year:       year of run
        run:        run number
    """
    
    def __init__(self, bdat, rebin=1, probe='Li8', T1=1000, nproc=1):
        """
            bdat:       bdata (or bmerged) object corresponding to run to analyze
                        OR YAML settings filename to read
            rebin:      rebinning in asymmetry calculation
            probe:      probe lifetime to use in exp calculation
            T1:         if int: number of T1 values in array within 0.01*tau and
                        100*tau
                        else:   user-specified T1 array
            nproc:      number of processsors to use
            
            if run is a filename, read from that file
        """
        
        if type(bdat) is str:
            self.read(bdat)
        elif type(bdat) is bd.bdata:
        
            # save inputs
            self.run = bdat.run
            self.year = bdat.year
            self.rebin = rebin
            self.probe = probe
            self.results = pd.Series(dtype=np.float64)
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
            self._setup(bdat)
            
    def _setup(self, bdat):
        """
            Input: bdata object
        """
        
        # get data
        self.bdat = bdat
        self.x,self.y,self.yerr = bdat.asym('c', rebin=self.rebin)
        
        # remove zero error values
        idx = self.yerr != 0
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.yerr = self.yerr[idx]
        
        # get function 
        f = pulsed_exp(lifetime=bd.life[self.probe], pulse_len=self.bdat.pulse_s)
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
        results = pd.Series(file_contents['p'], 
                            index=file_contents['alpha'],
                            name='p',
                            dtype=np.float64)    
        results.index.name = 'alpha'
        del file_contents['p']
        del file_contents['alpha']
    
        self.__dict__ = {**file_contents,'results':results}
        
        # read file(s)
        if self.year > 4000:
            
            self.year = str(self.year)
            self.run = str(self.run)
            
            # split year and runs
            years = [int(self.year[i:i+4]) for i in range(0, len(self.year), 4)]
            runs = [int(self.runs[i:i+5]) for i in range(0, len(self.runs), 5)]
            
            # get data
            bdat = bd.bmerged([bd.bdata(r, y) for r, y in zip(runs, years)])
            
        else:
            bdat = bd.bdata(self.run, self.year)
        
        # set up
        self._setup(bdat)
            
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
        output = {key:self.__dict__[key] for key in ('run','year','rebin',
                                                     'maxiter','lamb','probe') 
                                         if key in self.__dict__.keys()}
        output = {'title':self.bdat.title, **output, **notes}
        
        # make numpy arrays lists
        output['p'] = self.results.apply(np.ndarray.tolist).tolist()
        output['alpha'] = self.results.index.tolist()
        output['lamb'] = output['lamb'].tolist()
            
        # write to file 
        print("writing...", end=' ', flush=True)
        with open(filename, 'w') as fid:
            fid.write(yaml.safe_dump(output))
        print("done", flush=True)
            


