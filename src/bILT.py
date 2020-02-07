#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of a set of bnmr data
import yaml

import bdata as bd
import numpy as np
import matplotlib.pyplot as plt

from bfit.fitting.functions import pulsed_exp
from ilt import ilt

# =========================================================================== #
class bILT(ilt):
    """
        Inverse Laplace Transform for bNMR SLR data using pulsed exponentials
        as the Kernel matrix
    """
    
    def __init__(self,run,year=-1,rebin=1,probe='Li8'):
        """
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
            
            self._setup()
    
    def _setup(self):
        
        # get data
        dat = bd.bdata(self.run,self.year)
        self.x,self.y,self.yerr = dat.asym('c',rebin=self.rebin)
        
        # get function 
        f = pulsed_exp(lifetime=bd.life[self.probe],pulse_len=dat.get_pulse_s())
        self.fn = lambda x,w: f(x,w,1) 
                
        # build error matrix
        self.S = np.diag(1/self.yerr) 
    
    def fit(self,alpha,n=1000,T1=None,maxiter=None):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha, the regularization parameter
        
            alpha:      Tikhonov regularization parameter (may be list or number)
                        Try somewhere between 1e2 and 1e8
            n:          number of T1 values in array within 0.01*tau and 100*tau
                        (ignored if T1 is not none)
            T1:         user-specified T1 array
            maxiter:    max number of iterations in solver
        
        """
        
        # set weights
        self.n = n
        self.T1 = T1
        
        if T1 is None:    
            T1 = np.logspace(np.log(0.01 * 1.2096), np.log(100.0 * 1.2096), n)
        
        # run
        return super().fit(alpha,T1,maxiter)

    def read(self,filename):
        """
            Read yaml file and set properties
            
            filename:       name of file to write to 
        """
        
        # read file
        with open(filename,'r') as fid:
            self.__dict__ = yaml.safe_load(fid.read())
        self._setup()
        
            
        # make arrays
        for key in ('p',):
            self.__dict__[key] = np.array(self.__dict__[key])
            
        if self.T1 is not None:
            self.__dict__['T1'] = np.array(self.__dict__['T1'])
            self.w = self.T1
        else:
            self.w = np.logspace(np.log(0.01*1.2096), np.log(100*1.2096), self.n)
            
        if self.isiter:
            for key in ('alpha','chi'):
                self.__dict__[key] = np.array(self.__dict__[key])
        
        # assign some of the missing parts
        self.K = np.array([self.fn(self.x,i) for i in self.w]).T
    

    def write(self,filename,**notes):
        """
            Write to yaml file
            
            filename:       name of file to write to 
            notes:          additional fields to write
        """
        
        # read attributes
        dat = bd.bdata(self.run,self.year)
        output = {key:self.__dict__[key] for key in ('run','year','rebin','n',
                                                     'alpha','isiter','maxiter',
                                                     'p','T1','chi','probe')}
        output = {'title':dat.title,**output,**notes}
        
        # make numpy arrays lists
        for key in ('p',):
            output[key] = output[key].tolist()
        
        if self.isiter:
            for key in ('alpha','chi'):
                output[key] = output[key].tolist()
        
        if self.T1 is not None:
            output['T1'] = output['T1'].tolist()
        
        # write to file 
        print("writing...",end=' ',flush=True)
        with open(filename,'w') as fid:
            fid.write(yaml.safe_dump(output))
        print("done",flush=True)
            


