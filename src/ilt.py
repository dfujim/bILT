#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of a set of data

# https://stackoverflow.com/a/36112536
# https://stackoverflow.com/a/35423745
# https://gist.github.com/diogojc/1519756
import yaml

import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from scipy.optimize import nnls
from scipy.linalg import norm
from tqdm import tqdm

# =========================================================================== #
class ilt(object):
    """
        Get ILT for a kernel matrix defined using a general function, using a
        non-negative least squares solver for the constrained ridge resgression
        problem.
        
        Attributes 
        
            alpha:      Tikhonov regularization parameter       
            chi:        chisquared value of fit
            fity:       fit function corresponding to K*p
            fn:         function handle with signature f(x,w)
            isiter:     if True, alpha is a list, not a number
            maxiter:    max number of iterations in solver
            p:          array of probabilities corresponding to w, fit results
            S:          diagonal error matrix: diag(1/yerr)
            x:          array of time steps in data to fit
            y:          array of data points f(t) needing to fit
            yerr:       array of errors
            z:          array of transformed values corresponding to the 
                        probabilities in the output (ex: np.logspace(-5,5,500))
    """
    
    def __init__(self,x=None,y=None,yerr=None,fn=None):
        """
            x:          array of time steps in data to fit
            y:          array of data points f(t) needing to fit
            yerr:       array of errors
            fn:         function handle with signature f(x,w)
            
            If any inputs are none, make empty object. 
            Should then read from file.
        """    
        
        if not any([i is None for i in (x,y,yerr,fn)]):
            
            # Set inputs
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.yerr = np.asarray(yerr)
            self.fn = fn
            
            # build error matrix
            self.S = np.diag(1/yerr)

    def _annotate(self,ax,x,y,ptlabels):
        """
            Add annotation to figure
            
            x,y: coordinates of where to place the annotation
            ptlabels: labels for annotation
        """
        for label,xcoord,ycoord in zip(ptlabels,x,y):        
            if type(label) != type(None):
                ax.annotate(label,
                             xy=(xcoord,ycoord),
                             xytext=(-3, 20),
                             textcoords='offset points', 
                             ha='right', 
                             va='bottom',
                             bbox=dict(boxstyle='round,pad=0.1',
                                       fc='grey', 
                                       alpha=0.1),
                             arrowprops=dict(arrowstyle = '->', 
                                             connectionstyle='arc3,rad=0'),
                             fontsize='xx-small')    
    
    def _fit_single(self,alpha):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha, the regularization parameter
        """    
        
        # weighted variables for solving q = Lp
        L = np.matmul(self.K.T, np.matmul(self.S, self.K))
        q = np.matmul(self.K.T, np.matmul(self.S, self.y))
        
        # concatenate regularization
        # https://stackoverflow.com/a/35423745
        L = np.concatenate([L, np.sqrt(alpha) * np.eye(self.K.shape[1])])
        q = np.concatenate((q, np.zeros(self.K.shape[1])))

        # solve
        if self.maxiter is None:
            p, r = nnls(L, q)
        else:
            p, r = nnls(L, q, self.maxiter)
    
        # define functional output
        fity = np.dot(self.K, p)

        # chisquared
        chi = norm(np.matmul(self.S, (np.matmul(self.K, p) - self.y)))**2
        
        # calculate the generalize cross-validation (GCV) parameter tau
        # tau = np.trace(np.eye(K.shape[1]) - K ( np.matmul(K.T, K) + alpha * alpha * np.matmul(L.T, L) ) K.T )

        return (p, fity, chi)
        
    def draw(self,alpha_opt=None,fig=None):
        """
            Draw fit or range of fits. 
            
            alpha: if None draw:
                        alpha v chi
                        alpha v dchi/dalpha
                        L-curve
                    else draw:
                        data & fit
                        distribution 
            fig:    optional figure handle for redrawing when alpha_opt != None
        """
        
        # check if range of alphas
        if not self.isiter:
            alpha_opt = self.alpha
            
        # draw things for a single alpha only 
        if alpha_opt is not None:
            
            # get opt data 
            p, fity, chi = self._fit_single(alpha_opt)
            print(r"$\chi^2 = %f$" % chi)
            
            # get axes for drawing
            if fig is not None:
                ax1,ax2 = fig.axes
                for a in fig.axes:
                    a.clear()
                
            else:
                fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
            
            # draw the fit on the data
            ax1.errorbar(self.x,self.y,self.yerr,fmt='.k',zorder=1)
            ax1.plot(self.x,fity,'r',zorder=2)
            ax1.set_ylabel("$y$")
            ax1.set_xlabel("$x$")
            
            # draw the probability distribution 
            ax2.semilogx(self.z,p/np.sum(p))
            ax2.set_ylabel("Probability Density")
            ax2.set_xlabel("$z$")
            
            ax1.set_title(r"$\alpha = %g$" % alpha_opt)
            ax2.set_title(r"$\alpha = %g$" % alpha_opt)
            plt.tight_layout()
            
            # return values 
            return(p,fity,chi)
        
        # draw for a range of alphas
        else:     
            
            chi = self.chi**0.5  # chi, not chisquared
            ln_alpha = np.log(self.alpha)
            ln_chi = np.log(chi)
              
            # make canvas
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False,
                                           figsize=(6,7))
            
            # draw alpha as a function of chisquared ------------------
            ax1.plot(self.alpha, self.chi, "o", zorder=1)
            ax1.set_ylabel(r"$\chi^2(\alpha)$")
            ax1.set_yscale("log")
            plt.tight_layout()
            
            # draw dchi/dalpha as a function of chisquared ------------
            
            # derivative of logs
            # https://stackoverflow.com/a/19459160
            dlnchi_dlnalpha = np.gradient(ln_alpha) / np.gradient(ln_chi)
            
            
            ax2.plot(self.alpha, dlnchi_dlnalpha, ".-")
            ax2.set_xlabel(r"$\alpha$")
            ax2.set_ylabel(r"$\mathrm{d}\ln\chi(\alpha) / \mathrm{d}\ln\alpha$")
            ax2.axhline(0.1, linestyle="--", color="k", zorder=0,
                label=r"$\mathrm{d}\ln\chi(\alpha)/\mathrm{d}\ln\alpha = 0.1$")
            
            ax2.legend()
            ax2.set_xscale("log")
        
            # plot the L-curve ----------------------------------------
            p_norm = np.array([norm(i) for i in self.p])
            figp, axp = plt.subplots(1,1)

            axp.plot(chi, p_norm, "o-", zorder=1)
            self._annotate(axp,chi,p_norm,['%.3g'%a for a in self.alpha])
            
            axp.set_xlabel("$|| \Sigma ( K \mathbf{P} - \mathbf{y} ) ||$")
            axp.set_ylabel("$|| \mathbf{P} ||$")
            axp.set_title("L-curve")

            axp.set_xscale("log")
            axp.set_yscale("log")
            plt.tight_layout()

    def fit(self,alpha,z,maxiter=None):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha, the regularization parameter
        
            z:          array of transformed values corresponding to the 
                        probabilities in the output (ex: np.logspace(-5,5,500))
            alpha:      Tikhonov regularization parameter (may be list or number)
            maxiter:    max number of iterations in solver
        """    
        
        # Set inputs
        self.z = np.asarray(z)
        self.maxiter = maxiter
        x = self.x
        
        # build kernel matrix 
        self.K = np.array([self.fn(x,i) for i in z]).T
        
        # do list of alphas case
        if isinstance(alpha,Iterable):
            self.isiter = True
            self.alpha = np.asarray(alpha)
            p = []
            fity = []
            chi = []
            
            for a in tqdm(alpha,desc="calculating alphas:"):
                out = self._fit_single(a)
                p.append(out[0])
                fity.append(out[1])
                chi.append(out[2])
            
            self.p = np.array(p)
            self.fity = np.array(fity)
            self.chi = np.array(chi)
            
        # do a single alpha case
        else:
            self.isiter = False
            self.alpha = alpha
            self.p, self.fity, self.chi = self._fit_single(alpha)
        
        return (self.p,self.fity,self.chi)
        
    def read(self,filename):
        """
            Read yaml file and set properties
            
            filename:       name of file to write to 
        """
        
        # read file
        with open(filename,'r') as fid:
            self.__dict__ = yaml.safe_load(fid.read())
            
        # make arrays
        for key in ('x','y','yerr','z','p','fity','K' ):
            self.__dict__[key] = np.array(self.__dict__[key])
            
        if self.isiter:
            for key in ('alpha','chi'):
                self.__dict__[key] = np.array(self.__dict__[key])
        
        # assign some of the missing parts
        self.S = np.diag(1/self.yerr)
        
    def write(self,filename,**notes):
        """
            Write to yaml file
            
            filename:       name of file to write to 
            notes:          additional fields to write
        """
        
        # get all attributes
        output = {**self.__dict__,**notes}
        
        # remove the useless attributes, or those too large to be useful
        for key in ('fn','S'):
            del output[key]
            
        # make numpy arrays lists
        for key in ('x','y','yerr','z','p','K'):
            output[key] = output[key].tolist()
        
        if self.isiter:
            for key in ('alpha','chi'):
                output[key] = output[key].tolist()
        
        # write to file 
        print("writing...",end=' ',flush=True)
        with open(filename,'w') as fid:
            fid.write(yaml.safe_dump(output))
        print("done",flush=True)
            
