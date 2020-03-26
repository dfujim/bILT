#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of a set of data

# https://stackoverflow.com/a/36112536
# https://stackoverflow.com/a/35423745
# https://gist.github.com/diogojc/1519756

# annotate on hover
# https://stackoverflow.com/a/47166787

import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from collections.abc import Iterable
from scipy.optimize import nnls
from scipy.linalg import norm
from tqdm import tqdm
from multiprocessing import Pool

# =========================================================================== #
class ilt(object):
    """
        Get ILT for a kernel matrix defined using a general function, using a
        non-negative least squares solver for the constrained ridge resgression
        problem.
        
        Attributes 
        
            annot:      annotation object (blank, but shown on hover)
            axp:        L curve matplotlib axis
            lamb:       array of transformed values corresponding to the 
                        probabilities in the output (e.g., np.logspace(-5, 5, 500))
            line:       L curve line drawn, used for annotation shown on hover 
            figp:       L curve matplotlib figure
            fn:         function handle with signature f(x,w)
            K:          Kernel matrix: 2D array
            maxiter:    max number of iterations in solver
            nproc:      number of processors to use
            results:    pd.Series fit results with index alpha and values p
                        alpha:      Tikhonov regularization parameter      
                        p:          array of weights corresponding to each lamb
            S:          diagonal matrix of 1/yerr
                        diagonal error matrix: diag(1/yerr)
            x:          array of time steps in data to fit
            y:          array of data points f(t) needing to fit
            yerr:       array of errors
            
    """
    
    def __init__(self, x, y=None, yerr=None, fn=None, lamb=None, nproc=1):
        """
            x:          array of time steps in data to fit
            y:          array of data points f(t) needing to fit
            yerr:       array of errors
            fn:         function handle with signature f(x,w)
            lamb:       array of transformed values corresponding to the 
                        probabilities in the output (ex: np.logspace(-5,5,500))
            nproc:      number of processors to use
            
            If x is a string, read input from filename
        """    
        
        if type(x) is str:
            self.read(x)
        else:
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.yerr = np.asarray(yerr)
            self.fn = fn
            self.nproc = nproc
            
            # build error matrix
            self.S = np.diag(1/yerr)
            
            # data frame for storage
            self.results = pd.Series()
            
            # build kernel matrix 
            self.lamb = np.asarray(lamb)
            self.K = np.array([self.fn(x, i) for i in lamb]).T

    def _annotate(self,ind):
        """
            Show annotation 
        """
        x,y = self.line.get_data()
        idx = ind["ind"][0]
        self.annot.xy = (x[idx], y[idx])
        self.annot.set_text(r'$\alpha = $%.3g' % self.get_alpha()[idx])
        self.annot.get_bbox_patch().set_alpha(0.1)            
    
    def _hover(self,event):
        vis = self.annot.get_visible()
        if event.inaxes == self.axp:
            cont, ind = self.line.contains(event)
            if cont:
                self._annotate(ind)
                self.annot.set_visible(True)
                self.figp.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.figp.canvas.draw_idle()
        
    def draw(self, alpha=None, fig=None):
        """
            Draw fit or range of fits. 
            
            alpha:  if None draw:
                            alpha v chi
                            alpha v dchi/dalpha
                            L-curve
                        else draw:
                            data & fit
                            distribution 
            fig:    optional figure handle for redrawing when alpha != None
            
            returns: (p, fity, chi2)
            
                p:      array of unnormalized weights
                fity:   array of final fit function points
                chi2:   chisquared value of fit
        """
            
        # draw things for a single alpha only 
        if alpha is not None:
            
            # print chi
            print(r"$\chi^{2} = %f$" % self.get_chi2(alpha))
            print(r"$\tilde{\chi}^{2} = %f$" % self.get_rchi2(alpha))
            
            # get axes for drawing
            if fig is not None:
                ax1,ax2 = fig.axes
                for a in fig.axes:
                    a.clear()
                
            else:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # draw the fit on the data
            self.draw_fit(alpha, ax=ax1)
            
            # draw the probability distribution 
            self.draw_weights(alpha, ax=ax2)
        
        # draw for a range of alphas
        else:     
            
            # get alpha
            alpha = self.get_alpha()
            if len(alpha) == 0:    
                raise RuntimeError('No values of alpha found. Fit some data first.')
            
            # draw S-curve with a default threshold
            plt.figure()
            self.draw_Scurve(0.1)
            plt.title('S-Curve')
            plt.tight_layout()
            
            # plot the L-curve
            plt.figure()
            self.draw_Lcurve()
            plt.title('L-Curve')
            plt.tight_layout()
            
            # draw the GCV
            plt.figure()
            self.draw_gcv()
            plt.title('Generalized Cross-Validation')
            plt.tight_layout()

    def draw_fit(self, alpha, ax=None):
        """
            Draw the fit and the data
            
            alpha:  regularization parameter
            ax:     axis to draw in    
        """
        
        # get default axis
        if ax is None:
            ax = plt.gca()
        
        # draw data
        ax.errorbar(self.x, self.y, self.yerr, fmt='.k', zorder=0)
        
        # draw fit
        fity = self.get_fit(alpha)
        ax.plot(self.x, fity, 'r', zorder=1)
        
        # plot elements
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.tight_layout()
        
    def draw_gcv(self, ax=None):
        """
            Draw the Generalized Cross-Validation Parameter curve
        """
        
        # get gcv
        alpha,gcv = self.get_gcv()
        
        # get index of best alpha
        opt = np.argmin(gcv)

        # get default axis
        if ax is None:
            ax = plt.gca()
        
        # plot 
        ax.loglog(alpha, gcv, 'o-', zorder=1)
        ax.plot(alpha[opt], gcv[opt], 's', zorder=2, 
                label=r'$\alpha_{opt} = %g$' % alpha[opt])
        
        # plot elements
        ax.legend()
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('GCV')
        plt.tight_layout()
        
    def draw_Lcurve(self,*args,**kwargs):
        """
            Draw the L curve with fancy mouse hover and highlighting
            
            arguments passed to get_Lcurve_opt
        """
        
        # make figure
        self.figp = plt.gcf()
        axp = plt.gca()
        self.axp = axp
        
        # get data 
        chi,p_norm = self.get_Lcurve()
        
        # draw line
        self.line, = axp.plot(chi, p_norm, "o-", zorder=1)
        
        # get the point of max curvature
        alpha_opt = self.get_Lcurve_opt(*args,**kwargs)
        
        # draw the point of max curvature
        axp.plot(chi[alpha_opt], p_norm[alpha_opt], 's', zorder=2,
                 label = r'$\alpha_{opt} = %g$' % alpha_opt)
        plt.legend()
        
        # annotate the parametric plot on mouse hover
        self.annot = axp.annotate("",
                             xy=(0, 0),
                             xytext=(50, 20),
                             textcoords='offset points', 
                             ha='right', 
                             va='bottom',
                             bbox=dict(boxstyle='round,pad=0.1',
                                       fc='grey', 
                                       alpha=0.1),
                             arrowprops=dict(arrowstyle='->', 
                                             connectionstyle='arc3,rad=0'),
                             fontsize='xx-small')
        self.annot.set_visible(False)
        
        # connect the hovering mechanism
        self.figp.canvas.mpl_connect("motion_notify_event", self._hover)
        
        # axis labels
        axp.set_xlabel("$|| \Sigma ( K \mathbf{p} - \mathbf{y} ) ||$")
        axp.set_ylabel("$|| \mathbf{p} ||$")
        
        axp.set_xscale("log")
        axp.set_yscale("log")
        plt.tight_layout()
    
    def draw_Scurve(self, threshold=-1, ax=None):
        """
            Draw alpha vs gradient of logs
            
            threshold:  if > 0, draw dotted line at this value
            ax:         axis to draw in    
        """
        
        # get default axis
        if ax is None:
            ax = plt.gca()
            
        alpha, chi = self.get_Scurve()
        
        ax.semilogx(alpha, chi, "o-",zorder=1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\chi/\sqrt{N}$")
        
        if threshold > 0:
            alpha_opt = self.get_Scurve_opt(threshold)
            plt.plot(alpha_opt, chi[alpha_opt], 's', zorder=2,
                     label = r'$\alpha_{opt} = %g$ (threshold of %g)' % \
                            (alpha_opt,threshold))
            plt.legend()
        plt.tight_layout()
        
    def draw_logdist(self, alpha, ax=None):
        """
            Draw the weights as a function of lamb, normalized for a log 
            distribution of lambda
            
            alpha:  regularization parameter
            ax:     axis to draw in    
        """
        
        # get default axis
        if ax is None:
            ax = plt.gca()
        
        # draw the probability distribution 
        w = self.get_weights(alpha) / self.lamb
        w /= np.sum(w)
        ax.semilogx(self.lamb, w)
        ax.set_ylabel("Probability Density")
        ax.set_xlabel("$\lambda$")
        plt.tight_layout()
        
    def draw_weights(self, alpha, ax=None):
        """
            Draw the weights as a function of lamb
            
            alpha:  regularization parameter
            ax:     axis to draw in    
        """
        
        # get default axis
        if ax is None:
            ax = plt.gca()
        
        # draw the probability distribution 
        ax.semilogx(self.lamb, self.get_weights(alpha))
        ax.set_ylabel("Weight")
        ax.set_xlabel("$\lambda$")
        plt.tight_layout()
    
    def fit(self, alpha, maxiter=None):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha (the regularization parameter) or an array of alphas
        
            alpha:      Tikhonov regularization parameter (may be list or number)
            maxiter:    max number of iterations in solver
        """    
        
        # Set inputs
        self.maxiter = maxiter
        x = self.x
        
        # do list of alphas case
        if isinstance(alpha, Iterable):
            alpha = np.asarray(alpha)
            
            # don't repeat alphas that are already fitted
            alpha = np.setdiff1d(alpha, self.get_alpha())
            
            # easy end case
            if len(alpha) == 0:
                return
            
            # set up computation
            iterable = tqdm(alpha, 
                            desc="NNLS optimization @ each alpha",
                            total=len(alpha))
            
            # function to apply
            fn = partial(_fit_single, y=self.y, K=self.K, S=self.S, maxiter=maxiter)
            
            # serial processing
            if self.nproc <= 1:
                p = list(map(fn, iterable))
            else:
                pl = Pool(self.nproc)
                try:
                    p = list(pl.map(fn, iterable))
                finally:
                    pl.close()
            
        # do a single alpha case
        else:
            # don't repeat alphas that are already fitted
            if alpha in self.get_alpha():
                return
            
            p = [_fit_single(alpha, self.y, self.K, self.S, maxiter)]            
            alpha = [alpha]
            
        # save the results
        new_results = pd.Series(p, index=alpha,  name='p')
        new_results.index.name = 'alpha'
        self.results = self.results.append(new_results)
        
        # sort
        self.results.sort_index(inplace=True)
    
    def get_alpha(self):
        return self.results.index.values
    
    def get_chi2(self, alpha=None):
        """
            Calculate and return the chisquared for a particular value of alpha
            If alpha is None, get for all values of alpha
        """
        
        # function to calculate chisquared
        chifn = lambda p : norm(np.matmul(self.S, (np.matmul(self.K, p)) - self.y)) ** 2
        
        # do all alphas
        if alpha is None:
            chi2 = self.results.apply(chifn)
        
        # do single alpha
        else:
            if alpha not in self.get_alpha():
                self.fit(alpha)
            chi2 = chifn(self.results[alpha])
            
        return chi2
    
    def get_rchi2(self, alpha=None):
        """
            Calculate and return the reduced chisquared for a particular value 
            of alpha
            
            If alpha is None, get for all values of alpha
        """
        
        # calculate chi
        chi2 = self.get_chi2(alpha)
            
        # return the fit results
        return chi2 / len(self.x)
    
    def get_fit(self, alpha):
        """Calculate and return the fit points for a particular value of alpha"""
        
        # check if alpha is in the list of calculated alphas
        if alpha not in self.get_alpha():
            self.fit(alpha)
        
        # return the fit results
        return np.dot(self.K, self.results[alpha])
        
    def get_gcv(self):
        """Calculate the generalized cross-validation parameter"""
        
        # get needed data
        K = self.K
        p = self.results
        y = self.y
        KT = K.T
        I = np.eye(K.shape[1])
        alpha = self.get_alpha()
        chi2 = self.get_chi2()
        
        
        # prep calculations
        KTK = np.matmul(KT, K)
        
        # calculate gcv
        gcv = []    
        for a,c in zip(alpha,chi2):
        
            # regularized inverse of the kernel
            Kinv = np.matmul(np.linalg.inv(KTK + a*I), KT)
        
            # denominator
            denominator = (K.shape[1] - np.trace(np.matmul(K,Kinv)))**2
            
            gcv.append(c / denominator)
            
        return (self.get_alpha(),np.array(gcv))
    
    def get_gcv_opt(self):
        """
            Calculate alpha_opt based on the generalized cross-validation 
            parameter (min gcv)
        """
        
        # get gcv
        alpha, gcv = self.get_gcv()
        
        # find min
        return alpha[np.argmin(gcv)]
    
    def get_Lcurve(self):
        """
            return (chi, norm of weight vector)
        """
        
        # residual norm: fit chi2 square-rooted
        res_norm = np.sqrt(self.get_chi2())
        
        # solution norm
        sln_norm = self.results.apply(norm)
        
        return (res_norm,sln_norm)
    
    def get_Lcurvature(self):
        """
            find the curvature of the l curve
        """
        
        # get the Lcurve
        x, y = self.get_Lcurve()
        alpha = self.get_alpha()
        
        # take the log
        x = np.log(x)
        y = np.log(y)
        
        # take second gradient with respect to alpha
        y1 = np.gradient(y, alpha)
        y11 = np.gradient(y1, alpha)
        x1 = np.gradient(x, alpha)
        x11 = np.gradient(x1, alpha)
        
        # curvature
        curvature = (x1*y11 - x11*y1) / (x1**2 + y1**2)**1.5
        
        return (alpha,curvature)
    
    def get_Lcurve_opt(self,mode='auto',threshold=7):
        """
            Find alpha opt based on the L curve
            
            mode:       auto:       switch between other modes based on curvature
                        curvature:  point of maximum curvature
                        balance:    point of min(x**2*y)
                        
            threshold:  if < threshold, use balance mode, else use curvature mode
        """
        
        x,y = self.get_Lcurve()
        alpha = self.get_alpha()
        
        if mode in 'curvature':
            alpha,curve = self.get_Lcurvature()
            return alpha[np.argmax(curve)]
        
        elif mode in 'balance':
            return alpha[np.argmin(x**2*y)]
        
        elif mode in 'auto':
            
            alpha,curve = self.get_Lcurvature()
            if max(curve) > threshold:
                return self.get_Lcurve_opt(mode='curvature')
            else:
                return self.get_Lcurve_opt(mode='balance')
            
        else:
            raise RuntimeError('Bad input. Mode must be one of "curvature" or "balance"')
        
    def get_Scurve(self):
        """return ( alpha, rchi )"""
        
        # natural logarithm of chi
        chi = np.sqrt(self.get_rchi2())
        
        # return alpha, residual norm
        return (self.get_alpha(), chi)
    
    def get_Sgrad(self):
        """
            Get the gradient of the log of the S curve
            return ( alpha, dln(chi) / dln(alpha) )
        """
        
        chi = np.sqrt(self.get_chi2())
        ln_chi = np.log(chi)
        
        # ...and the natural logarithm of alpha
        ln_alpha = np.log(self.get_alpha())
    
        # take the gradient
        dlnchi_dlnalpha = np.gradient(ln_chi, ln_alpha)
        
        return (self.get_alpha(), dlnchi_dlnalpha)
    
    def get_Scurve_opt(self,threshold=0.1):
        """
            Get optimum value of alpha based on the S curve: when 
            d ln(chi) / d ln(alpha) > threshold
        """
        
        alpha, grad = self.get_Sgrad()
        alpha_contender = alpha[grad>threshold]
        return min(alpha_contender)
    
    def get_weights(self, alpha):
        """
            Calculate and return the distribution of weights for a particular 
            value of alpha
        """
        
        # check if alpha is in the list of calculated alphas
        if alpha not in self.get_alpha():
            self.fit(alpha)
            
        # return the fit results
        return self.results[alpha]
    
    def read(self,filename):
        """
            Read yaml file and set properties
            
            filename:       name of file to write to 
        """
        
        # read file
        with open(filename, 'r') as fid:
            file_contents = yaml.safe_load(fid.read())
            
        # make arrays
        for key in ('x', 'y', 'yerr', 'lamb', 'K' ):
            self.__dict__[key] = np.array(file_contents[key])
            
        # set results
        self.results = pd.Series(file_contents['p'], index=file_contents['alpha'],
                                 name='p')
        self.results.index.name = 'alpha'
            
        # assign error matrix
        self.S = np.diag(1 / self.yerr)
        
    def write(self, filename, **notes):
        """
            Write to yaml file
            
            filename:       name of file to write to 
            notes:          additional fields to write
        """
        
        # get needed attributes
        output = {k:self.__dict__[k] for k in ('x', 'y', 'yerr', 'lamb', 'K',)}
        output = {**output,**notes}
        
        # add results 
        output['p'] = self.results.apply(np.ndarray.tolist).tolist()
        output['alpha'] = self.get_alpha()
            
        # make numpy arrays lists
        for key in ('x', 'y', 'yerr', 'lamb', 'K','alpha'):
            output[key] = output[key].tolist()
        
        # write to file 
        print("writing...", end=' ', flush=True)
        
        try:
            with open(filename, 'w') as fid:
                fid.write(yaml.safe_dump(output))
        except Exception as err:
            print(err)
            return output
        print("done", flush=True)

# Fit a fingle function             
def _fit_single(alpha, y, K, S, maxiter):
    """
        Run the non-negative least squares algorithm for a single value of 
        alpha, the regularization parameter
        
        K:          kernel matrix
        S:          diagonal matrix of 1/yerr
        y:          data measurements
        maxiter:    maximum number of iterations in nnls
        
    """    
    
    # weighted variables for solving q = Lp
    L = np.matmul(S, K)
    q = np.matmul(S, y)
    
    # concatenate regularization
    # https://stackoverflow.com/a/35423745
    L = np.concatenate([L, alpha * np.eye(K.shape[1])])
    q = np.concatenate((q, np.zeros(K.shape[1])))

    # solve
    if maxiter is None: p, r = nnls(L, q)
    else:               p, r = nnls(L, q, maxiter)

    return p
