#!/usr/bin/python3

# Inverse Laplace Transform (ILT) of a set of data

# https://stackoverflow.com/a/36112536
# https://stackoverflow.com/a/35423745
# https://gist.github.com/diogojc/1519756

# annotate on hover
# https://stackoverflow.com/a/47166787

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
            annot:      annotation object (blank, but shown on hover)
            axp:        L curve matplotlib axis
            line:       L curve line drawn, used for annotation shown on hover 
            chi2:       chisquared value of fit
            figp:       L curve matplotlib figure
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
                        probabilities in the output (e.g., np.logspace(-5, 5, 500))
    """
    
    def __init__(self,x,y=None,yerr=None,fn=None):
        """
            x:          array of time steps in data to fit
            y:          array of data points f(t) needing to fit
            yerr:       array of errors
            fn:         function handle with signature f(x,w)
            
            If x is a string, read input from filename
        """    
        
        if type(x) is str:
            self.read(x)
        else:
            self.x = np.asarray(x)
            self.y = np.asarray(y)
            self.yerr = np.asarray(yerr)
            self.fn = fn
            
            # build error matrix
            self.S = np.diag(1/yerr)

    def _annotate(self,ind):
        """
            Show annotation 
        """
        x,y = self.line.get_data()
        idx = ind["ind"][0]
        self.annot.xy = (x[idx], y[idx])
        self.annot.set_text(r'$\alpha = $%.3g' % self.alpha[idx])
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
        
    def _fit_single(self,alpha):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha, the regularization parameter
        """    
        
        # weighted variables for solving q = Lp
        L = np.matmul(self.S, self.K)
        q = np.matmul(self.S, self.y)
        
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

        # calculate the fit's chisquared
        chi2 = norm(np.matmul(self.S, (np.matmul(self.K, p)) - self.y)) ** 2
        # calculate the fit's reduced chisquared
        rchi2 = chi2 / len(self.x)
        
        # calculate the generalize cross-validation (GCV) parameter tau
        # tau = np.trace(np.eye(K.shape[1]) - K ( np.matmul(K.T, K) + alpha * alpha * np.matmul(L.T, L) ) K.T )

        return (p, fity, chi2)
        
    def draw(self,alpha_opt=None,fig=None):
        """
            Draw fit or range of fits. 
            
            alpha_opt:  if None draw:
                            alpha v chi
                            alpha v dchi/dalpha
                            L-curve
                        else draw:
                            data & fit
                            distribution 
            fig:    optional figure handle for redrawing when alpha_opt != None
            
            returns: (p, fity, chi2)
            
                p:      array of unnormalized weights
                fity:   array of final fit function points
                chi2:   chisquared value of fit
        """
        
        # check if range of alphas
        if not self.isiter:
            alpha_opt = self.alpha
            
        # draw things for a single alpha only 
        if alpha_opt is not None:
            
            # get opt data 
            p, fity, chi2 = self._fit_single(alpha_opt)
            print(r"$\chi^{2} = %f$" % chi2)
            rchi2 = chi2 / len(self.x)
            print(r"$\tilde{\chi}^{2} = %f$" % rchi2)
            
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
            ax2.semilogx(self.z,p)
            ax2.set_ylabel("Weight")
            ax2.set_xlabel("$z$")
            
            ax1.set_title(r"$\alpha = %g$" % alpha_opt)
            ax2.set_title(r"$\alpha = %g$" % alpha_opt)
            plt.tight_layout()
            
            # return values 
            return(p, fity, chi2)
        
        # draw for a range of alphas
        else:     
            # get chi from the fit chisquared...
            # (i.e., the Euclidean norm of the (weighted) fit residuals)
            chi = np.sqrt(self.chi2)
            # ...and the natural logarithm of alpha and chi
            ln_alpha = np.log(self.alpha)
            ln_chi = np.log(chi)
            
            # calculate the norm of all the p-vectors
            p_norm = np.array([norm(i) for i in self.p])
            
            # indentify the alpha that gives the minium in chi2
            idx_min = np.argmin(self.chi2)
            # highlight this point on the generated plots
            chi2_min = self.chi2[idx_min]
            chi_min = chi[idx_min]
            alpha_min = self.alpha[idx_min]
            p_norm_min = p_norm[idx_min]
              
            # make canvas
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False,
                                           figsize=(6,7))
            
            # draw chi2 as a function of alpha ------------------
            ax1.plot(self.alpha, self.chi2, "o", zorder=1)
            ax1.plot(alpha_min, chi2_min, "s", zorder=2)
            ax1.set_ylabel(r"$\chi^{2}$")
            ax1.set_yscale("log")
            plt.tight_layout()
            
            # draw dchi/dalpha as a function of alpha ------------
            
            # derivative of logs
            # https://stackoverflow.com/a/19459160
            dlnchi_dlnalpha = np.gradient(ln_chi, ln_alpha)
            
            ax2.plot(self.alpha, dlnchi_dlnalpha, ".-")
            ax2.set_xlabel(r"$\alpha$")
            ax2.set_ylabel(r"$\mathrm{d} \ln \chi / \mathrm{d} \ln \alpha$")
            ax2.axhline(0.1, linestyle="--", color="k", zorder=0,
                label=r"$\mathrm{d} \ln \chi / \mathrm{d} \ln \alpha = 0.1$")
            
            ax2.legend()
            ax2.set_xscale("log")
        
            # plot the L-curve ----------------------------------------
            self.figp, axp = plt.subplots(1,1)
            self.axp = axp
            
            self.line, = axp.plot(chi, p_norm, "o-", zorder=1)
            axp.plot(chi_min, p_norm_min, "s", zorder=2)
            # annotate the parametric plot on mouse hover
            self.annot = axp.annotate("",
                                 xy=(0,0),
                                 xytext=(50, 20),
                                 textcoords='offset points', 
                                 ha='right', 
                                 va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.1',
                                           fc='grey', 
                                           alpha=0.1),
                                 arrowprops=dict(arrowstyle = '->', 
                                                 connectionstyle='arc3,rad=0'),
                                 fontsize='xx-small')
            self.annot.set_visible(False)
            
            # connect the hovering mechanism
            self.figp.canvas.mpl_connect("motion_notify_event", self._hover)
            
            # axis labels
            axp.set_xlabel("$|| \Sigma ( K \mathbf{p} - \mathbf{y} ) ||$")
            axp.set_ylabel("$|| \mathbf{p} ||$")
            axp.set_title("L-curve")

            axp.set_xscale("log")
            axp.set_yscale("log")
            plt.tight_layout()

    def fit(self,alpha,z,maxiter=None):
        """
            Run the non-negative least squares algorithm for a single value of 
            alpha (the regularization parameter) or an array of alphas
        
            alpha:      Tikhonov regularization parameter (may be list or number)
            z:          array of transformed values corresponding to the 
                        probabilities in the output (ex: np.logspace(-5,5,500))
            maxiter:    max number of iterations in solver
            
            returns: (p, fity, chi2)
        
                p:      array of unnormalized weights
                fity:   array of final fit function points
                chi2:   chisquared value of fit
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
            chi2 = []
            
            for a in tqdm(alpha,desc="NNLS optimization @ each alpha"):
                out = self._fit_single(a)
                p.append(out[0])
                fity.append(out[1])
                chi2.append(out[2])
            
            self.p = np.array(p)
            self.fity = np.array(fity)
            self.chi2 = np.array(chi2)
            
        # do a single alpha case
        else:
            self.isiter = False
            self.alpha = alpha
            self.p, self.fity, self.chi2 = self._fit_single(alpha)
        
        return (self.p,self.fity,self.chi2)
        
    def read(self,filename):
        """
            Read yaml file and set properties
            
            filename:       name of file to write to 
        """
        
        # read file
        with open(filename,'r') as fid:
            self.__dict__ = yaml.safe_load(fid.read())
            
        # make arrays
        for key in ('x','y','yerr','z','p','K' ):
            self.__dict__[key] = np.array(self.__dict__[key])
            
        if self.isiter:
            for key in ('alpha','chi2'):
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
        for key in ('fn','S','fity'):
            del output[key]
            
        # make numpy arrays lists
        for key in ('x','y','yerr','z','p','K'):
            output[key] = output[key].tolist()
        
        if self.isiter:
            for key in ('alpha','chi2'):
                output[key] = output[key].tolist()
        
        # write to file 
        print("writing...",end=' ',flush=True)
        with open(filename,'w') as fid:
            fid.write(yaml.safe_dump(output))
        print("done",flush=True)
            
