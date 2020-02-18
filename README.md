# `bILT`: Inverse Laplace Transform (ILT) objects, optimized for β-NMR SLR data

## API definition

The `bILT` package provides two objects: [`ilt`](https://github.com/dfujim/bILT/blob/master/src/ilt.py) and [`bILT`](https://github.com/dfujim/bILT/blob/master/src/bILT.py). Both take the inverse Laplace transform of a data set, but the latter is optimized for β-NMR data. We now outline the API: 

### bILT.ilt

Object constructor

```python
ilt(x,y=None,yerr=None,fn=None)
    """
        x:          array of time steps in data to fit
        y:          array of data points f(t) needing to fit
        yerr:       array of errors
        fn:         function handle with signature f(x,w)
        
        If x is a string, read input from filename
    """ 
```

Object attributes

```
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
```

Object public functions

```python
draw(alpha_opt=None,fig=None)
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
    
fit(alpha,z,maxiter=None)
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

read(filename)
    """
        Read yaml file and set properties
        
        filename:       name of file to write to 
    """
    
write(filename,**notes)
    """
        Write to yaml file
        
        filename:       name of file to write to 
        notes:          additional fields to write
    """
```

### bILT.bILT

Inherits from `ilt`

Object constructor

```python
bILT(run,year=-1,rebin=1,probe='Li8')
    """
        run:        run number
        year:       year 
        rebin:      rebinning in asymmetry calculation
        probe:      probe lifetime to use in exp calculation
        
        if run is a filename, read from that file
    """
```

Object attributes (in addition to those inherited)

```
p_lognorm:  normalized p, accounting for logarithmic bin spacing of T1
n:          number of T1 values in array within 0.01*tau and 100*tau
T1:         user-specified T1 array
```

Object functions (in addition to those inherited)

```python
fit(alpha,n=1000,T1=None,maxiter=None)
    """
        Run the non-negative least squares algorithm for a single value of 
        alpha, the regularization parameter
    
        alpha:      Tikhonov regularization parameter (may be list or number)
                    Try somewhere between 1e2 and 1e8
        n:          number of T1 values in array within 0.01*tau and40214 100*tau
                    (ignored if T1 is not none)
        T1:         user-specified T1 array
        maxiter:    max number of iterations in solver
        
        returns
            same as ilt.fit
    """

draw_pnorm(alpha_opt)
    """
        Draw the normalized probabilty distribution, assyming a logarithmic
        distribution of T1. 
        
        alpha_opt:  optimal alpha to use in ILT procedure
    """
```

## Examples of usage

### A general ILT

```python
from bILT import ilt

# Make some test data
x = np.linspace(0, 1, 100)
y = x ** 2
dy = np.random.random(len(x)) * 0.1
 
# we're going to fit this with some linear combination of exponentials
fn = lambda x, w : w * x ** 2
  
# make the transformation object
trans = ilt(x, y, dy, fn)

# select a range of alphas to test
alpha = np.logspace(-1, 2, 100)

# select the distribution for which we want to find the appropriate weights
w = np.logspace(-2, 5, 50)

# find the probability distribution 
trans.fit(alpha, w)
  
# draw the diagnostic curves
trans.draw()

# draw the fit with alpha = 1
trans.draw(1)
```

### ILT of β-NMR SLR data

```python
from bILT import bILT

# setup
trans = bILT(40214, 2009)

# select a range of alphas to test
alpha = np.logspace(2, 8, 100)

# fit
trans.fit(alpha, 100)

# draw the diagnostic curves
trans.draw()

# draw the fit with alpha = 4e4
trans.draw(4e4)
```
