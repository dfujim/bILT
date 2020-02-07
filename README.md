# bILT
Inverse Laplace Transform objects, optimized for BNMR data

# Examples of usage: general ILT
```python
from bILT impoort ilt

# Make some test data
x = np.linspace(0,1,100)
y = x**2
dy = np.random.random(len(x))*0.1
 
# we're going to fit this with some linear combination of exponentials
fn = lambda x,w : w*x**2
  
# make the transformation object
trans = ilt(x,y,dy,fn)

# select a range of alphas to test
alpha = np.logspace(-1,2,100)

# select the distribution for which we want to find the appropriate weights
w = np.logspace(-2,5,50)

# find the probability distribution 
trans.fit(alpha,w)
  
# draw the diagnostic curves
trans.draw()

# draw the fit with alpha=1
trans.draw(1)
```

# Example of usage: BNMR ILT

```python
from bILT impoort bILT

# setup
trans = bILT(40214,2009)

# select a range of alphas to test
alpha = np.logspace(2,8,100)

# fit
trans.fit(alpha,100)

# draw the diagnostic curves
trans.draw()

# draw the fit with alpha=4e4
trans.draw(4e4)
```
