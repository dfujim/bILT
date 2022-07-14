# Generating Test Data

Data generation is done at the base level by the [data_generator](https://github.com/dfujim/bILT/tree/master/testing/data_generator_pp). To compile the needed files go to `bILT/testing/data_generator_pp` and call `make`.

To improve ease of usage in generating data, use the [data_iterator](https://github.com/dfujim/bILT/blob/master/testing/data_iterator.py) object.

Basic Example (see also [gen_data_stats](https://github.com/dfujim/bILT/blob/master/testing/gen_data_stats.py)):

```python
from bILT.testing.data_iterator import data_iterator

# where to put the files
output_dir = "data"

# make the data iterator
d = data_iterator(output_dir)

# set the relaxation function (can set any and all parameters in this way)
d.fn = '0.7 * exp(-x)'

# run 
d.n = 1e7                         # set the number of counts
d.filename = 'test'               # set the input/output file names
d.run()                           # run the simulation
```

Now let's look at the file we created:

```python
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

df = pd.read_csv('data/test.csv')
plt.errorbar(df.t, df.Ac, df.dAc, fmt='.')
```

We can now fit this with bILT:

```python
from bILT import bILT_test
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# set up testing with multiprocessing (omit nproc flag for single processor usage)
test_ilt = bILT_test('data/test', nproc=8)

# pick a range of 100 alphas to fit, on a logrithmic scale
alpha = np.logspace(2, 8, 100)

# fit all 100 alphas using the ILT method
test_ilt.fit(alpha, maxiter=1000)

# draw the diagnostic curves
test_ilt.draw()
```

Note that the L curve plot has a mouse-over functionality
