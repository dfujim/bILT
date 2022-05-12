# Generating Test Data

Data generation is done at the base level by the [data_generator](https://github.com/dfujim/bILT/tree/master/testing/data_generator_pp). To compile the needed files go to `bILT/testing/data_generator_pp` and call `make`.

To improve ease of usage in generating data, use the [data_iterator](https://github.com/dfujim/bILT/blob/master/testing/data_iterator.py) object.

Basic Example (see also [gen_data_stats](https://github.com/dfujim/bILT/blob/master/testing/gen_data_stats.py)):

```python
from bILT.testing.data_iterator import data_iterator

# where to put the files
output_dir = "data/stats"

# set up the file naming pattern (for both input and output files)
filename = 'n%d'

# vary the number of counts
n = 10**np.arange(3,6)

# make the data iterator
d = data_iterator(output_dir)

# set the relaxation function (can set any and all parameters in this way)
d.fn = '0.7 * exp(-x)'

# run 
for i in n: 
    d.n = int(i)                      # set the number of counts
    d.filename = filename % int(i)    # set the input/output file names
    d.run()                           # run the simulation
```
