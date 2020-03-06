# Generate data based on stats
# Derek Fujimoto
# Mar 2020

from bILT.testing.data_iterator import data_iterator
import os

# where to put the files
output_dir = "data_stats"

# set up the file names
filename = 'n%d'

# we are varying the number of counts
n = 10**np.arange(3,6)

# run the data generator
d = data_iterator(output_dir)

# make sure we have single exponential relaxation 
d.fn = '0.7 * exp(-x)'

for i in n: 
    d.n = int(i)
    d.filename = filename % int(i)
    d.run()
