# Generate data based on seperation between biexponential SLR rates
# Derek Fujimoto
# Mar 2020

from bILT.testing.data_iterator import data_iterator
import os
import numpy as np

# where to put the files
output_dir = "data/biexp_n6e8"

# set up the file names
filename = 'T1b_%.2f'

# run the data generator
d = data_iterator(output_dir)

# set constants
d.n = 6e8

# set up variable biexponential relaxation 
fn = '0.35 * (exp(-x) + exp(-%f*x))'
T1b = np.logspace(-1,2,25)

for i in T1b: 
    d.filename = (filename % float(i)).replace('.','p')
    d.fn = fn%i
    d.inputs_to_save = {'T1a (1/s)':1,'T1b (1/s)':float(i)}
    d.run()
