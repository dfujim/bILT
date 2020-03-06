# Generate data based on stretching exponent
# Derek Fujimoto
# Mar 2020

from bILT.testing.data_iterator import data_iterator
import os

# where to put the files
output_dir = "data/strexp_n6e8"

# set up the file names
filename = 'beta%.1f'

# run the data generator
d = data_iterator(output_dir)

# set constants
d.n = 6e8

# set up variable biexponential relaxation 
fn = '0.7 * exp(-std::pow(x,%f))'
beta = np.arange(0.3,1,0.1)

for i in beta: 
    d.filename = (filename % float(i)).replace('.','p')
    d.fn = fn%i
    d.inputs_to_save = {'T1 (1/s)':1,'beta':float(i)}
    d.run()
