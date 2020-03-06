# Generate data based on changes in initial polarization
# Derek Fujimoto
# Mar 2020

from bILT.testing.data_iterator import data_iterator
import os

# where to put the files
output_dir = "data/amp_n6e8"

# set up the file names
filename = 'amp%.1f'

# run the data generator
d = data_iterator(output_dir)

# set constants
d.n = 6e5

# set up variable biexponential relaxation 
fn = '%f * exp(-x)'
amp = np.arange(0.1,1.1,0.1)

for i in amp: 
    d.filename = (filename % float(i)).replace('.','p')
    d.fn = fn%i
    d.inputs_to_save = {'T1 (1/s)':1,'amp':float(i)}
    d.run()
