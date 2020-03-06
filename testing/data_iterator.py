# Allow for the iteration of data generation 
# Derek Fujimoto
# Mar 2020

import yaml,os,sys
import datetime 

sys.path.append('data_generator_pp/')
from output2csv import output2csv

class data_iterator(object):

    # Default settings 
    lifetime = 1.2096
    fn = '0.7 * exp(-x)'
    beam_pulse = 4.0
    A_beta = -1/3
    n = 1e6
    hist_nbins = 1600
    hist_tmin = 0.0
    hist_tmax = 16.0
    filename = 'output'
    
    def __init__(self,output_dir):
        self.output_dir = output_dir
        
    def run(self):
        
        # make output directory
        os.makedirs(self.output_dir, exist_ok=True)
        os.chdir(self.output_dir)
        
        # generate run file
        yaml_dict = {
                'lifetime (s)':                 self.lifetime,
                'polarization function f(x)':   self.fn,
                'beam pulse (s)':               self.beam_pulse,
                'A_beta':                       self.A_beta,
                'n':                            self.n,
                'output':                       self.filename+'.root',
                'histogram n_bins':             self.hist_nbins,
                'histogram t_min':              self.hist_tmin,
                'histogram t_max':              self.hist_tmax
                }
    
        with open(self.filename+'.yaml','w') as fid:
            yaml.safe_dump(yaml_dict,fid)

        # run 
        os.system('../data_generator_pp/data_generator_pp %s.yaml' % self.filename)

        # make header for csv
        header = ['Monte Carlo simulation of β-NMR data',
                  'Date run: %s' % str(datetime.datetime.now()),
                  '']
        header.extend(['%s: %s' % (k,v) for k,v in yaml_dict.items()])
        header = '\n# '.join(header)
        header = '# ' + header + '\n'
        
        # convert to csv
        output2csv(self.filename+'.root',header)

        # remove root file 
        os.remove(self.filename+'.root')
