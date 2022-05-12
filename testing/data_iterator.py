# Allow for the iteration of data generation 
# Derek Fujimoto
# Mar 2020

import yaml,os,sys
import datetime 
from bILT.testing.output2csv import output2csv

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
    
    def __init__(self,output_dir,inputs_to_save=None):
        """
            output_dir:         save files to this location
            inputs_to_save:     dict, if not none, write this to yaml and csv
                                ex: {"T1":14.23}
        """
        self.output_dir = output_dir
        self.inputs_to_save = inputs_to_save
        
    def run(self):
        
        # make output directory
        os.makedirs(self.output_dir, exist_ok=True)
        rootfile = self.filename+'.root'
        csvfile = self.filename+'.csv'
        yamlfile = self.filename+'.yaml'
        
        # generate run file
        yaml_dict = {
                'lifetime (s)':                 self.lifetime,
                'polarization function f(x)':   self.fn,
                'beam pulse (s)':               self.beam_pulse,
                'A_beta':                       self.A_beta,
                'n':                            int(self.n),
                'output':                       rootfile,
                'histogram n_bins':             self.hist_nbins,
                'histogram t_min':              self.hist_tmin,
                'histogram t_max':              self.hist_tmax
                }
    
        # add inputs to save
        if self.inputs_to_save is not None: 
            yaml_dict = {**yaml_dict,**self.inputs_to_save}
    
        # write config file
        with open(self.filename+'.yaml','w') as fid:
            yaml.safe_dump(yaml_dict,fid)

        # run 
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(path, 'data_generator_pp', 'data_generator_pp')
        os.system(f'{path} {yamlfile}')

        # make header for csv
        header = ['Monte Carlo simulation of β-NMR data',
                  'Date run: %s' % str(datetime.datetime.now()),
                  '']
        header.extend(['%s: %s' % (k,v) for k,v in yaml_dict.items()])
        header = '\n# '.join(header)
        header = '# ' + header + '\n'
        
        # convert to csv
        output2csv(rootfile,header)

        # remove root file 
        os.remove(rootfile)
        
        # move files
        os.rename(csvfile,os.path.join(self.output_dir,csvfile))
        os.rename(yamlfile,os.path.join(self.output_dir,yamlfile))
