#!/usr/bin/python
# Convert root file to csv 
# Derek Fujimoto
# Mar 2020

import os, sys
import datetime
from ROOT import TH1D, TFile
import pandas as pd
import numpy as np


# =========================================================================== #
def output2csv(filename):

    # load file 
    fid = TFile(filename)

    # get histograms
    hist_names = {  "hF_p":'F+', 
                    "hB_p":'B+', 
                    "hB_m":'B-', 
                    "hF_m":'F-',
                    'hA':'Ac',
                    'hA_m':'A-',
                    'hA_p':'A+',}
    hist_names_err = {'d'+k:'d'+v for k,v in hist_names.items()}
    hists = {h:fid.Get(h) for h in hist_names}

    # extract content
    h0 = list(hists.values())[0]
    N = h0.GetNbinsX()
    t = np.array([h0.GetBinCenter(i) for i in range(N)])
    
    hist_content = {k:[h.GetBinContent(i) for i in range(N)] for k,h in hists.items()}
    hist_err = {'d'+k:[h.GetBinError(i) for i in range(N)] for k,h in hists.items()}
    
    # new filename
    filename2 = os.path.splitext(filename)[0]+".csv"
    
    # write header
    header = '\n'.join(('# Histograms and asymmetries for MC simulation of β-NMR data.',
                        '# Translated from hisotrams in ROOT file "%s"' % filename,
                        '# Translated on %s' % str(datetime.datetime.now()),
                        '# ',
                        '# Errors are marked with prefix "d"',
                        '# Ac denotes the four-counter combined asymmetry',
                        '# Times (t) are denoted in seconds',
                        '# \n'))
    
    with open(filename2,'w') as fout: 
        fout.write(header)
    
    # write to file
    df = pd.DataFrame({**hist_content,**hist_err},index=t)
    df.index.name = 't'
    df.rename(columns={**hist_names,**hist_names_err},inplace=True)
    df.to_csv(filename2,mode='a+')

# =========================================================================== #
if __name__ == "__main__":
    
    try:
        filename = sys.argv[1]
    except IndexError:
        print('Usage: output2csv.py filename')
    else:
        output2csv(filename)
