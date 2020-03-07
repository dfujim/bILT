# Peak finding using a discriminator-like method
# Derek Fujimoto
# Mar 2020

import numpy as np
import matplotlib.pyplot as plt

def discriminator(z,p,threshold=None, crop_distance=10, draw=False):
    """
        Detect peaks, heights, and widths using a discriminator-like method. 
        
        z: The distribution x axis; i.e. the "T1" value. 
        p: Weights corresponding to z; i.e. the distribution 
        threshold: p must be greater than threshold to detect a peak
        crop_distance: remove this many bins from either side of z and p
        draw: if true, draw the results. 
    
    
        returns: [location_of_max, height_of_max, width_at_threshold]
    """

    # sort
    idx = np.argsort(z)
    z = z[idx]
    p = p[idx]

    # chop the ends: frequently has glitches here
    if crop_distance > 0:
        p = p[crop_distance:-crop_distance]
        z = z[crop_distance:-crop_distance]

    # threshold 
    if threshold is None:
        threshold = np.mean(p)*0.1

    # check if edges are rising (1) or falling (-1) or not transitioning (0)
    idx = p > threshold
    edge = idx.astype(int)
    edge = edge[1:]-edge[:-1]
    edge = np.append(edge,0)

    # trivial case: delta function 
    if sum(idx) == 1:
        loc = np.argmax(z)
        height = np.max(z)
        width = 0
        return (loc,height,width)

    # get indexes of rising/falling edges
    idx_rising = np.where(edge > 0)[0]
    idx_falling = np.where(edge < 0)[0]

    # check for falling edges that happen before the first rising edge
    while not all([f>r for f,r in zip(idx_falling,idx_rising)]):
        if len(idx_falling) > len(idx_rising):
            if idx_falling[0] < idx_rising[0]:
                idx_falling = idx_falling[1:]
            elif idx_falling[-1] < idx_rising[-1]:
                idx_falling = idx_falling[:-1]
        elif len(idx_falling) < len(idx_rising):
            if idx_falling[0] < idx_rising[0]:
                idx_rising = idx_rising[1:]
            elif idx_falling[-1] < idx_rising[-1]:
                idx_rising = idx_rising[:-1]

    assert(all([f>r for f,r in zip(idx_falling,idx_rising)]))

    # get peak values 
    peak_height = []
    peak_loc = []
    peak_width = []
    for rise,fall in zip(idx_rising,idx_falling):
        x = z[rise:fall]
        y = p[rise:fall]
        
        idx_max = np.argmax(y)
        peak_height.append(y[idx_max])
        peak_loc.append(x[idx_max])
        peak_width.append(x[-1]-x[0])

    # draw 
    if draw:
        plt.figure(figsize=(7.5,5))
        plt.loglog(z,p,label='Full Data Set')       
        plt.axhline(threshold,color='k',ls='--',label='Threshold')   
        plt.loglog(z[idx],p[idx],label='Data > Threshold')     
        plt.loglog(z[idx_rising],np.ones(len(idx_rising))*threshold,'C2^',ms=8,label='Rising Edge')   
        plt.loglog(z[idx_falling],np.ones(len(idx_falling))*threshold,'C3v',ms=8,label='Falling Edge')

        ymin = plt.ylim()[0]
        for h,l in zip(peak_height,peak_loc):
            plt.plot([l,l],[ymin,h],color='k',ls=':')
        plt.plot([l,l],[ymin,h],color='k',ls=':',label='Peak')
        plt.ylim(ymin,None)
        plt.ylabel('p')
        plt.xlabel('z')
        plt.legend(fontsize='xx-small',loc=0,bbox_to_anchor=(1,1))
        plt.tight_layout()
            
    return np.array((peak_loc,peak_height,peak_width))
