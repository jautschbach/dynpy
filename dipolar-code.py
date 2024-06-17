import pandas as pd
import string
import numpy as np
import scipy as sp
from scipy import signal
from scipy import constants
from scipy.integrate import cumtrapz
import math
#import dask
from ddrax import *
import gc



intraHH = pd.read_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/water-01-gas-intraHH.csv',
                        usecols=['label0','label1','frame','time',
                                 'dx','dy','dz','dr','molecule0','molecule1'],
                        dtype={'label0':'category','label1':'category','frame':'i8',
                               'time':'f8','dx':'f8','dy':'f8','dz':'f8',
                               'dr':'f8','molecule0':'i8','molecule1':'i8'})
#allHH = allHH[allHH['time']!='time']                        
#allHH = allHH.astype({'frame':'i8','time':'f8','dx':'f8','dy':'f8','dz':'f8','dr':'f8','molecule0':'i8','molecule1':'i8'})
#allHH = allHH[(allHH['frame']>20)]                                            
intraHH.sort_values(by='frame',inplace=True)

#interHH = allHH[allHH['molecule0']!=allHH['molecule1']]
#intraHH = allHH[allHH['molecule0']==allHH['molecule1']]

#alldd = cart_to_dipolar_from_df(allHH)

#alldd.to_csv('test.csv')
#del allHH
#interdd = cart_to_dipolar_from_df(interHH)
#del interHH
intradd = cart_to_dipolar_from_df(intraHH)
del intraHH

gc.collect()


#acf = alldd.reset_index().groupby(['label0','label1']).apply(correlate)
#del alldd
#interacf = interdd.reset_index().groupby(['label0','label1']).apply(correlate)
#del interdd
intraacf = intradd.reset_index().groupby(['label0','label1']).apply(correlate)
del intradd

gc.collect()

#N = len(pd.unique(acf[['label0','label1']].values.ravel('K')))
#interN = len(pd.unique(interacf[['label0','label1']].values.ravel('K')))
intraN = len(pd.unique(intraacf[['label0','label1']].values.ravel('K')))

#acf_avg = ((acf.groupby('time')['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$'].sum().dropna())*2/N).reset_index()
#del acf
#interacf_avg = ((interacf.groupby('time')['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$'].sum().dropna())*2/interN).reset_index()
#del interacf
intraacf_avg = ((intraacf.groupby('time')['$G_{Y2,0}$', '$G_{Y2,1}$', '$G_{Y2,2}$', '$G_{r}$','$G_{2,0}$','$G_{2,1}$','$G_{2,2}$'].sum().dropna())*2/intraN).reset_index()
del intraacf

gc.collect()

#EN = spec_dens_extr_narr(acf_avg)
#acf_avg.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-totalacf.csv')
#del acf_avg
#interEN = spec_dens_extr_narr(interacf_avg)
#interacf_avg.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-interacf.csv')
#del interacf_avg
intraEN = spec_dens_extr_narr(intraacf_avg)
intraacf_avg.to_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/DD-intra-ACF.csv')
del intraacf_avg

gc.collect()

#fullrax = dipolar(EN, 'H')
#interrax = dipolar(interEN, 'H')
intrarax = dipolar(intraEN, 'H')

gc.collect()

#rax = fullrax.join(intrarax,lsuffix='',rsuffix='intra')

#fullrax.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-totalrax.csv')
#interrax.to_csv('/projects/academic/jochena/adamphil/projects/dipolar/Paesani/QM-01-0.04dt-interrax.csv')
intrarax.to_csv('/projects/academic/jochena/adamphil/projects/SR/water/gas-MD/01/NVE/TIP3P/DD-intra-rax.csv')
