import pandas as pd
import numpy as np

from models.equations import velocity, sin, combined_force
from models.sparc.profile import MASS_RATIOS

def augment_df(sim, adf=None, R=None, G=None):
    # use rotmass as base df
    df = sim.profile.rotmass_df.copy()

    # append sparc table1 interesting parameters
    for k in ('D', 'e_D', 'Inc', 'e_Inc', 'Vflat', 'e_Vflat', 'Q', 'MHI', 'L[3.6]', 'Reff'):
        df[k] = sim.profile.sparc_dict[k]

    # with mass ratios from sparc paper by
    # https://arxiv.org/pdf/1606.09251.pdf eq.2
    mrs = MASS_RATIOS.copy()

    # then override params
    distance, inclination = None, None
    if adf is not None:
        # baseline Lelli
        if 'Ydisk' in adf:
            mrs={
                'disk': adf.Ydisk.values[0],
                'bul': adf.Ybul.values[0],
                'gas': 1.0,
            }
        if 'D' in adf:
            distance = adf.D.values[0]
        if 'Inc' in adf:
            inclination = adf.Inc.values[0]

    # store Y's
    if adf is not None and 'Ymass' in adf:
        Ymass = adf.Ymass.values[0]
        for k, v in mrs.items():
            mrs[k] *= Ymass
        df['Ymass'] = Ymass

    else:
        for key, value in mrs.items():
            df['Y%s' % key] = value
            ekey = 'e_Y%s' % key
            if adf is not None and ekey in adf:
                df[ekey] = adf[ekey].values[0]
            else:
                df[ekey] = value*10**0.1

    # Tau - add it as standard so can use it in all equation uses
    tau_key = 'tau'
    df[tau_key] = adf[tau_key].values[0] if adf is not None and tau_key in adf else 0

    # make distance and inclination adjustments
    # from equations 4 & 5 in
    # https://www.aanda.org/articles/aa/pdf/2018/07/aa32547-17.pdf
    R_orig = df['R']
    R = df['R']
    if distance is not None:
        dist_adjust = distance/sim.profile.sparc_dict['D']
        R = R*dist_adjust
        for c in sim.profile.labels:
            df['V%s' % c] = df['V%s' % c]*(dist_adjust**0.5)
        
        # update original sparc values to newly supplied versions
        df['R'] = R
        df['D'] = distance
    
    if inclination is not None:
        df['Vobs'] = df['Vobs']*sin(df['Inc'])/sin(inclination)
        df['Inc'] = inclination

    # calculate additional added needed for benchmarking rar
    df['Vbar'] = np.sum([mrs[c]*df["V%s" % c]**2 for c in sim.profile.labels],axis=0)**0.5
    df['Vgbar'] = df['Vbar']**2/R
    df['gobs'] = df['Vobs']**2/R

    # interp force values from simulation
    # for now using unitary mass ratios
    # so can use raw component data in mcmc
    sdf = sim.dataframe(mass_ratios=False)

    # use the point to the right (slightly larger)
    # to interp value (is always closer than linear interp)
    x_right_points = sim.profile.rotmass_x(sim.space)+1

    # when doing traditional scalar fields on multiple components
    # or when have combined them ahead of time for speed improvements
    components = sdf.groupby('component')
    if len(components) > 1:
        for label, cdf in components:
            cdf = cdf.set_index('x').loc[x_right_points]
    
            def itp(ilab):
                return np.interp(R, cdf['rd'], cdf[ilab])
    
            df['Fnewton_%s' % label] = itp('x_vec')
            df['Fscalar_%s' % label] = itp('F_scalar')
            df['S%s' % label] = velocity(R, df['Fnewton_%s' % label])
    
        # combine components
        df['Fnewton'] = combined_force(df, 'Fnewton', sim.mass_labels, mrs)
        df['Fscalar'] = combined_force(df, 'Fscalar', sim.mass_labels, mrs)
    else:
        cdf = sdf.set_index('x').loc[x_right_points]
        df['Fnewton'] = np.interp(R, cdf['rd'], cdf['x_vec'])
        if adf is not None and 'Ymass' in adf:
            df['Fnewton'] *= Ymass

    # S for simulated
    # later use P for predicted
    # Sbar from Sgbar
    df['Sgbar'] = df['Fnewton']
    df['Sbar'] = velocity(R, df['Sgbar'])
    
    # benchmark log bars, so can filter data
    # for a certain quality threshold
    df['VSdiff'] = (df['Sgbar']/df['Vgbar'])-1
    df['VSdiffabs'] = np.abs(df['VSdiff'])

    # additional helper variables
    df['rel_R'] = df['R']/df['R'].max()
    df['mhi_R'] = df['R']/df['MHI']

    return df


