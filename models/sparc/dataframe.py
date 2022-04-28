import pandas as pd
import numpy as np

from models.equations import velocity, sin, null_gravity


def augment_df(sim, mrs=None, distance=None, inclination=None):
    
    # use rotmass as base df
    df = sim.profile.rotmass_df.copy()

    # append sparc table1 interesting parameters
    for k in ('D', 'e_D', 'Inc', 'e_Inc', 'Vflat', 'e_Vflat', 'Q', 'MHI', 'L[3.6]'):
        df[k] = sim.profile.sparc_dict[k]

    # with mass ratios from sparc paper by default
    # https://arxiv.org/pdf/1606.09251.pdf eq.2
    if mrs is None:
        mrs = {'disk': 0.5, 'gas': 1.0, 'bul': 0.7}

    # make distance and inclination adjustments
    # from equations 4 & 5 in
    # https://www.aanda.org/articles/aa/pdf/2018/07/aa32547-17.pdf
    R = df['R']
    if distance is not None:
        dist_adjust = distance/sim.profile.sparc_dict['D']
        R = R*dist_adjust
        for c in sim.mass_labels:
            df['V%s' % c] = df['V%s' % c]*(dist_adjust**0.5)
        
        # update original sparc values to newly supplied versions
        df['R'] = R
        df['D'] = distance
    
    if inclination is not None:
        df['Vobs'] = df['Vobs']*sin(df['Inc'])/sin(inclination)
        df['Inc'] = inclination

    # calculate additional added needed for benchmarking rar
    df['Vbar'] = np.sum([mrs[c]*df["V%s" % c]**2 for c in sim.mass_labels],axis=0)**0.5
    df['gobs'] = df['Vobs']**2/R
    df['log_gobs'] = np.log10(df['gobs'])

    # interp force values from simulation
    sdf = sim.dataframe(mass_ratios=mrs, combined=True)
    df['Fnewton'] = np.interp(R, sdf['rd'], sdf['x_vec'])
    df['Fabs'] = np.interp(R, sdf['rd'], sdf['x_abs'])
    df['Fnulled'] = df['Fabs']-df['Fnewton']

    # calculate Wbar from simulation force values
    df['Wbar'] = velocity(R, df['Fnewton'])

    # finally calculate the gbars
    for v in ('V', 'W'):
        key = '%sgbar' % v
        df[key] = df['%sbar' % v]**2/R
        df['log_%s' % key] = np.log10(df[key])

    # benchmark log bars, so can filter data
    # for a certain quality threshold
    df['VWdiff'] = np.abs(df['log_Vgbar']-df['log_Wgbar'])

    # additional helper variables
    df['rel_R'] = df['R']/df['R'].max()
    df['mhi_R'] = df['R']/df['MHI']

    return df


