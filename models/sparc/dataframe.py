import pandas as pd
import numpy as np

from models.load import load_sparc
from models.equations import velocity, sin, null_gravity


def load_analysis():
    simulations = load_sparc()
    dfs = [augment_df(sim, idx=i) for i, sim in enumerate(simulations.values())]
    return pd.concat(dfs, ignore_index=True)


IDENTIFIERS = {
    'V': 'Original SPARC mass models',
    'W': 'Simulation data of SPARC models',
    #'T': 'Null adjusted'
}


def augment_df(sim, mrs=None, distance=None, inclination=None, idx=None):
    
    # use rotmass as base df
    df = sim.profile.rotmass_df.copy()

    # index galaxy
    if idx is not None:
        df['gidx'] = idx

    # append sparc table1 interesting parameters
    for k in ('D', 'e_D', 'Inc', 'e_Inc', 'Vflat', 'e_Vflat', 'Q', 'MHI', 'L[3.6]'):
        df[k] = sim.profile.sparc_dict[k]
    
    # additional calculations based on that data
    df['u_Inc'] = sim.profile.inc_used
    df['rel_R'] = df['R']/df['R'].max()
    df['mhi_R'] = df['R']/df['MHI']

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
        R = R*inc_adjust
        for c in sim.mass_labels:
            df['V%s' % c] = df['V%s' % c]*(inc_adjust**0.5)
    if inclination is not None:
        df['Vobs'] = df['Vobs']*sin(df['Inc'])/sin(inclination)

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

    # benchmark log bars, so can filter data
    # for a certain quality threshold
    df['VWdiff'] = (full_df['log_V_gbar']-full_df['log_W_gbar'])**0.5

    # finally calculate the gbars
    for v in IDENTIFIERS.keys():
        key = '%s_gbar' % v
        df[key] = df['%sbar' % v]**2/R
        df['log_%s' % key] = np.log10(df[key])

    return df


