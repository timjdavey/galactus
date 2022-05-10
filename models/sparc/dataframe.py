import pandas as pd
import numpy as np

from models.equations import velocity, sin, null_gravity, combined_force


def augment_df(sim, adf=None, full_interp=False):
    # use rotmass as base df
    df = sim.profile.rotmass_df.copy()

    # append sparc table1 interesting parameters
    for k in ('D', 'e_D', 'Inc', 'e_Inc', 'Vflat', 'e_Vflat', 'Q', 'MHI', 'L[3.6]'):
        df[k] = sim.profile.sparc_dict[k]

    # with mass ratios from sparc paper by defaultadjustment
    # https://arxiv.org/pdf/1606.09251.pdf eq.2
    mrs = {'disk': 0.5, 'gas': 1.0, 'bul': 0.7}

    # then override params
    distance, inclination = None, None
    if adf is not None:
        # handle defaults
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
    for key, value in mrs.items():
        df['Y%s' % key] = value
        ekey = 'e_Y%s' % key
        if adf is not None and ekey in adf:
            df[ekey] = adf[ekey].values[0]
        else:
            df[ekey] = value*10**0.1

    # make distance and inclination adjustments
    # from equations 4 & 5 in
    # https://www.aanda.org/articles/aa/pdf/2018/07/aa32547-17.pdf
    R_orig = df['R']
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
    df['Vgbar'] = df['Vbar']**2/R
    df['log_Vgbar'] = np.log10(df['Vgbar'])

    df['gobs'] = df['Vobs']**2/R
    df['log_gobs'] = np.log10(df['gobs'])

    # interp force values from simulation
    # for now using unitary mass ratios
    # so can use raw component data in mcmc
    sdf = sim.dataframe(mass_ratios=False)
    for label in sim.mass_labels:
        if full_interp:
            cdf = sdf.query('component=="%s"' % label)
            df['Fnewton_%s' % label] = np.interp(R_orig, cdf['rd'], cdf['x_vec'])
            df['Fabs_%s' % label] = np.interp(R_orig, cdf['rd'], cdf['x_abs'])
        else:
            x_right_points = sim.profile.rotmass_x(sim.space)+1
            cdf = sdf[(sdf['component']==label) & (sdf['x'].isin(x_right_points))]
            df['Fnewton_%s' % label] = cdf['x_vec'].to_numpy()
            df['Fabs_%s' % label] = cdf['x_abs'].to_numpy()

        df['Fnulled_%s' % label] = df['Fabs_%s' % label]-df['Fnewton_%s' % label]
        df['W%s' % label] = velocity(R, df['Fnewton_%s' % label])

    # combine components
    df['Fnewton'] = combined_force(df, 'Fnewton', sim.mass_labels, mrs)
    df['Fabs'] = combined_force(df, 'Fabs', sim.mass_labels, mrs)
    df['Fnulled'] = combined_force(df, 'Fnulled', sim.mass_labels, mrs)

    # Wbar from Wgbar
    df['Wgbar'] = df['Fnewton']
    df['log_Wgbar'] = np.log10(df['Wgbar'])
    df['Wbar'] = velocity(R, df['Wgbar'])
    
    # benchmark log bars, so can filter data
    # for a certain quality threshold
    df['VWdiff'] = (df['Wgbar']/df['Vgbar'])-1
    df['VWdiffabs'] = np.abs(df['VWdiff'])

    # additional helper variables
    df['rel_R'] = df['R']/df['R'].max()
    df['mhi_R'] = df['R']/df['MHI']

    return df


