import sys
sys.path.append("../")

from models.space import Space
from models.galaxy import Galaxy
from models.equations import cos
from references import sparc as sparc_imports

import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt


def generate_profiles():
    """
    Generates Mass Profiles for all galaxies possible 
    """
    # load all the data
    sparc_table1 = sparc_imports.sparc_df()
    guids = sparc_table1['Galaxy'].unique()
    mm_df = sparc_imports.massmodels_df()
    decomps = sparc_imports.decomp_dict(guids)
    rt = sparc_imports.rotmass_dict(guids)
    #rt_df = pd.concat(rt.values(), ignore_index=True)
    rar_df = sparc_imports.rar_df()
    
    # making new list of uniques, minus the missing files from rotmass
    guids = list(rt.keys())
    
    profiles = {}
    for uid in guids:
        profiles[uid] = SparcMassProfile(
            uid = uid,
            sparc_df = sparc_table1.query('Galaxy=="%s"' % uid),
            rar_df = rar_df.query('Galaxy=="%s"' % uid),
            decomps_df = decomps[uid],
            rotmass_df = rt[uid],
            mm_df = mm_df.query('Galaxy=="%s"' % uid),
        )
    return profiles


def df2dict(df):
    """ Like .to_dict() but avoids the index """
    return dict([(k, np.array(df[k].values)) for k in df.columns])

def dfrowdict(df):
    return dict([(k, list(v.values())[0]) for k,v in df.to_dict().items()])


COLOR_SCHEME = {'gas': 'teal', 'disk': 'darkorange', 'bul': 'mediumpurple', 'obs': 'black'}



class SparcMassProfile:
    """
    Mass Profile for Sparc data for a given galaxy
    """
    def __init__(self, uid,
            sparc_df, decomps_df, rotmass_df, mm_df, rar_df,
            auto_fit=True, extend_decomp=True):

        self.uid = uid
        self.sparc_dict = dfrowdict(sparc_df)
        self.rar_dict = dfrowdict(rar_df)

        self.decomps_df = decomps_df
        self.rotmass_df = rotmass_df
        self.mm_df = mm_df

        self.orig_decomps_df = decomps_df
        self.orig_rotmass_df = rotmass_df
        self.orig_mm_df = mm_df

        self.decomps_dict = df2dict(decomps_df)
        self.rotmass_dict = df2dict(rotmass_df)
        self.mm_dict = df2dict(mm_df)

        self.inc_used = None
        self.auto_fit = True
        self.extend_decomp = extend_decomp


    @property
    def is_bul(self):
        """ Does it have a bulge? """
        return np.sum(self.rotmass_dict['SBbul']) > 0
    
    @property
    def max_r(self):
        """ Last R recorded in profile data """
        return self.decomps_dict['R'].max()
    
    @property
    def is_valid(self):
        """ Do we want to include this galaxy in our analysis? """
        Q < 3
        Inc > 30
        Inc < 80

    def _decomps(self):
        """ Decomposition data """

        data = {}
        self.inc_used = {}

        # automatically fit decomposition profile
        # to the pre-decomposed points in the mass model
        # this to be updated when find out what the actual is
        fit_components = ['disk', 'bul'] if self.is_bul else ['disk',]
        for comp in fit_components:

            if self.auto_fit:
                def interop(xdata, deg):
                    return np.interp(xdata,
                        self.decomps_dict['R'],
                        (self.decomps_dict['SB%s' % comp])*cos(deg))
                
                inc = sp.optimize.curve_fit(interop, self.mm_dict['R'],
                            self.mm_dict['SB%s' % comp],
                            p0=self.rar_dict['Inc'],
                            bounds=[0,90])[0][0]
                
            else:
                inc = self.sparc_dict['Inc']

            self.inc_used[comp] = inc
            err = self.sparc_dict['e_Inc']
            adjs = [cos(inc), cos(inc+err), cos(inc-err)]
            
            R = list(self.decomps_dict['R'])
            profile = list(self.decomps_dict['SB%s' % comp])
    
            # make sure the data tampers to zero
            # otherwise nulled will have a harsh cusp
            if self.extend_decomp:
                R.append(R[-1]*2)
                profile.append(0)
    
            data[comp] = (R, [np.array(profile)*a for a in adjs])
        
        # gas
        # using rotmass data, rather than decomp data
        # has already been deprojected
        R = list(self.rotmass_dict['R'])
        gas = list(self.rotmass_dict['SBgas'])

        if self.extend_decomp:
            R.append(R[-1]*2)
            gas.append(0)

        data['gas'] = (R, [gas,])
        
        return data


        # run the rest
        err = self.sparc_dict['e_Inc']
        adjs = [cos(inc), cos(inc+err), cos(inc-err)]
        
        data = {}
        
        # disk
        R = list(self.decomps_dict['R'])
        disk = list(self.decomps_dict['SBdisk'])

        # make sure the data tampers to zero
        # otherwise nulled will have a harsh cusp
        if self.extend_decomp:
            R.append(R[-1]*2)
            disk.append(0)

        data['disk'] = (R, [np.array(disk)*a for a in adjs])
        
        # gas
        # using rotmass data, rather than decomp data
        # has already been deprojected
        R = list(self.rotmass_dict['R'])
        gas = list(self.rotmass_dict['SBgas'])

        if self.extend_decomp:
            R.append(R[-1]*2)
            gas.append(0)

        data['gas'] = (R, [gas,])
        
        # bulge
        # TODO: bulge not handled well
        # for analysis are avoiding for now
        if self.is_bul:
            data['bul'] = (self.decomps_dict['R'], [self.decomps_dict['SBbul']*a for a in adjs])
        
        return data

    def masses(self, space):
        """ Generates the mass profiles for a given space object """
        dc = self._decomps()
        r, z = space.rz()
        
        # need to adjust for scale
        # as density is in pc2
        # we need kpc and account for scaling
        scale = ((1000*space.scale)**2)
        
        masses = []
        for R, comp in dc.values():
            # interp raw sparc data
            # that data doesn't smooth
            masses.append(np.interp(r, R, comp[0])*scale)
            
        return np.array(masses), list(dc.keys())
    
    def rotmass_points(self, space):
        """ Returns the points to be analysed
        to give the most accurate interp values
        for the given set of rotmass or mass model 'R's
        """
        points = []
        c = space.center
        for i in np.floor(self.rotmass_dict['R']/space.scale).astype(int):
            points.append((c[0],c[1],c[2]+i))
            points.append((c[0],c[1],c[2]+i+1))
        return points


    def plot(self, ax=None, index=0):
        """ Plots the profile """
        dc = self._decomps()
        
        i = 0
        for label, comp in dc.items():
            r, data = comp
            color = COLOR_SCHEME[label]
            i += 1
            
            sns.scatterplot(data=self.rotmass_df, x='R', y='SB%s' % label, color=color, ax=ax, label="Rotmass")
            
            g = sns.lineplot(x=r, y=data[0], color=color, ax=ax, label=label)
            if len(data) > 1:
                g.fill_between(r, data[1], data[2], alpha=0.1, color=color)
        
        g.set(title="%s. %s sparc:%s, rar:%s, used:%s" % (index, self.uid, self.sparc_dict['Inc'], self.rar_dict['Inc'], self.inc_used))
        