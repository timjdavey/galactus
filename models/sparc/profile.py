import sys
sys.path.append("../")

from models.space import Space
from models.galaxy import Galaxy
from models.equations import cos, velocity
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
    def fit_components(self):
        return ['disk', 'bul'] if self.is_bul else ['disk',]

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

    def _decomps(self, incs=None):
        """ Decomposition data """

        data = {}
        self.inc_used = {}

        # automatically fit decomposition profile
        # to the pre-decomposed points in the mass model
        # this to be updated when find out what the actual is
        for comp in self.fit_components:
            # if passing always override
            if self.auto_fit:
                if incs is None or comp not in incs:
                    def interop(xdata, deg):
                        return np.interp(xdata,
                            self.decomps_dict['R'],
                            (self.decomps_dict['SB%s' % comp])*cos(deg))
                    
                    inc = sp.optimize.curve_fit(interop, self.mm_dict['R'],
                                self.mm_dict['SB%s' % comp],
                                p0=self.rar_dict['Inc'],
                                bounds=[0,90])[0][0]
                else:
                    inc = incs[comp]
            else:
                inc = self.sparc_dict['Inc']

            self.inc_used[comp] = inc
            err = self.sparc_dict['e_Inc']
            adjs = cos(inc)
            
            R = list(self.decomps_dict['R'])
            profile = list(self.decomps_dict['SB%s' % comp])
    
            # make sure the data tapers to zero
            # otherwise nulled will have a harsh cusp
            if self.extend_decomp:
                R.append(R[-1]*2)
                profile.append(0)
    
            data[comp] = (R, np.array(profile)*adjs)
        
        # gas
        # using rotmass data, rather than decomp data
        # has already been deprojected
        R = list(self.rotmass_dict['R'])
        gas = list(self.rotmass_dict['SBgas'])

        if self.extend_decomp:
            R.append(R[-1]*2)
            gas.append(0)

        data['gas'] = (R, gas)
        
        return data

    def masses(self, space):
        """ Generates the mass profiles for a given space object """
        dc = self._decomps()
        r, z = space.rz()
        
        # if choosing to do 2d galaxy, where z = 1
        flat = space.points[0] == 1

        # need to adjust for scale
        # as density is in pc2
        # we need kpc and account for scaling
        scale = ((1000*space.scale)**2)
        
        masses = []
        for label, decomp in dc.items():
            R, data = decomp
            
            if flat:
                # interp raw sparc data
                # that data doesn't smooth
                # if donut left=0, right=0
                # decomp[0] is the standard
                m = np.interp(r, R, data)*scale
            else:
                if label == 'disk':
                    # z to project exponentially
                    scale_height = 0.5
                    m = np.interp(r, R, data)*scale*np.exp(-z/scale_height)
                elif label == 'bul':
                    # projects as sphere
                    m = np.interp(r+z, R, data)*scale*np.exp(-(r+z)/2)
                elif label == 'gas':
                    # do not project gas
                    # just keep as flat in centre
                    m = space.blank()
                    m[space.center[0]] = np.interp(r[0], R, data)*scale
                else:
                    raise ValueError("Unknown component %s" % label)
            
            masses.append(m)
        
        return np.array(masses), list(dc.keys()) 
    
    def fit_simulation(self, simulation):
        """ Fits a simulation to the Lelli mass model velocity components """
        
        fits = {}
        x_points = self.rotmass_x(simulation.space)+1
        
        for i, component in enumerate(self.fit_components):
            def interop(rot_r, ratio):
                mrs = dict([(c, 1) for c in ('disk', 'gas', 'bul')])
                mrs[component] = ratio
                df = simulation.dataframe(mrs)
                cdf = df.query('component=="%s"' % component).set_index('x').loc[x_points]
                return velocity(rot_r, np.interp(rot_r, cdf['rd'], cdf['x_vec']))
            
            fit, _ = sp.optimize.curve_fit(interop, self.rotmass_dict['R'],
                                    self.rotmass_dict['V%s' % component])
            
            fits[component] = fit[0]
        return fits

    def rotmass_x(self, space):
        points = []
        c = space.center
        R = self.rotmass_dict['R']
        s = space.scale
        return np.array([c[2]+i for i in np.floor(R/s).astype(int)])

    def rotmass_points(self, space, left=False, right=True):
        """ Returns the points to be analysed
        to give the most accurate interp values
        for the given set of rotmass or mass model 'R's
        """
        points = []
        c = space.center
        for x in self.rotmass_x(space):
            if left: points.append((c[0],c[1],x))
            if right: points.append((c[0],c[1],x+1))
        return points

    def plot(self, ax=None, index=0):
        """ Plots the profile """
        dc = self._decomps()
        rot_df = self.rotmass_df

        i = 0
        for label, decomp in dc.items():
            color = COLOR_SCHEME[label]
            i += 1
            
            # reference points
            sns.scatterplot(data=rot_df, x='R', y='SB%s' % label, color=color, ax=ax, label="Rotmass")
            
            # decomp data
            r, data = decomp
            g = sns.lineplot(x=r, y=data, color=color, ax=ax, label=label)
        
        g.set(title="%s. %s" % (index, self.uid))






