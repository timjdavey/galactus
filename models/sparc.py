import sys
sys.path.append("../")

from models.space import Space
from models.galaxy import Galaxy
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
            mm_df = mm_df.query('ID=="%s"' % uid),
        )
    return profiles


def generate_galaxy(profile, space_points=5000, calc_points=20, cp=None):
    """
    Generates a galaxy given a profile
    """
    uid = profile.uid
    space = Space((1,space_points,space_points), profile.max_r*4/space_points)
    masses, labels = profile.masses(space)
    
    sim = Galaxy(masses, space, mass_labels=labels, cp=cp)
    sim.profile = profile
    sim.name = uid
    
    if calc_points:
        points = sim.radius_points(profile.max_r*1.5, calc_points)
    else:
        points = profile.rotmass_points(space)
    
    sim.analyse(points)
    return sim



def df2dict(df):
    """ Like .to_dict() but avoids the index """
    return dict([(k, np.array(df[k].values)) for k in df.columns])

def dfrowdict(df):
    return dict([(k, list(v.values())[0]) for k,v in df.to_dict().items()])

def cos(deg, func=np.cos):
    """ Do cos in degrees rather than radians """
    return func(deg*np.pi/180)

COLOR_SCHEME = {'gas': 'teal', 'disk': 'darkorange', 'bul': 'mediumpurple', 'obs': 'black'}



class SparcMassProfile:
    """
    Mass Profile for Sparc data for a given galaxy
    """
    def __init__(self, uid,
            sparc_df, decomps_df, rotmass_df, mm_df, rar_df,
            auto_fit=True, rar_fit=False, extend_decomp=True):

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

        self.auto_fit = auto_fit
        self.rar_fit = rar_fit
        self.extend_decomp = extend_decomp
        self.inc_used = None


    def fit_rar_data(self, do_rar):
        """
        Feels weird doing it this way,
        but here we've got to the option to call a function
        to update the data using rar_fit

        But runs similar to auto_fit,
        where you can mark the rar_fit bool
        so stays consistent
    
        Also, is a good way to reset all the df
        values without having to think about
        which df's to use

        Eq.4 & 5
        """
        df_keys = ('decomps', 'rotmass', 'mm')

        if do_rar:
            # R's
            dprimed = self.rar_dict['D']/self.sparc_dict['D']
            
            # make copies of the original dfs
            dfs = {}
            for df_key in df_keys:
                dfs[df_key] = getattr(self, 'orig_%s_df' % df_key).copy()

            for df in dfs.values():
                df['R'] = df['R']*dprimed
    
            # D's
            dfs['mm']['D'] = self.rar_dict['D']
    
            # adjust V components
            for df_name in ('rotmass', 'mm'):
                df = dfs[df_name]
                for component in ('Vgas', 'Vbul', 'Vdisk'):
                    df[component] = df[component]*(dprimed**0.5)
    
            # Vobs Eq.5
            iiprime = cos(self.sparc_dict['Inc'], np.sin)/cos(self.rar_dict['Inc'], np.sin)
    
            for df_name in ('rotmass', 'mm'): 
                df = dfs[df_name] 
                df['Vobs'] = df['Vobs']*iiprime
                df['Vobs'] = df['e_Vobs']*iiprime
            
            for df_key in df_keys:
                setattr(self, '%s_df' % df_key, dfs[df_key])
                setattr(self, '%s_dict' % df_key, df2dict(dfs[df_key]))
        
        else:
            # reset to the original values
            for df_key in df_keys:
                odf = getattr(self, 'orig_%s_df' % df_key)
                setattr(self, '%s_df' % df_key, odf)
                setattr(self, '%s_dict' % df_key, df2dict(odf))


    @property
    def is_bul(self):
        """ Does it have a bulge? """
        return np.sum(self.rotmass_dict['SBbul']) > 0
    
    @property
    def max_r(self):
        """ Last R recorded in profile data """
        return self.decomps_dict['R'][-1]
    
    @property
    def is_valid(self):
        """ Do we want to include this galaxy in our analysis? """
        Q < 3
        Inc > 30
        Inc < 80

    def _decomps(self):
        """ Decomposition data """
        
        # then update the values to rar_fit if need be
        self.fit_rar_data(self.rar_fit)

        # automatically fit decomposition profile
        # to the pre-decomposed points in the mass model
        # this to be updated when find out what the actual is
        if self.auto_fit:
            def interop(xdata, deg):
                return np.interp(xdata,
                    self.decomps_dict['R'],
                    self.decomps_dict['SBdisk'])*cos(deg)
            
            bounds = [0,90]
            inc = sp.optimize.curve_fit(interop, self.mm_dict['R'],
                        self.mm_dict['SBdisk'], bounds=bounds)[0][0]
        else:
            inc = self.sparc_dict['Inc']
        

        # run the rest
        self.inc_used = inc
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
    
    def mass_ratios(self):
        """
        Using sparc defaults
        Or Rar fits
        """
        defaults = {'disk': 0.5, 'gas': 1.0}
        if self.is_bul:
            defaults = {'bul': 0.7}

        if self.rar_fit:
            for key in defaults.keys():
                defaults[key] = self.rar_fit[key]
        else:
            return defaults

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
        
        g.set(title="%s. %s inc:%sdeg rar:%sdeg" % (index, self.uid, self.inc_used, self.rar_dict['Inc']))
        