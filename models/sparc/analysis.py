import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az
from functools import cached_property

from models.equations import velocity, combined_force, null_gravity
from models.sparc.result import Result

class Analysis:

    UNIVERSE = ['gamma', 'alpha', 'epsilon']
    GALAXY = ['Inc', 'D', 'N', 'Ydisk', 'Ybul', 'tau']

    def __init__(self, name, model, null_function=null_gravity):
        self.name = name
        self.model = model
        self.null_function = null_function
        self.uni = None
        self.adjs = None

    @cached_property
    def params_galaxy(self):
        return [x for x in self.GALAXY if hasattr(self.model, x)]

    @cached_property
    def params_uni(self):
        return [x for x in self.UNIVERSE if hasattr(self.model, x)]

    @cached_property
    def MAP(self):
        """ Quick find MAP method """
        with self.model:
            return pm.find_MAP()

    @property
    def trace(self):
        if hasattr(self, '_trace'):
            return self._trace
        else:
            return self.sample()

    @cached_property
    def az_summary(self):
        return az.summary(trace, fmt='long')
    
    def adjustment_MAP(self):
        """ Same as adjustment_df but uses
        the fast MAP params rather than
        the full posterior mean values """
        data = {'Galaxy': self.model.coords['Galaxy']}
        for p in self.params_galaxy:
            data[p] = self.MAP[p]
        df = pd.DataFrame(data)
        df['Source'] = self.name
        return df

    def adjustment_FULL(self):
        """ The per galaxy settings to create a new
        Result set for Inc, Yetcs """
        summary = self.az_summary
        adjustments = []
    
        for galaxy, gdf in summary.groupby('Galaxy'):
            data = {'Galaxy': galaxy, 'Source': self.name}
            for param in self.params_galaxy:
                data[param] = gdf[param]['mean']
                data['e_%s' % param] = gdf[param]['sd']
            adjustments.append(data)
        return pd.DataFrame(adjustments)

    def params(self, fast=False):
        if fast:
            self.adjs = self.adjustment_MAP()
            self.uni = dict([(p, self.MAP[p]) for p in self.params_uni])
        else:
            self.adjs = self.adjustment_FULL()
            mean = self.summary.reset_index().query('index=="mean"')
            self.uni = dict([(p, mean[p].mean()) for p in self.params_uni])
        return self.adjs, self.uni

    def sample(self, draws=500, cores=4, tune=None):
        """ Samples the model """
        if tune is None: tune == draws*2
        with self.model:
            self._trace = pm.sample(tune=tune, draws=draws,
                cores=cores, return_inferencedata=True,
                target_aceept=0.9, start=self.MAP())
        return self._trace

    def plot_posterior(self, galaxy=False):
        if galaxy:
            az.plot_posterior(self.trace)
        else:
            az.plot_posterior(self.trace, var_names=self.params_uni)

    def Result(self, fast=False):
        """ Generates a result object, using a fast MAP or default full sample """

        adjs, uni = self.params(fast)
        result = Result(adjustments=adjs)
        
        # don't assume that there are universal params
        if uni:
            # once you've done the adjustments via inputting it into a Results object
            # this will update the mass ratios, R etc for us
            # so can safely use here
            T_force = self.null_function(result.dataframe['Fnewton'], result.dataframe['Fnulled'], **uni)
            result.dataframe['Tgbar'] = T_force
            result.dataframe['log_Tgbar'] = np.log10(T_force)
            result.dataframe['Tbar'] = velocity(result.dataframe['R'], T_force)
            result.idens=('V','W','T')

        return result




