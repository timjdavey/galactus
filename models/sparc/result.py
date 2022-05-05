import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt

from models.load import load_sparc
from models.sparc.dataframe import augment_df
from models.sparc.profile import COLOR_SCHEME

threshold_label = 'Threshold'

DEBUG = {
    'Everything': 'R>0',
    threshold_label: 'VWdiffabs<%s',
    'Quality': 'Q<3 & Inc<80 & Inc>20',
    'Quality_%s' % threshold_label: 'VWdiffabs<%s & Q<3 & Inc<80 & Inc>20',
    'Anti_%s' % threshold_label: 'VWdiffabs>%s',
}

ANALYSIS = {}
ANALYSIS[threshold_label] = DEBUG[threshold_label]
ANALYSIS['Quality_%s' % threshold_label] = DEBUG['Quality_%s' % threshold_label]


class Result:
    """
    Wrapper class to do basic analysis on the whole SPARC set of results.
    """

    def __init__(self, adjustments=None, queries_strs=DEBUG,
                threshold=0.1, idens=('V','W'), simulations=None, full_interp=False):
        
        # load all that it can find
        if simulations is None:
            simulations = load_sparc()

        dfs = []
        for name, sim in simulations.items():
            if adjustments is None:
                dfs.append(augment_df(sim, full_interp=full_interp))
            else:
                gdf = adjustments.query("Galaxy=='%s'" % name)

                # if the galaxy isn't present in adjustments
                # because adjustments is from a trained MCMC
                # from a filtered dataset
                # would need to go back and retain on
                # set with global values of gamma, alpha etc
                if len(gdf) > 0:
                    dfs.append(augment_df(sim, gdf, full_interp=full_interp))

        self.dataframe = pd.concat(dfs, ignore_index=True)
        self.simulations = simulations
        self.queries_strs = queries_strs
        self.threshold = threshold
        self.idens = idens
        self.adjustments = adjustments

    @property
    def queries(self):
        quer = {}
        for k,v in self.queries_strs.items():
            quer[k] = v % self.threshold if threshold_label in k else v
        return quer

    def datasets(self):
        """
        Automatically return the dataframe filtered
        using the `query` param
        """
        return dict([(k, self.dataframe.query(q)) for k, q in self.queries.items()])

    def plot_thresholds(self):
        """
        Plots a histogram of the differences in V (sparc model) & W (simulated model)
        """
        datasets = self.datasets()

        # first plot the histograms
        fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(20,5))

        for i, name in enumerate(datasets.keys()):
            df = datasets[name]
            g = sns.histplot(df['VWdiff'], ax=axes[i])
            g.set(title=name)

        # next plot the histograms
        fig, axes = plt.subplots(1, len(datasets), sharex=True, sharey=True, figsize=(20,5))

        for i, name in enumerate(datasets.keys()):
            # then plot scatter residuals
            df = datasets[name]
            h = sns.scatterplot(data=df, x='rel_R', y='VWdiff',
                s=2, hue='MHI', palette='icefire', ax=axes[i])
            h.axhline(y=0, color='black', linestyle='dashed')

        # then plot the CDF for picking the right level
        fig, axes = plt.subplots(1, 1, figsize=(20,5))
        dfs = []
        for label, df in datasets.items():
            df = df[['VWdiffabs']].copy()
            df['set'] = label
            dfs.append(df)
        
        sns.ecdfplot(data=pd.concat(dfs, ignore_index=True), x='VWdiffabs', hue='set', linestyle='dotted', ax=axes)


    def plot_rar(self, kind=0, idens=None, query_key=None, line=[1,6]):
        """
        Plots various 
        kind == 0 is density plot
        kind == 1 is a rel_R plot
        kind == 2 is a regression plot
        """
        datasets = self.datasets()
        datakeys = [query_key,] if query_key else datasets.keys()
        if idens is None: idens = self.idens

        height = len(datakeys)
        width = len(idens)
        fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(20,10*height))

        # plot filter queries on each row
        for row, name in enumerate(datakeys):
            df = datasets[name]
            axrow = axes[row] if height > 1 else axes

            # plot references on each column
            for col, iden in enumerate(idens):
                ax = axrow[col]
                x = 'log_%sgbar' % iden
                y = 'log_gobs'

                 # rel_R coloured scatter
                if kind == 0:
                    g = sns.scatterplot(data=df, x=x, y=y,
                        alpha=1.0, s=3, hue='rel_R', palette='Spectral', ax=ax)

                # density
                elif kind == 1:
                    g = sns.scatterplot(data=df, x=x, y=y,
                        color='black', s=10, alpha=0.5, ax=ax)
                    sns.histplot(data=df, x=x, y=y,
                        bins=30, pthresh=.01, cmap="mako_r", alpha=0.6, ax=ax)
                    sns.kdeplot(data=df, x=x, y=y,
                        levels=4, color="w", linewidths=2, ax=ax)
                
                # regression
                elif kind == 2:
                    g = sns.regplot(data=df, x=x, y=y, order=1, ax=ax, x_bins=10)

                else:
                    raise ValueError("%s for kind is invalid" % kind)
                
                if col == 0:
                    g.set(title=name)
                
                # reference line
                sns.lineplot(x=line, y=line, color='grey', ax=ax, linestyle='dotted')


    def plot_rars(self, *args, **kwargs):
        """ Plots all the rar plots """

        for i in range(3):
            self.plot_rar(kind=i, *args, **kwargs)


    def plot_nullrelationship(self):
        """ Plots the nulled relationship """

        datasets = self.datasets()
        x = 'mhi_R'
        y = 'Fnulled'
        fig, axes = plt.subplots(1, len(datasets), figsize=(15,5))
        for i, name in enumerate(datasets.keys()):
            df = datasets[name]

            sns.scatterplot(data=df, x=x, y=y,
                color='black', s=10, alpha=0.2,
                ax=axes[i]).set(xscale='log', yscale='log', title=name)
            
            sns.histplot(data=df, x=x, y=y,
                bins=50, pthresh=.2, cmap="mako_r", alpha=0.6, ax=axes[i])


    def plot_residuals(self, idens=None, query_key=None,
            checks=('rel_R', 'Fnulled', 'mhi_R', 'R', 'D', 'MHI'), non_log=('rel_R',)):
        """ Plots the residuals """

        # only makes sense to do for one query group
        # visually would get messy otherwise 
        if query_key is None: query_key = list(self.queries.keys())[-1]
        df = self.datasets()[query_key]

        # it's likely you'll only want to plot for T
        if idens is None: idens = self.idens

        # return regression data
        data = []
        fig, axes = plt.subplots(len(checks), len(idens), figsize=(20,10*len(checks)))
        for col, iden in enumerate(idens):
            for row, c in enumerate(checks):
                ax = axes[row][col]

                if c in non_log: x = df[c]
                else: x = np.log10(df[c])

                y = np.log10(df['gobs']/df['%sbar' % iden])
                result = sp.stats.linregress(x, y)
                sns.scatterplot(x=x, y=y, color='black', s=10, alpha=0.5, ax=ax)
                sns.histplot(x=x, y=y, bins=30, pthresh=.05, cmap="mako_r", alpha=0.6, ax=ax)
                sns.lineplot(x=x, y=result.slope*x+result.intercept, color='red', ax=ax)
                ax.axhline(y=1, color='orange')
                data.append({
                    'iden': iden,
                    'check': c,
                    'r2': result.rvalue**2,
                    'rslope': result.slope,
                    'rstderr': result.stderr,
                    'rintercept': result.intercept
                })
            
        return pd.DataFrame(data)


    def plot_velocities(self, compare=None, count=None, profiles=False, sharex=True):

        def plot_sim(df, ax, idens=('V', 'W')):
           for key, color in COLOR_SCHEME.items():
               g = sns.scatterplot(data=df, x='R', y='V%s' % key, ax=ax, color=color, label='V%s' % key)
               if key == 'obs':
                   g.errorbar(df['R'], df['Vobs'], yerr=df['e_Vobs'], ecolor=color, fmt='.k')
                   sns.scatterplot(data=df, x='R', y='Vbar', ax=ax, color='grey', label='Vbar')
                   sns.lineplot(data=df, x='R', y='Wbar', ax=ax, color='grey', label='Wbar')
                   if 'Tbar' in df:
                       sns.lineplot(data=df, x='R', y='Tbar', ax=ax, color=color, label='Tbar')
               else:
                   sns.lineplot(data=df, x='R', y='W%s' % key, ax=ax, color=color, label='W%s' % key)
           return g

        # how many plots
        sharey = True
        columns = 1
        if compare: columns += 1
        if profiles:
            columns += 1
            sharey = False

        for i, group in enumerate(self.dataframe.groupby('Galaxy')):
            galaxy, gdf = group
            fig, axes = plt.subplots(1, columns, figsize=(20,8), sharex=sharex, sharey=sharey)
            
            if columns == 1:
                g = plot_sim(gdf, axes, self.idens)

            if compare or profiles:
                # plot this result object on the right
                # as it's the last most thing in the
                # analysis chain
                g = plot_sim(gdf, axes[columns-1], self.idens)

                j = 0
                if compare:
                    plot_sim(compare.dataframe[compare.dataframe['Galaxy']==galaxy], axes[columns-2], compare.idens)
                if profiles:
                    self.simulations[galaxy].profile.plot(axes[0], i)
            
            g.set(title=galaxy)
            
            if count and i == count-1: return
    
    
