import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt

from models.sparc.dataframe import augment_df
from models.load import load_sparc


SPARC_THRESHOLD = 'VWdiff<0.1'
# should automatically catch
# Vbul==0.0 (only where there is no bul)
# and rel_R > 0.1 + rel_R < 0.9 where we had accuracy issues

QUERIES = {
    'Everything': 'R>0',
    'Thresholded': SPARC_THRESHOLD,
    'Quality': 'Q<3 & Inc<80 & Inc>20',
    'Quality Thresholded': '%s & Q<3 & Inc<80 & Inc>20' % SPARC_THRESHOLD,
}

QUALTIY = QUERIES.copy()
del QUALTIY['Everything']

class Result:
    """
    Wrapper class to do basic analysis on the whole SPARC set of results.
    """

    def __init__(self, queries=QUERIES, idens=('V','W'), adjustments=None, simulations=None):
        
        # load all that it can find
        if simulations is None:
            simulations = load_sparc()

        dfs = []
        for name, sim in simulations.items():
            if adjustments is None:
                dfs.append(augment_df(sim))
            else:
                gdf = adjustments.query("Galaxy=='%s'" % name)
                dfs.append(augment_df(sim,
                    mrs={
                        'disk': gdf.Ydisk.values[0],
                        'bul': gdf.Ybul.values[0],
                        'gas': 1.0,
                    },
                    distance=gdf.D.values[0],
                    inclination=gdf.Inc.values[0]))

        self.dataframe = pd.concat(dfs, ignore_index=True)
        self.simulations = simulations
        self.queries = queries
        self.idens = idens
        self.adjustments = adjustments


    def datasets(self):
        """
        Automatically return the dataframe filtered
        using the `query` param
        """
        return dict([(k, self.dataframe.query(q)) for k, q in self.queries.items()])


    def apply_gT(self, iden='T'):
        """
        Calculates the g values for a given set of Tbar's
        (null adjusted Velocity calculations)
        """
        key = '%sgbar' % iden
        self.dataframe[key] = self.dataframe['%sbar' % iden]**2/R
        self.dataframe['log_%s' % key] = np.log10(self.dataframe[key])


    def plot_thresholds(self):
        """
        Plots a histogram of the differences in V (sparc model) & W (simulated model)
        """
        datasets = self.datasets()
        fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(20,5))

        for i, name in enumerate(datasets.keys()):
            df = datasets[name]
            g = sns.histplot(df['log_Vgbar']-df['log_Wgbar'], ax=axes[i])
            g.set(title=name)


    def plot_rar(self, kind=0, idens=None, line=[1,6]):
        """
        Plots various 
        kind == 0 is density plot
        kind == 1 is a rel_R plot
        kind == 2 is a regression plot
        """
        datasets = self.datasets()
        if idens is None: idens = self.idens

        height = len(datasets)
        width = len(idens)
        fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(20,10*height))

        # plot filter queries on each row
        for row, name in enumerate(datasets.keys()):
            df = datasets[name]

            # plot references on each column
            for col, iden in enumerate(idens):
                ax = axes[row][col]
                x = 'log_%sgbar' % iden
                y = 'log_gobs'

                # density
                if kind == 0:
                    g = sns.scatterplot(data=df, x=x, y=y,
                        color='black', s=10, alpha=0.5, ax=ax)
                    sns.histplot(data=df, x=x, y=y,
                        bins=30, pthresh=.01, cmap="mako_r", alpha=0.6, ax=ax)
                    sns.kdeplot(data=df, x=x, y=y,
                        levels=4, color="w", linewidths=2, ax=ax)
                
                # rel_R coloured scatter
                elif kind == 1:
                    g = sns.scatterplot(data=df, x=x, y=y,
                        alpha=0.2, hue='rel_R', palette='Spectral', ax=ax)

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

