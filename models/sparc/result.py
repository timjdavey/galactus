import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
import matplotlib.pyplot as plt

from models.load import load_sparc
from models.sparc.dataframe import augment_df
from models.sparc.profile import COLOR_SCHEME
from models.equations import velocity


ANALYSIS = {
    'Everything': 'R>0',
    'Quality data': 'Q<2 & Inc<80 & Inc>20',
    'Quality simulation': 'Q<2 & Inc<80 & Inc>20 & rel_R>0.15 & rel_R<0.9',
}

COLORS = {
    'Everything': 'yellow',
    'Quality data': 'orange',
    'Quality simulation': 'red',
}


threshold_label = 'Threshold'

DEBUG = ANALYSIS.copy()
DEBUG[threshold_label] = 'VSdiffabs<%s'
DEBUG['Quality_%s' % threshold_label] = 'VSdiffabs<%s & Q<3 & Inc<80 & Inc>20'
DEBUG['Anti_%s' % threshold_label] = 'VSdiffabs>%s'


IDENS = {
    'V': 'Lelli',
    'S': 'Baryonic', # Simulated from decomps
    'P': 'Predicted', # Using adjusted values
}

class Result:
    """
    Wrapper class to do basic analysis on the whole SPARC set of results.
    """

    def __init__(self, simulations, adjustments=None, queries_strs=ANALYSIS, iden_labels=IDENS,
                threshold=0.1, idens=('S')):

        dfs = []
        for name, sim in simulations.items():
            if adjustments is None:
                dfs.append(augment_df(sim))
            else:
                gdf = adjustments.query("Galaxy=='%s'" % name)

                # if the galaxy isn't present in adjustments
                # because adjustments is from a trained MCMC
                # from a filtered dataset
                # would need to go back and retrain on
                # set with global values of gamma, alpha etc
                if len(gdf) > 0:
                    dfs.append(augment_df(sim, gdf))

        self.dataframe = pd.concat(dfs, ignore_index=True)
        self.simulations = simulations
        self.queries_strs = queries_strs
        self.iden_labels = iden_labels
        self.threshold = threshold
        self.idens = idens
        self.adjustments = adjustments

    @property
    def queries(self):
        quer = {}
        for k,v in self.queries_strs.items():
            quer[k] = v % self.threshold if threshold_label in k else v
        return quer

    def datasets(self, df=None):
        """
        Automatically return the dataframe filtered
        using the `query` param
        """
        df = self.dataframe if df is None else df
        return dict([(k, df.query(q)) for k, q in self.queries.items()])


    def statistics(self, g_not_vel=True, query_key=None, iden=None, weight=False, residuals=True):
        """
        Returns a selection of key statistics about the 
        """
        from sklearn.metrics import r2_score,\
            mean_absolute_error, mean_squared_error, mean_squared_log_error
        df = self.dataframe.copy()
        datasets = self.datasets()
        if iden is None: iden = self.idens[-1]
        datakeys = [query_key,] if query_key else datasets.keys()
        observations = {
            'little g': ('gobs', '%sgbar' % iden),
            'velocity': ('Vobs', '%sbar' % iden),
        }

        data = []
        for dk in datakeys:
            df = datasets[dk]
            for label, ys in observations.items():
                y_true, y_pred = df[ys[0]], df[ys[1]]
                weight = None
                stats = {
                    'dataset': dk,
                    'observation': label,
                    'r2': r2_score(y_true, y_pred, sample_weight=weight),
                    'RMSE': mean_squared_error(y_true, y_pred, sample_weight=weight, squared=False),
                    'MAE': mean_absolute_error(y_true, y_pred, sample_weight=weight),
                    'MLSE': mean_squared_log_error(y_true, y_pred, sample_weight=weight)
                }
                if residuals:
                    stats['Res(Log y)'] = self.residual(df, resid='%sgbar' % iden, iden=iden, plot=False).slope
                    stats['Res(Log mhi_R)'] = self.residual(df, resid='mhi_R', iden=iden, plot=False).slope
                
                data.append(stats)
        return pd.DataFrame(data=data)

    def plot_thresholds(self):
        """
        Plots a histogram of the differences in V (sparc model) & W (simulated model)
        """
        datasets = self.datasets()

        # first plot the histograms
        fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(20,5))

        for i, name in enumerate(datasets.keys()):
            df = datasets[name]
            g = sns.histplot(df['VSdiff'], ax=axes[i])
            g.set(title=name)

        # next plot the histograms
        fig, axes = plt.subplots(1, len(datasets), sharex=True, sharey=True, figsize=(20,5))

        for i, name in enumerate(datasets.keys()):
            # then plot scatter residuals
            df = datasets[name]
            h = sns.scatterplot(data=df, x='rel_R', y='VSdiff',
                s=2, hue='MHI', palette='icefire', ax=axes[i])
            h.axhline(y=0, color='black', linestyle='dashed')

        # then plot the CDF for picking the right level
        fig, axes = plt.subplots(1, 1, figsize=(20,5))
        dfs = []
        for label, df in datasets.items():
            df = df[['VSdiffabs']].copy()
            df['set'] = label
            dfs.append(df)
        
        sns.ecdfplot(data=pd.concat(dfs, ignore_index=True), x='VSdiffabs', hue='set', linestyle='dotted', ax=axes)


    def plot_rar(self, kind=0, idens=None, query_key=None, title=None, line=[1,6], velocity=False, color=None, label=None, axis=None):
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
        fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(10,10*height))

        # plot filter queries on each row
        for row, name in enumerate(datakeys):
            df = datasets[name]
            axrow = axes[row] if height > 1 else axes

            # plot references on each column
            for col, iden in enumerate(idens):
                if axis is None:
                    ax = axrow[col] if len(idens) > 1 else axrow
                else:
                    ax = axis

                if velocity:
                    x, y = '%sbar' % iden, 'Vobs'
                    lx, ly = np.log10(df[x]), np.log10(df[y])
                    ylabel = 'Observed Velocity'
                    xlabel = '%s Velocity' % self.iden_labels[iden]
                else:
                    x, y = '%sgbar' % iden, 'gobs'
                    lx, ly = np.log10(df[x]), np.log10(df[y])
                    ylabel = 'Log of Observed g'
                    xlabel = 'Log of %s g' % self.iden_labels[iden]
                
                # rel_R coloured scatter
                if kind == 0:
                    g = sns.scatterplot(data=df, x=lx, y=ly,
                        alpha=1.0, s=2, hue=np.log10(df['mhi_R']), palette='icefire', ax=ax)
                    
                # density
                elif kind == 1:
                    g = sns.scatterplot(x=lx, y=ly,
                        color='black', s=3, alpha=0.5, ax=ax)
                    sns.histplot(x=lx, y=ly,
                        bins=30, pthresh=.01, cmap="mako_r", alpha=0.6, ax=ax)
                    sns.kdeplot(x=lx, y=ly,
                        levels=4, color="w", linewidths=2, ax=ax)

                # reg
                elif kind == 2:
                    g = sns.regplot(x=lx, y=ly, scatter=False, ax=ax)

                # cleaner density
                elif kind == 3:
                    g = sns.histplot(x=lx, y=ly,
                        bins=30, cmap="Blues", alpha=1.0, ax=ax)
                    g = sns.regplot(x=lx, y=ly, scatter=False, ax=ax, color='red')
                
                # for overlay plots
                elif kind == 4:
                    g = sns.scatterplot(data=df, x=lx, y=ly,
                        alpha=1.0, s=2, color=color, label=label, ax=ax)

                # title (dataset), reference line, labels
                if col == 0:
                    g.set(title=title if title else name)

                sns.lineplot(x=line, y=line, color='grey', ax=ax, linestyle='dotted')
                g.set(xlabel=xlabel, ylabel=ylabel)
        return g
        
    def plot_rars(self, *args, **kwargs):
        """ Plots all the rar plots. Similar to a QQ plot """

        for i in range(4):
            self.plot_rar(kind=i, *args, **kwargs)


    def residual(self, df=None, resid='Sgbar', ax=None, plot=True, **kwargs):
        """ Plots a specific log residual """
        if df is None: df = self.dataframe
        
        y = np.log10(df['gobs']/df[resid])
        x = np.log10(df[resid])
        reg = sp.stats.linregress(x, y)

        if plot:
            g = sns.histplot(x=x, y=y, bins=50, pthresh=.01, cmap="Blues", ax=ax)
            sns.lineplot(x=x, y=reg.slope*x+reg.intercept, color='red', linestyle='dashed', ax=ax)
            g.axhline(y=0, color='grey', linestyle='dotted')
            g.set(ylabel='Residuals [dex]', **kwargs)
            return g
        else:
            return reg

    def residual_hist(self, query_key=None, resid='Sgbar', bins=100, color=None, label=None):
        """ Plots the histogram of the residual """
        df = self.datasets()[query_key] if query_key else self.dataframe
        data = np.log10(df['gobs']/(df[resid]))
        g = sns.histplot(data, bins=bins, color=color, label=label)
        g.set(xlabel='Residuals [dex]', ylabel='Measurements')
        return g, data

    def residual_hists(self):
        """ Plots all the histograms for all query keys on a single plot """
        outputs = {}
        for k, v in self.queries_strs.items():
            outputs[k] = self.residual_hist(k, color=COLORS[k], label=k)
        return outputs

    def plot_comparison(self, compare=None, count=None, profiles=False, sharex=False):
        """
        Plots the velocity graphs of individual galaxies.
        :compare: takes another Result object to compare adj values against
        :count: is total number to plot
        :profiles: _False_ plots the profile graphs of each galaxy if _True_
        :sharex: _True_ does what it says on the tin
        """
        def plot_sim(df, ax, idens):
            for key, color in COLOR_SCHEME.items():
                g = sns.scatterplot(data=df, x='R', y='V%s' % key, ax=ax, color=color, label='V%s' % key)
                if key == 'obs':
                    g.errorbar(df['R'], df['Vobs'], yerr=df['e_Vobs'], ecolor=color, fmt='.k')
                    sns.scatterplot(data=df, x='R', y='Vbar', ax=ax, color='grey', label='Vbar')
                    sns.lineplot(data=df, x='R', y='Sbar', ax=ax, color='grey', label='Sbar')
                    if 'Pbar' in df:
                        sns.lineplot(data=df, x='R', y='Pbar', ax=ax, color=color, label='Pbar')
                else:
                    if 'S%s' % key in df.columns:
                        sns.lineplot(data=df, x='R', y='S%s' % key, ax=ax, color=color, label='S%s' % key)
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
    
    def apply_prediction(self, params):
        # once you've done the adjustments via inputting it into a Results object
        # this will update the mass ratios, R etc for us
        # so can safely use here
        self.dataframe['Fnewton'] *= params['gamma']
        self.dataframe['Sgbar'] *= params['gamma']
        self.dataframe['Sbar'] = velocity(self.dataframe['R'], self.dataframe['Sgbar'])
