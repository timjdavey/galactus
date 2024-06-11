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
    "Everything": "R>0",
    "Quality data": "Q==1",
    "Quality simulation": "Q==1 & rel_R<0.95 & rel_R>0.15",
}

IDENS = {
    "V": "Lelli",
    "S": "Simulated",
}


class Result:
    """
    Wrapper class to do basic analysis on the whole SPARC set of results.
    """

    def __init__(
        self,
        simulations,
        adjustments=None,
        queries_strs=ANALYSIS,
        iden_labels=IDENS,
        idens=("S"),
    ):
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
        self.idens = idens
        self.adjustments = adjustments

    def datasets(self, df=None):
        """
        Automatically return the dataframe filtered
        using the `query` param
        """
        df = self.dataframe if df is None else df
        return dict([(k, df.query(q)) for k, q in self.queries_strs.items()])

    def counts(self):
        for key, data in self.datasets().items():
            print(key, len(data), len(data.groupby("Galaxy")))

    def plot_thresholds(self, everything=False):
        """
        Plots a histogram of the differences in V (sparc model) & W (simulated model)
        """
        datasets = self.datasets()

        if everything:
            # first plot the histograms
            fig, axes = plt.subplots(1, len(datasets), sharey=True, figsize=(20, 5))

            for i, name in enumerate(datasets.keys()):
                df = datasets[name]
                g = sns.histplot(df["VSdiff"], ax=axes[i])
                g.set(title=name)

            # next plot the histograms
            fig, axes = plt.subplots(
                1, len(datasets), sharex=True, sharey=True, figsize=(20, 5)
            )

            for i, name in enumerate(datasets.keys()):
                # then plot scatter residuals
                df = datasets[name]
                h = sns.scatterplot(
                    data=df,
                    x="rel_R",
                    y="VSdiff",
                    s=2,
                    hue="log_M",
                    palette="icefire",
                    ax=axes[i],
                )
                h.axhline(y=0, color="black", linestyle="dashed")

            # then plot the CDF for picking the right level
            fig, axes = plt.subplots(1, 1, figsize=(20, 5))
            dfs = []
            for label, df in datasets.items():
                df = df[["VSdiffabs"]].copy()
                df["set"] = label
                dfs.append(df)

            sns.ecdfplot(
                data=pd.concat(dfs, ignore_index=True),
                x="VSdiffabs",
                hue="set",
                linestyle="dotted",
                ax=axes,
            )
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            for i, k in enumerate(("Quality data", "Quality simulation")):
                df = datasets[k]
                sns.histplot(df["VSperc"], ax=axes[0][i]).set(
                    xlabel="Percentage difference to Lelli model",
                    title=k,
                    xlim=(-0.5, 1),
                )
                h = sns.scatterplot(
                    data=df,
                    x="rel_R",
                    y="VSperc",
                    s=2,
                    color="black",
                    ax=axes[1][i],
                )
                h.set(
                    xlabel="Relative radius (R/Rmax)",
                    ylabel="Percentage difference to Lelli model",
                    xlim=(-0.05, 1.05),
                    ylim=(-0.5, 1),
                )
                h.axhline(y=0, color="black", linestyle="dashed")

    def plot_rar(
        self,
        kind=0,
        idens=None,
        query_key=None,
        title=None,
        size=5,
        line=[-12, -8],
        velocity=False,
        label="Log($g_{pre}$) [$ms^{-2}$]",
        axis=None,
    ):
        """
        Plots various
        kind == 0 is density plot
        kind == 1 is a rel_R plot
        kind == 2 is a regression plot
        """
        datasets = self.datasets()
        datakeys = (
            [
                query_key,
            ]
            if query_key
            else datasets.keys()
        )
        if idens is None:
            idens = self.idens

        height = len(datakeys)
        width = len(idens)
        if axis is None:
            fig, axes = plt.subplots(
                height,
                width,
                sharex=True,
                sharey=True,
                figsize=(size * width, size * height),
            )
        else:
            fig, axes = None, None

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
                    x, y = "%sbar" % iden, "Vobs"
                    lx, ly = np.log10(df[x]), np.log10(df[y])
                    ylabel = "Observed Velocity [$ms^{-1}$]"
                    xlabel = "%s Velocity" % self.iden_labels[iden]
                else:
                    x, y = "%sgbar" % iden, "gobs"
                    lx, ly = np.log10(df[x]), np.log10(df[y])
                    ylabel = "Log($g_O$) [$ms^{-2}$]"
                    xlabel = label

                # M coloured scatter
                if kind == 0:
                    g = sns.scatterplot(
                        data=df,
                        x=lx,
                        y=ly,
                        alpha=1.0,
                        s=3,
                        hue=np.log10(df["M"]),
                        palette="Spectral",
                        ax=ax,
                    )

                # density
                elif kind == 1:
                    g = sns.scatterplot(
                        x=lx, y=ly, color="black", s=3, alpha=0.5, ax=ax
                    )
                    sns.histplot(
                        x=lx,
                        y=ly,
                        bins=30,
                        pthresh=0.01,
                        cmap="mako_r",
                        alpha=0.6,
                        ax=ax,
                    )
                    sns.kdeplot(x=lx, y=ly, levels=4, color="w", linewidths=2, ax=ax)

                # reg
                elif kind == 2:
                    g = sns.regplot(x=lx, y=ly, scatter=False, ax=ax)

                # cleaner density
                elif kind == 3:
                    g = sns.histplot(
                        x=lx, y=ly, bins=30, cmap="Blues", alpha=1.0, ax=ax
                    )
                    # g = sns.regplot(x=lx, y=ly, scatter=False, ax=ax, color='red')

                # title (dataset), reference line, labels
                if col == 0:
                    g.set(title=title if title else name)

                sns.lineplot(x=line, y=line, color="grey", ax=ax, linestyle="dotted")
                g.set(xlabel=xlabel, ylabel=ylabel)
        return fig if axis is None else axis

    def residual(self, df=None, resid="Sgbar", ax=None, plot=True, **kwargs):
        """Plots a specific log residual"""
        if df is None:
            df = self.dataframe

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 3))

        y = np.log10(df["gobs"] / df["Sgbar"])
        x = np.log10(df[resid])
        reg = sp.stats.linregress(x, y)

        if plot:
            g = sns.histplot(x=x, y=y, bins=50, pthresh=0.01, cmap="Blues", ax=ax)
            sns.lineplot(
                x=x,
                y=reg.slope * x + reg.intercept,
                color="red",
                linestyle="dashed",
                ax=ax,
            )
            g.axhline(y=0, color="grey", linestyle="dotted")
            g.set(ylabel="Residuals [dex]", **kwargs)
            return ax
        else:
            return reg

    def residual_stats(self, resid="Sgbar"):
        from sklearn.metrics import mean_squared_error

        stats = []
        for key, df in self.datasets().items():
            data = np.log10(df["gobs"] / (df[resid]))
            mse = mean_squared_error(data, np.zeros((len(data))))
            stats.append(
                {
                    "filter": key,
                    "mean": data.mean(),
                    "std": data.std(),
                    "count": data.count(),
                    "mse": mse,
                    "rmse": mse**0.5,
                }
            )
        return pd.DataFrame(stats)

    def residual_hist(
        self, query_key=None, resid="Sgbar", bins=100, color=None, label=None, ax=None
    ):
        """Plots the histogram of the residual"""
        df = self.datasets()[query_key] if query_key else self.dataframe
        data = np.log10(df["gobs"] / (df[resid]))
        g = sns.histplot(data, bins=bins, color=color, label=label, ax=ax)
        g.set(xlabel="Residuals [dex]", ylabel="Measurements")
        return g, data

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
                g = sns.scatterplot(
                    data=df, x="R", y="V%s" % key, ax=ax, color=color, label="V%s" % key
                )
                if key == "obs":
                    g.errorbar(
                        df["R"], df["Vobs"], yerr=df["e_Vobs"], ecolor=color, fmt=".k"
                    )
                    sns.scatterplot(
                        data=df, x="R", y="Vbar", ax=ax, color="grey", label="Vbar"
                    )
                    sns.lineplot(
                        data=df, x="R", y="Sbar", ax=ax, color="grey", label="Sbar"
                    )
                    if "Pbar" in df:
                        sns.lineplot(
                            data=df, x="R", y="Pbar", ax=ax, color=color, label="Pbar"
                        )
                else:
                    if "S%s" % key in df.columns:
                        sns.lineplot(
                            data=df,
                            x="R",
                            y="S%s" % key,
                            ax=ax,
                            color=color,
                            label="S%s" % key,
                        )
            return g

        # how many plots
        sharey = True
        columns = 1
        if compare:
            columns += 1
        if profiles:
            columns += 1
            sharey = False

        for i, group in enumerate(self.dataframe.groupby("Galaxy")):
            galaxy, gdf = group
            fig, axes = plt.subplots(
                1, columns, figsize=(20, 8), sharex=sharex, sharey=sharey
            )

            if columns == 1:
                g = plot_sim(gdf, axes, self.idens)

            if compare or profiles:
                # plot this result object on the right
                # as it's the last most thing in the
                # analysis chain
                g = plot_sim(gdf, axes[columns - 1], self.idens)

                j = 0
                if compare:
                    plot_sim(
                        compare.dataframe[compare.dataframe["Galaxy"] == galaxy],
                        axes[columns - 2],
                        compare.idens,
                    )
                if profiles:
                    self.simulations[galaxy].profile.plot(axes[0], i)

            g.set(title=galaxy)

            if count and i == count - 1:
                return

    def plot_curves(self, reference=None, cols=5):
        """Plots a mini version of all the rotation curves"""
        groups = self.dataframe.query("Q == 1").groupby("Galaxy")
        fig, axes = plt.subplots((len(groups) // cols) + 1, cols, figsize=(20, 50))

        i = 0
        for galaxy, df in groups:
            ax = axes[i // cols][i % cols]
            if reference is not None:
                # typically baryonic
                sdf = reference.dataframe.query('Galaxy=="%s"' % galaxy)
                g = sns.lineplot(x=sdf["R"], y=sdf["Sbar"], ax=ax)

            # sims
            g = sns.lineplot(x=df["R"], y=df["Sbar"], ax=ax)
            # observations
            g.errorbar(df["R"], df["Vobs"], yerr=df["e_Vobs"], fmt=".k")
            # labels
            g.set(title=galaxy, xlabel=None, ylabel=None)
            i += 1

        return fig
