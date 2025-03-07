import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
from functools import cached_property

from models.sparc.result import Result
from references.sparc import adjustment_df


class Tune:
    UNIVERSE = ["kappa", "nu"]
    GALAXY = ["Inc", "D", "Ybul", "Ydisk", "Ygas", "beta"]

    def __init__(self, model, name=None, null_function=None):
        self.name = name
        self.model = model
        self.null_function = null_function
        self.uni = None
        self.adjs = None
        self.result = None

    @cached_property
    def params_galaxy(self):
        return [x for x in self.GALAXY if hasattr(self.model, x)]

    @cached_property
    def params_uni(self):
        return [x for x in self.UNIVERSE if hasattr(self.model, x)]

    @cached_property
    def MAP(self):
        """Quick find MAP method"""
        with self.model:
            return pm.find_MAP()

    @property
    def trace(self):
        if hasattr(self, "_trace"):
            return self._trace
        else:
            return self.sample()

    def az_summary(self):
        return az.summary(self.trace, fmt="long")

    def adjustment_MAP(self):
        """Same as adjustment_df but uses
        the fast MAP params rather than
        the full posterior mean values"""
        data = {"Galaxy": self.model.coords["Galaxy"]}
        for p in self.params_galaxy:
            data[p] = self.MAP[p]
            data["e_%s" % p] = data[p] * 0.1  # dummy 10%
        df = pd.DataFrame(data)
        df["Source"] = self.name
        return df

    def adjustment_FULL(self):
        """The per galaxy settings to create a new
        Result set for Inc, Yetcs"""
        if self.params_galaxy:
            summary = self.az_summary()
            adjustments = []
            for galaxy, gdf in summary.groupby("Galaxy"):
                data = {"Galaxy": galaxy, "Source": self.name}
                for param in self.params_galaxy:
                    data[param] = gdf[param]["mean"]
                    data["e_%s" % param] = gdf[param]["sd"]
                adjustments.append(data)
            return pd.DataFrame(adjustments)
        else:
            return None

    def params(self, fast=False):
        if fast:
            adjs = self.adjustment_MAP()
            for p in self.params_uni:
                adjs[p] = self.MAP[p]
            self.adjs = adjs
            self.uni = dict([(p, self.MAP[p]) for p in self.params_uni])
        else:
            self.adjs = self.adjustment_FULL()
            mean = self.az_summary().reset_index().query('index=="mean"')
            self.uni = dict([(p, mean[p].mean()) for p in self.params_uni])
        return self.adjs, self.uni

    def sample(self, draws=500, cores=4, tune=None):
        """Samples the model"""
        if tune is None:
            tune = draws * 2
        with self.model:
            self._trace = pm.sample(
                tune=tune,
                draws=draws,
                cores=cores,
                return_inferencedata=True,
                target_accept=0.9,
                start=self.MAP,
            )
        return self._trace

    def plot_posterior(self, galaxy=False):
        if galaxy:
            az.plot_posterior(self.trace)
        else:
            az.plot_posterior(self.trace, var_names=self.params_uni)

    def Result(self, fast=False, *args, **kwargs):
        """Generates a result object, using a fast MAP or default full sample"""

        adjs, uni = self.params(fast)
        # saves last generated result for convience
        self.result = Result(adjustments=adjs, *args, **kwargs)

        return self.result

    def corner(self):
        """Plots a corner plot"""
        import corner

        corner.corner(self.trace)

    def plot_nuissance(
        self,
        source="SPARC",
        context="RAR",
        ylims={
            "Inc": (0, 100),
            "D": (-10, None),
            "Ybul": (0, 2),
            "Ydisk": (0, 2),
            "Ygas": (0, 2),
        },
        ylabels={
            "Inc": "Inclination (Degrees)",
            "D": "Distance (Mpc)",
            "Ybul": "Mass/Luminosity adjustment of Bulge",
            "Ydisk": "Mass/Luminosity adjustment of Disk",
            "Ygas": "Mass/Luminosity adjustment of Disk",
        },
        xlabel="Galaxy in order of SPARC reference parameter value",
        title="Nuisance parameters",
        label="SMOG",
        figsize=(10, 5),
    ):
        adjs, params = self.adjs, self.params_galaxy

        # clean data
        fig, axes = plt.subplots(len(params), 1, figsize=figsize)
        if len(params) == 1:
            axes = [axes]
        def_adjs = adjustment_df()
        select_params = ["Galaxy"]
        for p in params:
            select_params.append("e_%s" % p)
            select_params.append(p)

        # create single reference dataframe
        joined = (
            adjs.set_index("Galaxy")
            .join(
                def_adjs.query("Source=='%s'" % source)[select_params].set_index(
                    "Galaxy"
                ),
                rsuffix="_ref",
            )
            .reset_index()
        )
        if context:
            joined = (
                joined.set_index("Galaxy")
                .join(
                    def_adjs.query("Source=='%s'" % context)[select_params].set_index(
                        "Galaxy"
                    ),
                    rsuffix="_context",
                )
                .reset_index()
            )

        for i, p in enumerate(params):
            # sort data into ascending
            joined = joined.sort_values("%s_ref" % p)
            galaxy = joined["Galaxy"]
            adjustment = joined[p]
            sparc = joined["%s_ref" % p]
            error = joined["e_%s_ref" % p]

            # source reference
            g = sns.lineplot(
                x=galaxy, y=sparc, ax=axes[i], linestyle="dashed", color="lightgrey"
            )
            g.fill_between(
                galaxy, sparc - error * 2, sparc + error * 2, color="whitesmoke"
            )
            g.fill_between(galaxy, sparc - error, sparc + error, color="lightgrey")

            # labels
            g.set(xlabel=None, xticks=[])
            if i == 0:
                g.set(title=title)
            if p in ylabels:
                g.set(ylabel=ylabels[p])
            if p in ylims:
                g.set(ylim=ylims[p])

            # context points
            if context:
                g.errorbar(
                    x=galaxy,
                    y=joined["%s_context" % (p)],
                    yerr=joined["e_%s_context" % (p)],
                    fmt=".r",
                    label=context,
                )

            # nuisance points
            g.errorbar(
                x=galaxy, y=adjustment, yerr=adjs["e_%s" % p], fmt="ok", label=label
            )

        # only want on final
        g.set(xlabel=xlabel)
        return g
