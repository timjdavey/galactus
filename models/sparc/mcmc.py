import pymc3 as pm
import theano.tensor as at
import pandas as pd
import numpy as np
from models.sparc.profile import MASS_RATIOS

TIGHT = {"Inc": 1, "D": 1, "Ydisk": 1, "Ybul": 1}


def mcmc(
    df,
    mode,
    tight=None,
    normals=True,
):
    df = df.copy()

    coords = {"Galaxy": df.Galaxy.unique(), "Observation": df.Vobs.index}

    if tight:
        new_tight = TIGHT.copy()
        for k, v in tight.items():
            new_tight[k] = v
        tight = new_tight
    else:
        tight = TIGHT

    # using the ref values as the initial reference points
    params = ["Inc", "e_Inc", "D", "e_D", "M_bul", "M_disk", "M_gas"]
    reference = df.groupby("Galaxy").mean()[params]

    # for g param (i.e. galaxy id)
    # need to build from given df, to make sure index & orders match up etc
    if "gidx" not in df.columns:
        uniqs = df.Galaxy.unique()
        indx = pd.DataFrame({"Galaxy": uniqs, "gidx": np.arange(len(uniqs))})
        df = df.join(indx.set_index("Galaxy"), on="Galaxy")

    with pm.Model(coords=coords) as galaxy_model:
        # Data
        radius = pm.Data("radius", df.R, dims="Observation")
        galaxy = pm.Data("g", df.gidx, dims="Observation")

        # the calculated forces
        force_bul = pm.Data("force_bul", df.F_bul, dims="Observation")
        force_disk = pm.Data("force_disk", df.F_disk, dims="Observation")
        force_gas = pm.Data("force_gas", df.F_gas, dims="Observation")

        # these masses are per Galaxy, but easy to say Observation as from dataframe
        mass_bul = pm.Data("mass_bul", reference.M_bul, dims="Galaxy")
        mass_disk = pm.Data("mass_disk", reference.M_disk, dims="Galaxy")
        mass_gas = pm.Data("mass_gas", reference.M_gas, dims="Galaxy")

        # when adjusting for inc and d, need to normalise by sparc values
        sparc_inc = pm.Data("sparc_inc", reference.Inc, dims="Galaxy")
        sparc_d = pm.Data("sparc_d", reference.D, dims="Galaxy")

        # all sparc simulations are done with these implicit values
        # of MASS_RATIOS and distances / inclications
        # so when calculating adjustments, need to do it in relation to these
        force_components = (force_bul, force_disk, force_gas)
        mass_components = (mass_bul[galaxy], mass_disk[galaxy], mass_gas[galaxy])

        # Universal priors
        if "k" in mode:
            # kappa adjusts universal multipler
            kappa = pm.Uniform("kappa", 0.0001, 10e20)

        if "n" in mode:
            # power to which total mass is divided
            nu = pm.Uniform("nu", 0.1, 1)

        # Galaxy priors
        if "b" in mode:
            # magnitude on a per galaxy basis
            beta = pm.Uniform("beta", 0.0001, 10e20, dims="Galaxy")

        if "i" in mode:
            if normals:
                # As per method of RAR paper
                DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
                inc = DegreesNormal(
                    "Inc",
                    mu=reference.Inc,
                    sigma=reference.e_Inc / tight["Inc"],
                    dims="Galaxy",
                )
            else:
                inc = pm.Uniform("Inc", 0.0, 90.0, dims="Galaxy")

        if "d" in mode:
            # trains distance and R
            DistanceNormal = pm.Bound(pm.Normal, lower=0.1)
            dist = DistanceNormal(
                "D", mu=reference.D, sigma=reference.e_D / tight["D"], dims="Galaxy"
            )

        if "y" in mode:
            if normals:
                # trains surface mass ratios
                SurfaceNormal = pm.Bound(
                    pm.Normal, lower=0.01, upper=1.5
                )  # reasonable physical bounds
                astro_scatter = 10**0.1  # from Li's rar paper
                Ydisk = SurfaceNormal(
                    "Ydisk",
                    mu=MASS_RATIOS["disk"],
                    sigma=MASS_RATIOS["disk"] * astro_scatter,
                    dims="Galaxy",
                )
                Ybul = SurfaceNormal(
                    "Ybul",
                    mu=MASS_RATIOS["bul"],
                    sigma=MASS_RATIOS["bul"] * astro_scatter,
                    dims="Galaxy",
                )
            else:
                Ybul = pm.Uniform("Ybul", 0.01, 1.5, dims="Galaxy")
                Ydisk = pm.Uniform("Ydisk", 0.01, 1.5, dims="Galaxy")
            ratio_components = (
                Ybul[galaxy] / MASS_RATIOS["bul"],
                Ydisk[galaxy] / MASS_RATIOS["disk"],
                1,
            )

        else:
            ratio_components = (1, 1, 1)

        Force = 0
        if "n" in mode:
            for i, f in enumerate(force_components):
                ratio = ratio_components[i]
                m = mass_components[i]
                Force += f * ratio / (m * ratio) ** nu
        else:
            for i, f in enumerate(force_components):
                Force += f * ratio_components[i]

        if "k" in mode:
            Force *= kappa

        if "b" in mode:
            Force *= beta[galaxy]

        if "d" in mode:
            ratio_distance = dist[galaxy] / sparc_d[galaxy]
            Force /= ratio_distance**2

        # calculate velocity
        # don't adjust the radius in velocity
        # as this is cancelled when working out the velocity of the observed
        Velocity = at.sgn(Force) * (at.abs_(Force * radius) ** 0.5)

        # adjust the prediction for inclination of Vobs
        # where this is reversed from the SPARC paper
        # as SPARC paper adjusts the observed velocity
        if "i" in mode:
            conv = np.pi / 180
            Velocity *= at.sin(inc[galaxy] * conv) / at.sin(sparc_inc[galaxy] * conv)

        # Define likelihood
        obs = pm.Normal(
            "obs", mu=Velocity, sigma=df.e_Vobs, observed=df.Vobs, dims="Observation"
        )

    return galaxy_model
