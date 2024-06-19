import pandas as pd
import numpy as np

from models.equations import velocity, sin
from models.sparc.profile import MASS_RATIOS

kpc_to_km = 3.08567758128e16


def augment_df(sim, adf=None, R=None):
    """
    Takes the results of a single galaxy
    and augments the dataframe with sparc data
    and adjusts those values based on any tuning for that specific galaxy
    """
    # use rotmass as base df
    df = sim.profile.rotmass_df.copy()

    # append sparc table1 interesting parameters
    for k in (
        "D",
        "e_D",
        "Inc",
        "e_Inc",
        "Vflat",
        "e_Vflat",
        "Q",
        "MHI",
        "L[3.6]",
        "Reff",
    ):
        df[k] = sim.profile.sparc_dict[k]

    # set default overrides
    mrs = MASS_RATIOS.copy()
    distance, inclination = None, None
    R = df["R"]

    # then adjust them for plots once done only
    # pulling these values from adf for ease
    if adf is not None:
        # override with adjustment dataframe
        if "Ydisk" in adf:
            mrs = {
                "disk": adf.Ydisk.values[0],
                "bul": adf.Ybul.values[0],
                "gas": adf.Ygas.values[0],
            }
            # apply these to df for reference only
            # these updates can be applied here
            # as mrs defaults are fixed
            # for D & Inc the df values are needed later
            for key, value in mrs.items():
                df["Y%s" % key] = value
                ekey = "e_Y%s" % key
                if ekey in adf:
                    df[ekey] = adf[ekey].values[0]
                else:
                    df[ekey] = value * 10**0.1
        if "D" in adf:
            distance = adf.D.values[0]
        if "Inc" in adf:
            inclination = adf.Inc.values[0]
        if "nu" in adf:
            nu = adf.nu.values[0]
            df["nu"] = nu  # safe to put here as not overriding

    # store total mass for plotting hues
    df["M"] = 0
    df["M_bul"] = 1  # default for MCMC, as can't divide by 0
    for i, label in enumerate(sim.mass_labels):
        M = sim.mass_sums[i] * mrs[label] / MASS_RATIOS[label]
        df["M_%s" % label] = M
        df["M"] += M
    # for use with plotting hue
    df["log_M"] = np.log10(df["M"])

    # interp force values from simulation
    # for now using unitary mass ratios
    # so can use raw component data in mcmc
    sdf = sim.dataframe()

    # use the point to the right (slightly larger)
    # to interp value (is always closer than linear interp)
    x_right_points = sim.profile.rotmass_x(sim.space) + 1

    # when doing traditional scalar fields on multiple components
    # or when have combined them ahead of time for speed improvements
    components = sdf.groupby("component")
    F_combined = 0

    # set a default value for bulge
    # otherwise mcmc freaks out if does not exist
    if "bul" not in components.groups:
        df["F_bul"], df["Ybul"], df["e_Ybul"] = 0, 0, 0

    for label, cdf in components:
        cdf = cdf.set_index("x").loc[x_right_points]

        # we only care about the x-vector, as that's the inward force pull
        # the other vectors should be symmetrical and won't count towards velocity
        # we also use the relative R (i.e. before MCMC adjustment)
        F_x = np.interp(R, cdf["rd"], cdf["x_vec"])

        # adjust for mass ratios
        df["Y%s_adj" % label] = mrs[label] / MASS_RATIOS[label]
        F_x *= df["Y%s_adj" % label]

        # do the nu adjustment with /total_mass**nu
        if adf is not None and "nu" in adf:
            F_x /= df["M_%s" % label] ** nu

        df["F_%s" % label] = F_x
        F_combined += F_x

    # then combine components for plotting
    df["F"] = F_combined

    # kappa and beta are just massive multipliers
    # when not trying to stick to within measurement sigma
    if adf is not None:
        if "kappa" in adf:
            # kappa is per universe
            df["kappa"] = adf.kappa.values[0]
            df["F"] *= df["kappa"]
        if "beta" in adf:
            # beta is per galaxy
            df["beta"] = adf.beta.values[0]
            df["F"] *= df["beta"]

    # make distance and inclination adjustments
    # from equations 4 & 5 in
    # https://www.aanda.org/articles/aa/pdf/2018/07/aa32547-17.pdf
    if inclination is not None:
        # here we adjust the calculated Vbar rather than Vobs
        # df["Vobs"] *= sin(df["Inc"]) / sin(inclination)
        # so that the heights don't change on the plot, just the widths
        df["Inc_adj"] = sin(inclination) / sin(df["Inc"])
        df["F"] *= df["Inc_adj"]
        df["Inc"] = inclination

    # set defaults
    if distance is not None:
        df["distance_adj"] = distance / sim.profile.sparc_dict["D"]
        R = R * df["distance_adj"]  # for velocity later

        # F is calculated as F (which is actually g) g = GM/R^2
        # so need to update the R^2
        df["F"] /= df["distance_adj"] ** 2

        # update original sparc values to newly supplied versions
        df["R"] = R
        df["D"] = distance

    # Observations which compare against
    # need to adjust R but not mass etc
    df["gobs"] = df["Vobs"] ** 2 / (R * kpc_to_km)
    # S for simulated
    # Sbar from Sgbar
    df["Sgbar"] = df["F"] / (kpc_to_km)
    df["Sbar"] = velocity(R * kpc_to_km, df["Sgbar"])

    # These V values are for benchmarking against Lelli
    # so we don't bother updating against the MCMC values
    # e.g. Vbar, Vbul etc
    df["Vbar"] = np.sum([df["V%s" % c] ** 2 for c in sim.profile.labels], axis=0) ** 0.5
    df["Vgbar"] = df["Vbar"] ** 2 / (R * kpc_to_km)
    df["VSdiff"] = (df["Sgbar"] / df["Vgbar"]) - 1
    df["VSdiffabs"] = np.abs(df["VSdiff"])
    df["VSperc"] = (df["Sgbar"] - df["Vgbar"]) / df["Vgbar"]

    # additional helper variables
    df["rel_R"] = df["R"] / df["R"].max()
    df["mhi_R"] = df["R"] / df["MHI"]

    return df
