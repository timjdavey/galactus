import pymc3 as pm
import numpy as np
import pandas as pd



def universe_model(df):
    coords = {
        "galaxy": df.ID.unique(),
        "observation": df.Vobs.index
    }
    
    with pm.Model(coords=coords) as universe_model:
        
        # Universal priors
        gamma = pm.Uniform('gamma', 0, 50)
        alpha = pm.Uniform('alpha', 0, 0.5)
        epsilon = pm.Uniform('epsilon', -20, 20)
        sigma = pm.HalfCauchy("sigma", beta=5) # observational error
    
        # Data
        force = pm.Data("force", df.Fnewton, dims="observation")
        radius = pm.Data("radius", df.R, dims="observation")
        nulled = pm.Data("nulled", df.Fnulled, dims="observation")
        
        # Prediction model
        ftd = force*(gamma/(1+(epsilon*nulled)**alpha))
        vtd = np.sqrt(ftd*radius)
        
        # Define likelihood
        obs = pm.Normal("obs", mu=vtd, sigma=sigma, observed=df.Vobs, dims="observation")

    return universe_model


def galaxy_model(df, use_tau=False):
    coords = {
        "galaxy": df.ID.unique(),
        "observation": df.Vobs.index
    }
    
    with pm.Model(coords=coords) as galaxy_model:
        
        # Universal priors
        gamma = pm.Uniform('gamma', 0, 100)
        alpha = pm.Uniform('alpha', 0, 1)
        
        # Error
        sigma = pm.HalfCauchy("sigma", beta=5, dims="observation")
        
        # Galaxy priors
        DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
        inc = DegreesNormal('Inc', mu=sparc['Inc'], sigma=sparc['e_Inc'], dims='galaxy')
        PositiveNormal = pm.Bound(pm.Normal, lower=0.0)
        dist = PositiveNormal('D', mu=sparc['D'], sigma=sparc['e_D'], dims='galaxy')
        if use_tau: tau = pm.Exponential('tau', 1)
        else: tau = 0
    
        # Data
        force = pm.Data("force", df.Fnewton, dims="observation")
        radius = pm.Data("radius", df.R, dims="observation")
        nulled = pm.Data("nulled", df.Fnulled, dims="observation")
        sparc_d = pm.Data("sparc_distance", df.D, dims="observation")
        sparc_inc = pm.Data("sparc_inc", df.Inc, dims="observation")
        g = pm.Data("g", df.gidx, dims="observation")
        
        # Prediction model
        # adjust for nulled field
        ftd = force*(gamma/(1+(nulled)**alpha))
        # adjust r for distance when calc V
        vtd = np.sqrt(ftd*radius*dist[g]/sparc_d[g])
        # adjust the predicition for inclination of Vobs
        vpred = vtd*np.sin(inc[g]*np.pi/180)/np.sin(sparc_inc[g]*np.pi/180)
        
        # Define likelihood
        obs = pm.Normal("obs", mu=vpred, sigma=sigma, observed=df.Vobs, dims="observation")

    return galaxy_model


