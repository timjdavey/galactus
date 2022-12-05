import pymc3 as pm
import theano.tensor as at
import pandas as pd
import numpy as np

TIGHT = {'Inc': 1.3, 'D': 3, 'Ymass': 15}

def mcmc(df, train_g=False, train_y=True, y_uni=False, train_inc=True, train_d=True, tight=None):

    df = df.copy()

    coords = {
        "Galaxy": df.Galaxy.unique(),
        "Observation": df.Vobs.index
    }
    
    if tight:
        TIGHT.update(tight)

    # using the ref values as the initial reference points
    params = ['Inc', 'e_Inc', 'D', 'e_D']
    reference = df.groupby('Galaxy').mean()[params]
    
    # for g param (i.e. galaxy id)
    # need to build from given df, to make sure index & orders match up etc
    if 'gidx' not in df.columns:
        uniqs = df.Galaxy.unique()
        indx = pd.DataFrame({'Galaxy':uniqs, 'gidx':np.arange(len(uniqs))})
        df = df.join(indx.set_index('Galaxy'), on='Galaxy')
    
    with pm.Model(coords=coords) as galaxy_model:
        
        # Universal priors
        if train_g:
            # gamma adjusts the k factor
            gamma = pm.Uniform('gamma', 0.5, 2000)
        else:
            gamma = 1
        
        # Galaxy priors
        if train_inc:
            # As per method of RAR paper
            DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
            inc = DegreesNormal('Inc', mu=reference.Inc, sigma=reference.e_Inc/TIGHT['Inc'], dims='Galaxy')

        if train_d:
            DistanceNormal = pm.Bound(pm.Normal, lower=0.1)
            dist = DistanceNormal('D', mu=reference.D, sigma=reference.e_D/TIGHT['D'], dims='Galaxy')

        if train_y:
            if y_uni:
                Ymass = pm.Uniform('Ymass', 0.1, 200, dims='Galaxy')
            else:
                Ymass = pm.Normal('Ymass', mu=1.0, sigma=1/TIGHT['Ymass'], dims='Galaxy')

        # Data
        sparc_inc = pm.Data("sparc_inc", reference.Inc, dims="Galaxy")
        sparc_d = pm.Data("sparc_d", reference.D, dims="Galaxy")

        radius = pm.Data("radius", df.R, dims="Observation")
        g = pm.Data("g", df.gidx, dims="Observation")
        force = pm.Data("force", df.F, dims="Observation")
        
        Fprime = force*gamma
        if train_y:
            Fprime *= Ymass[g]

        if train_d:
            Rprime = radius*dist[g]/sparc_d[g]
        else:
            Rprime = radius

        # calculate velocity
        Velocity = at.sgn(Fprime)*(at.abs_(Fprime*Rprime)**0.5)

        # adjust the predicition for inclination of Vobs
        if train_inc:
            conv = np.pi/180
            Vpred = Velocity*at.sin(inc[g]*conv)/at.sin(sparc_inc[g]*conv)
        else:
            Vpred = Velocity
        
        # Define likelihood
        obs = pm.Normal("obs", mu=Vpred, sigma=df.e_Vobs, observed=df.Vobs, dims="Observation")
    
    return galaxy_model





