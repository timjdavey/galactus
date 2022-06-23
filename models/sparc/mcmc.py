import pymc3 as pm
import theano.tensor as at
import pandas as pd
import numpy as np
from models.equations import null_gravity

TIGHT = {'Inc': 1, 'D': 1, 'Ydisk': 20, 'Ybul': 20}

def mcmc(df, 
    velocity=False, null_func=null_gravity,
    train_null=True, train_tau=False, epsilon=0,
    train_inc=False, train_d=False, train_y=False, tight=None):
    coords = {
        "Galaxy": df.Galaxy.unique(),
        "Observation": df.Vobs.index
    }
    
    if tight:
        TIGHT.update(tight)

    # using the ref values as the initial reference points
    params = ['Inc', 'e_Inc', 'D', 'e_D', 'Ydisk','e_Ydisk','Ybul','e_Ybul']
    reference = df.groupby('Galaxy').mean()[params]
    
    # for g param
    # need to build from given df, to make sure index & orders match up etc
    if 'gidx' not in df.columns:
        uniqs = df.Galaxy.unique()
        indx = pd.DataFrame({'Galaxy':uniqs, 'gidx':np.arange(len(uniqs))})
        df = df.join(indx.set_index('Galaxy'), on='Galaxy')
    
    with pm.Model(coords=coords) as galaxy_model:
        
        # Universal priors
        if train_null:
            gamma = pm.Uniform('gamma', 0.5, 200)
            alpha = pm.Uniform('alpha', 0.05, 2)
        else:
            gamma = 1
            alpha = 1

        if epsilon:
            epsilon = pm.Normal('epsilon', mu=epsilon, sigma=epsilon/2)
        
        # Galaxy priors
        if train_inc:
            # As per method of RAR paper
            DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
            inc = DegreesNormal('Inc', mu=reference.Inc, sigma=reference.e_Inc/TIGHT['Inc'], dims='Galaxy')

        if train_d:
            DistanceNormal = pm.Bound(pm.Normal, lower=0.1)
            dist = DistanceNormal('D', mu=reference.D, sigma=reference.e_D/TIGHT['D'], dims='Galaxy')

        if train_y:
            SurfaceNormal = pm.Bound(pm.Normal, lower=0.2, upper=1.5) # reasonable physical bounds
            Ydisk = SurfaceNormal('Ydisk', mu=reference.Ydisk, sigma=reference.e_Ydisk/TIGHT['Ydisk'], dims='Galaxy')
            Ybul = SurfaceNormal('Ybul', mu=reference.Ybul, sigma=reference.e_Ybul/TIGHT['Ybul'], dims='Galaxy')
        
        if train_tau:
            tau = pm.Exponential('tau', 1, dims='Galaxy')

        # Data
        sparc_inc = pm.Data("sparc_inc", reference.Inc, dims="Galaxy")
        sparc_d = pm.Data("sparc_d", reference.D, dims="Galaxy")

        radius = pm.Data("radius", df.R, dims="Observation")
        g = pm.Data("g", df.gidx, dims="Observation")
        
        if train_y:
            comps = ['disk', 'bul', 'gas']
            force_comp = dict([(c, pm.Data("force_%s" % c, df['Fnewton_%s' % c].fillna(0), dims="Observation")) for c in comps])
            nulled_comp = dict([(c, pm.Data("nulled_%s" % c, df['Fnulled_%s' % c].fillna(0), dims="Observation")) for c in comps])            
            
            force = force_comp['disk']*Ydisk[g] + force_comp['bul']*Ybul[g] + force_comp['gas']
            nulled = nulled_comp['disk']*Ydisk[g] + nulled_comp['bul']*Ybul[g] + nulled_comp['gas']
        else:
            force = pm.Data("force", df.Fnewton, dims="Observation")
            nulled = pm.Data("nulled", df.Fnulled, dims="Observation")
        
        # Prediction model
        if train_tau:
            total_null = nulled+tau[g]
        else:
            total_null = nulled

        Fprime = null_func(force, total_null, gamma, alpha, epsilon)

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
        if velocity or train_d:
            obs = pm.Normal("obs", mu=Vpred, sigma=df.e_Vobs, observed=df.Vobs, dims="Observation")
        else:
            # seems to give flatter residuals
            # and tighter 
            Fpred = (Vpred**2)/radius
            Fobs = (df.Vobs**2)/df.R
            obs = pm.Normal("obs", mu=Fpred, sigma=df.e_Vobs**2, observed=Fobs, dims="Observation")
        
    
    return galaxy_model




def scalar(df, 
    train_y=False, train_inc=False, train_d=False, tight=None):

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
    
    # for g param
    # need to build from given df, to make sure index & orders match up etc
    if 'gidx' not in df.columns:
        uniqs = df.Galaxy.unique()
        indx = pd.DataFrame({'Galaxy':uniqs, 'gidx':np.arange(len(uniqs))})
        df = df.join(indx.set_index('Galaxy'), on='Galaxy')
    
    with pm.Model(coords=coords) as galaxy_model:
        
        # Universal priors
        gamma = pm.Uniform('gamma', 0.5, 200)
        
        # Galaxy priors
        if train_inc:
            # As per method of RAR paper
            DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
            inc = DegreesNormal('Inc', mu=reference.Inc, sigma=reference.e_Inc/TIGHT['Inc'], dims='Galaxy')

        if train_d:
            DistanceNormal = pm.Bound(pm.Normal, lower=0.1)
            dist = DistanceNormal('D', mu=reference.D, sigma=reference.e_D/TIGHT['D'], dims='Galaxy')

        if train_y:
            Ymass = pm.Normal('Ymass', mu=1.0, sigma=1/TIGHT['Ymass'], dims='Galaxy')

        # Data
        sparc_inc = pm.Data("sparc_inc", reference.Inc, dims="Galaxy")
        sparc_d = pm.Data("sparc_d", reference.D, dims="Galaxy")

        radius = pm.Data("radius", df.R, dims="Observation")
        g = pm.Data("g", df.gidx, dims="Observation")
        force = pm.Data("force", df.Fnewton, dims="Observation")
        
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






