import pymc3 as pm
import theano.tensor as tt
import pandas as pd
import numpy as np

def mcmc(df, train_null=True, train_inc=True, train_y=False, train_tau=False, train_d=False, train_epsilon=False):
    coords = {
        "Galaxy": df.Galaxy.unique(),
        "Observation": df.Vobs.index
    }
    
    # using the ref values as the initial reference points
    params = []
    if train_inc: params += ['Inc', 'e_Inc']
    if train_d: params += ['D', 'e_D']
    if train_y: params += ['Ydisk','e_Ydisk','Ybul','e_Ybul']
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
            gamma = pm.Uniform('gamma', 0.5, 300)
            alpha = pm.Uniform('alpha', 0.05, 1)
            epsilon = pm.Uniform('epsilon', 0, 20) if train_epsilon else 1
        
        # Galaxy priors
        
        # As per method of RAR paper
        if train_inc:
            DegreesNormal = pm.Bound(pm.Normal, lower=0.0, upper=90.0)
            inc = DegreesNormal('Inc', mu=reference.Inc, sigma=reference.e_Inc, dims='Galaxy')
        
        if train_d:
            DistanceNormal = pm.Bound(pm.Normal, lower=0.0)
            dist = DistanceNormal('D', mu=reference.D, sigma=reference.e_D, dims='Galaxy')
        
        if train_y:
            SurfaceNormal = pm.Bound(pm.Normal, lower=0.2, upper=1.2) # reasonable physical bounds
            Ydisk = SurfaceNormal('Ydisk', mu=reference.Ydisk, sigma=reference.e_Ydisk, dims='Galaxy')
            Ybul = SurfaceNormal('Ybul', mu=reference.Ybul, sigma=reference.e_Ybul, dims='Galaxy')
        
        if train_tau:
            tau = pm.Exponential('tau', 1, dims='Galaxy')

        # Data
        radius = pm.Data("radius", df.R, dims="Observation")
        sparc_d = pm.Data("sparc_distance", df.D, dims="Observation")
        sparc_inc = pm.Data("sparc_inc", df.Inc, dims="Observation")
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
        # adjust for nulled field
        total_null = nulled+tau[g] if train_tau else nulled
        Fprime = gamma*force/(1+(epsilon*(total_null))**alpha) if train_null else force
        # adjust r for distance
        Rprime = radius*dist[g]/sparc_d[g] if train_d else radius
        # calculate velocity
        Velocity = tt.sgn(Fprime)*(tt.abs_(Fprime*Rprime)**0.5)
        # adjust the predicition for inclination of Vobs
        conv = np.pi/180
        Predicition = Velocity*tt.sin(inc[g]*conv)/tt.sin(sparc_inc[g]*conv) if train_inc else Velocity
        
        # Define likelihood
        obs = pm.Normal("obs", mu=Predicition, sigma=df.e_Vobs, observed=df.Vobs, dims="Observation")
    
    return galaxy_model